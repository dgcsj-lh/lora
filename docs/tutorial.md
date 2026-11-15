# SD1.5 LoRA 手写微调教程（从零到能训练）

> 本教程是 `docs/ROADMAP.md` 阶段 3 的详细展开。目标不是"能跑起来"，而是**每一行代码你都知道它在做什么、为什么这么写**。
>
> 配套材料：
> - 完整参考实现：`docs/handwritten_lora_reference.py`（约 300 行，先自己写，卡住再对照）
> - 学习路线：`docs/ROADMAP.md`
> - 现有加载脚本：`src/scripts/loder_lora.py`（训练后用它加载 LoRA 验证效果）
>
> 练习文件统一写在 `src/training/` 下，按 `01` → `05` 顺序完成，**每个文件跑通了再做下一个**。

---

## 0. 怎么用这份教程

1. 顺序读 第 1 节（三个核心概念），不要求背公式，但要知道"模型在干什么"。
2. 第 2、3 节确认环境和数据就绪。
3. 第 4 节是核心：5 个练习文件。**建议你先自己写，卡住超过半小时再来看下面的代码**；看完代码合上，再自己写一遍。
4. 每写完一个文件，用文件顶部注释里的命令运行，按"完成标准"判断是否通过。
5. 全部跑完后，用第 5 节的方法验证，再用第 6、7 节调参、排错。

---

## 1. 先理解三件事

### 1.1 扩散模型在做什么

直觉版：**把一张图不断加噪直到变成纯噪声，再训练一个网络学会"从噪声还原回图"的每一步。**

- **前向过程（加噪，训练时用）**：拿一张干净图 `x_0`，随机选一个时间步 `t`，按公式混入随机噪声 `ε`，得到"半糊"的图 `x_t`。`t` 越大，噪声越多。
- **反向过程（去噪，推理时用）**：网络输入 `(x_t, t)`，预测"这一步该去掉多少噪声"（其实等价于预测 `ε`）。
- **训练目标**：让网络预测的噪声 `ε̂` 尽可能接近真实的 `ε`：

  ```
  loss = MSE(ε̂, ε)
  ```

  就这么简单。训练时不需要走完整采样，只需要随机选 `t`、加一次噪、预测一次、算一次 MSE、反传一次。

- **采样（生成）**：从纯噪声 `x_T` 开始，一步步让网络去噪，几十步后得到干净图。

**给一张图上的时间步 `t`**：DDPM 用 `t ∈ [0, 999]`（SD1.5 的 `num_train_timesteps = 1000`），`t` 是一个整数，输入网络前会编码成 embedding。`t=0` 是干净图，`t=999` 是纯噪声。

### 1.2 SD1.5 的三块积木

SD1.5 是一个 `pipeline`，由下面几个独立模型拼成。**训练时只动 UNet，其他全部冻结。**

| 组件 | 作用 | 训练时状态 |
|------|------|-----------|
| **VAE** | 把 512×512×3 的像素图压缩成 64×64×4 的**潜变量**，采样时再还原 | 冻结，只做编码/解码 |
| **CLIP Text Encoder** | 把 prompt 变成 77×768 的条件向量，指导生成内容 | 冻结 |
| **UNet** | 核心：输入加噪潜变量 + 时间步 + 文本条件，输出预测噪声 | **训练它（LoRA 就装在这里）** |
| **Scheduler** | 提供加噪公式（训练）和去噪迭代规则（采样） | 无参数 |

**为什么训练在潜空间而不是像素空间？**

- 512×512×3 ≈ 78 万维 → 64×64×4 ≈ 1.6 万维，**数据量缩小约 48 倍**。
- 潜空间是 VAE 学出来的"更紧凑的表示"，去掉了像素里的高频冗余（草叶纹理、噪点），UNet 可以专心学语义。
- 结果就是：8GB 显存也能训。

**潜变量的缩放**：VAE 编码出来的潜变量分布不标准，SD 约定乘一个常数 `0.18215` 让它更接近标准正态分布。这是 SD1.5 的固定值，**编码时乘上，解码前除回来**，别漏。

### 1.3 LoRA 到底改了什么

LoRA（Low-Rank Adaptation）的核心假设：微调大模型时，权重变化量 `ΔW` 不需要是满秩的大矩阵，可以用两个小矩阵的乘积近似：

```
W' = W + ΔW ≈ W + B·A
```

其中 `W` 是原权重（`out×in`），`A` 是 `r×in`，`B` 是 `out×r`，`r`（rank）很小（比如 8~32）。

- **参数量对比**：一个 `768×768` 的 attention 投影层有 59 万参数；用 rank=16 的 LoRA 只需要 `768×16 + 16×768 ≈ 2.4 万` 参数，**少了约 96%**。
- **训练时**：`W` 冻结（不计算梯度），只有 `A`、`B` 是可训练参数。
- **缩放系数**：实际增量是 `(alpha / r) · B·A`。`alpha` 是另一个超参，常用 `alpha = r` 或 `2r`。`alpha/r` 就是增量整体的放大倍数。
- **初始化关键**：`A` 用随机小值初始化，`B` 初始化为 **0**。这样训练开始的那一刻 `ΔW = 0`，输出和原模型完全一样，训练再慢慢"长"出增量。

**LoRA 装在 UNet 的哪些层？** SD1.5 的 UNet 里每个 attention 模块有 4 个投影层：

- `to_q` / `to_k` / `to_v`：把输入投影成 Query / Key / Value
- `to_out`：把 attention 输出再投影一次

这 4 个层覆盖了 self-attention 和 cross-attention 的全部投影，是 LoRA 最常用的目标层。本教程只挂这 4 个，进阶再加 FFN 层。

---

## 2. 环境确认（你基本已就绪）

运行下面命令确认能加载模型：

```bash
cd /home/lipenghui/lora
.venv/bin/python src/scripts/loder_lora.py --no-generate
```

如果能看到 VAE / Text Encoder / UNet 的参数统计，说明模型加载正常。

**还需要确认 `torchvision` 已安装**（写 Dataset 要用到 `transforms`）：

```bash
.venv/bin/python -c "import torchvision; print(torchvision.__version__)"
```

若报 `ModuleNotFoundError`：

```bash
uv add torchvision
```

---

## 3. 数据准备

最小可用数据集：**20~50 张图 + 每张图一个同名 `.txt` 标注**，目录结构：

```
src/dataset/
└── my_images/
    ├── photo_01.jpg
    ├── photo_01.txt      # 内容："a photo of my cat, sitting"
    ├── photo_02.png
    ├── photo_02.txt
    └── ...
```

**标注（caption）怎么打：**

1. 自动打标：用 WD14 tagger 批量生成，再人工修正（删掉错误 tag、统一主角描述）。
2. 建议给目标主体一个**触发词**（trigger word），比如 `mycat`，所有 caption 里都带上它。训练后你用 `mycat` 这个 prompt 就能召唤出你的主体。
3. caption 要"描述图里有什么"，不要写"a picture of"这种废话，也不要只写一个词（信息量太少）。

**（可选）正则化图 / prior preservation**：如果你的主体是"一只具体的猫"，为了防止模型忘记"猫"的一般概念（语言漂移），可以准备 10~20 张**同类的通用猫图**，caption 写通用描述（如 `a cat`），和目标图按 1:1~1:3 混入训练。本教程的代码不需要特殊处理，把它们放同一个目录即可。

---

## 4. 五个练习文件（核心）

### 4.0 公共约定（所有文件都会用到）

```python
from pathlib import Path

# 本地 SD1.5 模型目录：src/training/ 下的脚本往上两级到项目根
MODEL_DIR = Path(__file__).resolve().parents[2] / "src/models/stable-diffusion-v1-5"

VAE_SCALE = 0.18215      # VAE 潜变量缩放因子（编码乘上、解码除回）
RESOLUTION = 512         # 训练分辨率
MAX_TOKEN_LEN = 77       # CLIP 固定 token 长度
```

- **`parents[2]` 是为什么**：`__file__` 是 `src/training/01_dataset.py`，`parents[0]` 是 `src/training/`，`parents[1]` 是 `src/`，`parents[2]` 才是项目根 `/home/lipenghui/lora`。用 `resolve()` 保证是绝对路径，不受运行目录影响。

- **dtype 策略**（贯穿全部代码）：
  - VAE / CLIP / UNet 的**冻结权重用 fp16**（省显存、计算快）
  - **LoRA 可训练权重用 fp32**（梯度更新更稳定，`torch.autocast` 会自动混合精度）
  - 算 loss 时转回 fp32（数值稳定）

---

### 4.1 `01_dataset.py` —— 图片目录 → Dataset / DataLoader

**目的**：把图片和标注组织成训练循环能直接消费的 batch。

**完成标准**：运行后打印出 `[2, 3, 512, 512]` 的张量，数值范围约 `[-1, 1]`，caption 正确读出。

```python
"""01_dataset.py —— 图片目录 → Dataset / DataLoader
运行：.venv/bin/python src/training/01_dataset.py
"""
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATA_DIR = Path("src/dataset/my_images")
RESOLUTION = 512


class ImageCaptionDataset(Dataset):

    def __init__(self, data_dir: Path, resolution: int = RESOLUTION):
        # 只收常见图片格式，排序保证顺序稳定
        self.items = sorted(
            p for p in Path(data_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not self.items:
            raise FileNotFoundError(f"{data_dir} 里没有图片")

        self.transform = transforms.Compose([
            # ① 短边缩放到 512（LANCZOS 质量最好）
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            # ② 中心裁剪成 512×512（非方形图会被裁边）
            transforms.CenterCrop(resolution),
            # ③ PIL → Tensor，形状 [C,H,W]，数值 [0,1]
            transforms.ToTensor(),
            # ④ 归一化到 [-1,1]：VAE 的约定输入范围
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path = self.items[idx]
        image = Image.open(img_path).convert("RGB")  # 统一成 RGB，丢掉透明通道/灰度
        pixel_values = self.transform(image)

        # 同名 .txt 就是 caption；没有就返回空字符串
        caption_path = img_path.with_suffix(".txt")
        caption = caption_path.read_text().strip() if caption_path.exists() else ""

        return {"pixel_values": pixel_values, "caption": caption}


if __name__ == "__main__":
    ds = ImageCaptionDataset(DATA_DIR)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print("图片张量形状:", tuple(batch["pixel_values"].shape))   # [2, 3, 512, 512]
    print("数值范围:", batch["pixel_values"].min().item(), batch["pixel_values"].max().item())
    print("第一条 caption:", repr(batch["caption"][0]))
```

**逐段解释：**

- `Resize(512)` 只保证短边到 512，长边按比例缩放；`CenterCrop(512)` 再裁掉长边多余部分，保证输出一定是 512×512。
- `Normalize([0.5], [0.5])` 的公式是 `(x - 0.5) / 0.5 = 2x - 1`，把 `[0,1]` 映射到 `[-1,1]`。SD 的 VAE 就是这么训练的，**训练输入、采样输出都要用这个约定**。
- `__getitem__` 返回 dict 而不是 tuple，字段名清晰，训练循环里不容易写错。
- `num_workers=0`：Windows 上多进程 DataLoader 有坑；Linux 上可以设 2 加速，但**训练脚本里若用多 worker，务必把主循环包在 `if __name__ == "__main__":` 里**，否则会无限 fork。

---

### 4.2 `02_encode.py` —— 验证 VAE：像素 ⇄ 潜空间

**目的**：确认 VAE 编码/解码正确，感受潜空间。这一步不做，后面训练出了错你都不知道是 VAE 的问题还是 UNet 的问题。

**完成标准**：打印潜变量形状 `[1, 4, 64, 64]`，重建图 `src/output/recon.png` 与原图肉眼几乎一致。

```python
"""02_encode.py —— 验证 VAE：图片 → 潜空间 → 图片
运行：.venv/bin/python src/training/02_encode.py
"""
from pathlib import Path
import torch
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms
from torchvision.utils import save_image

MODEL_DIR = Path(__file__).resolve().parents[2] / "src/models/stable-diffusion-v1-5"
VAE_SCALE = 0.18215


def load_vae():
    vae = AutoencoderKL.from_pretrained(MODEL_DIR, subfolder="vae").to("cuda", torch.float16)
    vae.eval()
    return vae


@torch.no_grad()
def roundtrip(vae, img_path, out_path):
    img = Image.open(img_path).convert("RGB")
    x = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])(img).unsqueeze(0).to("cuda", torch.float16)

    # 编码：像素 [1,3,512,512] → 潜变量 [1,4,64,64]
    latent = vae.encode(x).latent_dist.sample() * VAE_SCALE
    print("潜变量形状:", tuple(latent.shape))                 # [1, 4, 64, 64]
    print("潜变量统计: mean %.3f  std %.3f" % (latent.mean().item(), latent.std().item()))

    # 解码：潜变量 → 像素，记得先除掉缩放因子
    recon = vae.decode(latent / VAE_SCALE).sample
    recon = (recon / 2 + 0.5).clamp(0, 1)                     # [-1,1] → [0,1]
    save_image(recon, out_path)
    print("重建图已保存:", out_path)


if __name__ == "__main__":
    vae = load_vae()
    roundtrip(vae, "src/dataset/my_images/photo_01.jpg", "src/output/recon.png")
```

**逐段解释：**

- `vae.encode(x)` 返回一个**分布**（不是确定值），`.latent_dist.sample()` 从这个分布里采样。这是 VAE 的本质：编码器输出高斯分布的均值/方差，解码需要采样一个具体潜变量。
- **为什么要乘 `VAE_SCALE`**：编码出的潜变量数值范围大约在 `[-20, 20]`，直接给 UNet 学噪声会数值不稳。乘 `0.18215` 后接近 `[-1, 1]` 范围。这是 SD1.5 训练时就定下的缩放，`scheduler` 的加噪公式是围绕"潜变量≈标准正态"设计的。
- **解码前必须 `latent / VAE_SCALE`**：把缩放还原，否则 VAE 解出来的图会严重失真。训练里保存的也是"缩放后的潜变量"。
- `(recon / 2 + 0.5)` 是 `Normalize([0.5],[0.5])` 的逆变换：`[-1,1]` → `[0,1]`，`clamp` 防止浮点误差越界。
- 一个检查点：如果重建图颜色不对、很糊，先怀疑**归一化或缩放因子**搞反了，这是最高频的错误。

---

### 4.3 `03_overfit.py` —— 单图过拟合，证明训练循环正确

**目的**：把"加噪 → UNet 预测 → MSE → 反传"这条最核心的数据流跑通。**这一步用的是全量微调 UNet（不注入 LoRA）**，目的是排除 LoRA 的干扰，纯粹验证训练循环本身是对的。LoRA 在下一步才加入。

**为什么先全量微调单图**：单图、batch=1、500 步，UNet 全量在 8GB 上靠梯度检查点是跑得动的。全量微调收敛快，如果训练循环正确，loss 会一路下降；如果循环写错了（比如 timestep 搞反、噪声没加对），loss 会卡住不降，一眼就能看出来。

**完成标准**：500 步后 loss < 0.05，采样出的图就是那张原图。

```python
"""03_overfit.py —— 单图全量微调 UNet，验证训练循环
运行：.venv/bin/python src/training/03_overfit.py
"""
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms

MODEL_DIR = Path(__file__).resolve().parents[2] / "src/models/stable-diffusion-v1-5"
VAE_SCALE = 0.18215
STEPS = 500

device = "cuda"
dtype = torch.float16
torch.manual_seed(42)

# ---- 冻结组件 ----
vae = AutoencoderKL.from_pretrained(MODEL_DIR, subfolder="vae").to(device, dtype).eval()
tokenizer = CLIPTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_DIR, subfolder="text_encoder").to(device, dtype).eval()
scheduler = DDPMScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")
unet = UNet2DConditionModel.from_pretrained(MODEL_DIR, subfolder="unet").to(device, dtype)

# ---- 全量微调 UNet：所有参数都可训练（只为验证数据流，下一步才换 LoRA）----
unet.train()
unet.enable_gradient_checkpointing()   # 8GB 关键：激活重算省显存
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)


def load_one_image(path):
    x = transforms.Compose([
        transforms.Resize(512), transforms.CenterCrop(512),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])(Image.open(path).convert("RGB")).unsqueeze(0).to(device, dtype)
    return x


def encode_prompt(prompt):
    ids = tokenizer([prompt], padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    return text_encoder(ids)[0]   # last_hidden_state: [1, 77, 768]


prompt_emb = encode_prompt("a photo of my subject")
image = load_one_image("src/dataset/my_images/photo_01.jpg")

with torch.no_grad():
    z0 = vae.encode(image).latent_dist.sample() * VAE_SCALE   # [1,4,64,64]

for step in range(1, STEPS + 1):
    optimizer.zero_grad(set_to_none=True)

    # ① 随机噪声 + 随机时间步，前向加噪
    noise = torch.randn_like(z0)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
    zt = scheduler.add_noise(z0, noise, t)

    # ② UNet 预测噪声
    with torch.autocast("cuda", dtype=torch.float16):
        pred = unet(zt, t, prompt_emb).sample
        loss = F.mse_loss(pred.float(), noise.float())   # fp32 算 loss

    # ③ 反传更新
    loss.backward()
    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
    optimizer.step()

    if step % 50 == 0 or step == STEPS:
        print(f"step {step:4d}  loss {loss.item():.4f}")


# ---- 采样：从纯噪声走完整去噪，验证"采样即原图" ----
@torch.no_grad()
def sample(prompt_emb, filename):
    scheduler.set_timesteps(30)   # 采样步数
    latents = torch.randn((1, 4, 64, 64), device=device, dtype=dtype,
                          generator=torch.Generator(device).manual_seed(42))
    neg_emb = encode_prompt("")   # 空 prompt = 无条件分支
    emb = torch.cat([neg_emb, prompt_emb])   # [2, 77, 768]

    for t in scheduler.timesteps:
        inp = torch.cat([latents] * 2)                 # 无条件 + 有条件 两个输入
        pred = unet(inp, t.repeat(2).to(device), emb).sample
        uncond, cond = pred.chunk(2)
        pred = uncond + 7.5 * (cond - uncond)          # classifier-free guidance
        latents = scheduler.step(pred, t, latents).prev_sample

    img = vae.decode(latents / VAE_SCALE).sample
    img = (img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
    Image.fromarray((img * 255).round().astype("uint8")).save(filename)


sample(prompt_emb, "src/output/overfit_result.png")
print("采样图已保存: src/output/overfit_result.png")
```

**逐段解释（训练部分）：**

- **`scheduler.add_noise(z0, noise, t)` 是整个训练的数据核心**。它做的是 DDPM 前向过程：按时间步 `t` 把干净潜变量 `z0` 和随机噪声 `noise` 按比例混合成 `zt`。`t` 越大，`zt` 越接近纯噪声。
- **`t` 必须是 `long` 类型、范围 `[0, num_train_timesteps)`**。这是 UNet 拿到时间步后查时间步 embedding 表的索引，类型错了会报错或出乱子。
- **`torch.autocast("cuda", dtype=torch.float16)`**：让 UNet 前向在 fp16 下跑（快、省显存），但梯度仍保留。`pred.float()` 转回 fp32 再和 `noise.float()` 算 loss，避免 fp16 下 loss 溢出。
- **`enable_gradient_checkpointing()`**：默认 UNet 前向会保存每一层激活值供反向用，8GB 扛不住。开启后，反向时需要的激活值**重新算一遍**，显存降约一半，代价是慢 20% 左右。这是 8GB 能训 SD 的关键开关之一。
- **`clip_grad_norm_`**：限制梯度范数，防止某一步梯度爆炸把权重炸飞。LoRA 训练里也一样要加。
- **`lr=1e-5`**：全量微调的学习率比 LoRA（`1e-4`）小一个数量级，因为全量更新动的是整个大模型，lr 大了容易崩。这是全量微调和 LoRA 的重要区别之一，第 6 节会展开。

**逐段解释（采样部分）：**

- 采样和训练是两套流程。训练只走"加噪+预测"一次；采样要从纯噪声出发，**按 `scheduler.set_timesteps(30)` 给出的 30 个时间步，从大到小一步步去噪**。
- **classifier-free guidance（CFG）**：SD 生成时每次都跑两个分支——无条件（空 prompt）和条件（你的 prompt），然后按 `uncond + 7.5·(cond - uncond)` 混合。系数 `7.5` 就是 `guidance_scale`，越大越"听话"但越容易过饱和。训练时随机 drop 掉文本条件（本教程简化没做），推理时才能用 CFG。
- **`t.repeat(2)`**：因为 `emb` 是两个分支，所以时间步也要复制成两份，`[1]` → `[2]`。
- 采样用的初始噪声 `torch.randn` 固定了 seed，所以同一张图多次采样结果一致，方便对比不同训练的差异。

**排错提示**：如果 loss 卡在 1.0 左右不降，大概率是某个环节写错（常见：`zt` 没真正加到噪声、时间步没用上、`z0` 和 `noise` 的 device/dtype 不一致）。如果 loss 到 0.1 以下但采样图还是不像原图，检查采样循环的 CFG 或 `latents / VAE_SCALE`。

---

### 4.4 `04_lora_inject.py` —— 给 UNet 注入手写 LoRA

**目的**：不依赖 peft，纯手写 LoRA 层和注入逻辑。这是整份教程里"最能加深理解"的一步。

**完成标准**：注入后打印可训练参数量（rank=16 时约 1~2M），并确认 `unet` 里其他参数都 `requires_grad=False`，只有 LoRA 的 `A`/`B` 可训练。

```python
"""04_lora_inject.py —— 纯手写 LoRA 层 + 注入 UNet
运行：.venv/bin/python src/training/04_lora_inject.py
"""
import torch
from pathlib import Path
from diffusers import UNet2DConditionModel

MODEL_DIR = Path(__file__).resolve().parents[2] / "src/models/stable-diffusion-v1-5"


class LoRALinear(torch.nn.Module):
    """y = W x + (alpha/r) · B(A x)

    A: [r, in]    B: [out, r]
    初始化：A ~ N(0, 0.01)，B = 0 → 训练开始时增量恰好为 0
    """

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                        # 冻结原权重

        self.lora_A = torch.nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, rank))
        self.scaling = alpha / rank
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

    def forward(self, x):
        return self.base(x) + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


def inject_lora(unet, rank=16, alpha=16, target=("to_q", "to_k", "to_v", "to_out")):
    """遍历 UNet，把名字命中 target 的 Linear 换成 LoRALinear，返回可训练参数列表。"""
    trainable = []
    for name, module in unet.named_modules():
        for attr in target:
            child = getattr(module, attr, None)
            if isinstance(child, torch.nn.Linear):
                lora = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, attr, lora)
                trainable += [lora.lora_A, lora.lora_B]
    return trainable


if __name__ == "__main__":
    unet = UNet2DConditionModel.from_pretrained(MODEL_DIR, subfolder="unet", dtype=torch.float16)
    lora_params = inject_lora(unet, rank=16, alpha=16)

    n_trainable = sum(p.numel() for p in lora_params)
    n_frozen = sum(p.numel() for p in unet.parameters() if not p.requires_grad)
    print(f"LoRA 可训练参数: {n_trainable/1e6:.2f} M")
    print(f"冻结参数: {n_frozen/1e6:.1f} M")

    # 验证：UNet 里应该只有 LoRA 的 A/B 需要梯度
    grad_params = [n for n, p in unet.named_parameters() if p.requires_grad]
    print(f"需要梯度的参数数量: {len(grad_params)}")
    print("前 4 个:", grad_params[:4])
```

**逐段解释：**

- **`LoRALinear.forward` 的矩阵运算**：`x` 形状 `[..., in]`。`x @ A.T` 得 `[..., r]`，再 `@ B.T` 得 `[..., out]`，乘上 `scaling = alpha/r` 后加到 `base(x)` 上。`A.T`、`B.T` 用的是 PyTorch 对 `Parameter` 的转置，不会产生额外拷贝。
- **为什么 `A` 要乘 `0.01`、`B` 要初始化为 0**：`B=0` 保证训练第 0 步时 `B·A = 0`，输出完全等于原模型，训练只是"从零开始长增量"。`A` 用随机小值是为了打破对称性（如果 A 也全 0，梯度也是 0，永远训不动）。
- **`scaling = alpha / rank`**：这个系数只参与前向，不影响反向时 A/B 各自的梯度方向，但影响"增量整体的大小"。`alpha` 不变、`rank` 翻倍时，`scaling` 减半，两个 rank 的效果会互相补偿——这就是为什么调 `alpha` 通常要配合 `rank`。
- **`inject_lora` 为什么用 `getattr(module, attr, None)`**：不是每个模块都有 `to_q/to_k/to_v/to_out`，用默认值 `None` 安全跳过。命中的 `Linear` 会被整体替换成 `LoRALinear`，原 `Linear` 作为 `.base` 保留（冻结）。
- **`trainable += [lora.lora_A, lora.lora_B]`**：返回的是 Parameter 对象列表，优化器直接吃这个列表，就只会更新 LoRA。

**验证技巧**：注入后先跑一遍前向（`unet(sample_latents, t, emb).sample`），确认不会报错；再检查 `named_parameters()` 里 `requires_grad=True` 的确实都是 `lora_A`/`lora_B`。

---

### 4.5 `05_train.py` —— 完整训练脚本

**目的**：把前面所有零件拼成能用的训练脚本：真实数据集 + LoRA + 梯度累积 + 定期采样预览 + 检查点保存。

**完成标准**：几十张图跑 1000~3000 步不 OOM，`src/output/lora/` 下出现 `lora_step*.safetensors` 和 `sample_step*.png`，采样图逐渐变好。

```python
"""05_train.py —— 完整 LoRA 训练脚本
运行：
    .venv/bin/python src/training/05_train.py \
        --data-dir src/dataset/my_images --steps 1500 --rank 16 --alpha 16 --lr 1e-4
"""
import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms

MODEL_DIR = Path(__file__).resolve().parents[2] / "src/models/stable-diffusion-v1-5"
VAE_SCALE = 0.18215
RESOLUTION = 512
MAX_TOKEN_LEN = 77
TRAINABLE_DTYPE = torch.float32
SAVE_EVERY = 500
SAMPLE_EVERY = 200


class ImageCaptionDataset(Dataset):
    # 与 01_dataset.py 完全一致，这里省略注释
    def __init__(self, data_dir: Path, resolution: int = RESOLUTION):
        self.items = sorted(p for p in Path(data_dir).iterdir()
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not self.items:
            raise FileNotFoundError(f"{data_dir} 里没有图片")
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path = self.items[idx]
        image = Image.open(path).convert("RGB")
        pixel = self.transform(image)
        cap = path.with_suffix(".txt")
        caption = cap.read_text().strip() if cap.exists() else ""
        return {"pixel_values": pixel, "caption": caption}


class LoRALinear(torch.nn.Module):
    # 与 04_lora_inject.py 完全一致，这里省略注释
    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, rank))
        self.scaling = alpha / rank
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

    def forward(self, x):
        return self.base(x) + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


def inject_lora(unet, rank=16, alpha=16, target=("to_q", "to_k", "to_v", "to_out")):
    trainable = []
    for _, module in unet.named_modules():
        for attr in target:
            child = getattr(module, attr, None)
            if isinstance(child, torch.nn.Linear):
                lora = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, attr, lora)
                trainable += [lora.lora_A, lora.lora_B]
    return trainable


@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompts, device):
    ids = tokenizer(prompts, padding="max_length", max_length=MAX_TOKEN_LEN,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    return text_encoder(ids)[0]   # [B, 77, 768]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("src/output/lora"))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-prompt", type=str, default="a photo of my subject")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载冻结组件（fp16 省显存）----
    vae = AutoencoderKL.from_pretrained(MODEL_DIR, subfolder="vae").to(device, torch.float16).eval()
    text_encoder = CLIPTextModel.from_pretrained(MODEL_DIR, subfolder="text_encoder").to(device, torch.float16).eval()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(MODEL_DIR, subfolder="unet").to(device, torch.float16)

    # ---- 注入 LoRA，只有 A/B 需要梯度 ----
    lora_params = inject_lora(unet, rank=args.rank, alpha=args.alpha)
    print(f"LoRA 可训练参数: {sum(p.numel() for p in lora_params)/1e6:.2f} M (rank={args.rank})")

    # LoRA 权重升到 fp32（AMP 下更稳）
    for p in lora_params:
        p.data = p.data.to(TRAINABLE_DTYPE)
        p.requires_grad_(True)

    unet.train()
    unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)
    def lr_lambda(step):
        return min(1.0, step / 100)          # 前 100 步线性 warmup，之后恒定
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    dataset = ImageCaptionDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)
    loader_iter = iter(loader)

    print(f"数据集: {len(dataset)} 张 | 总步数 {args.steps} | 等效 batch {args.batch_size * args.grad_accum}")

    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        pixel_values = batch["pixel_values"].to(device, torch.float16)
        prompts = batch["caption"]

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * VAE_SCALE
        encoder_hidden_states = encode_prompts(tokenizer, text_encoder, prompts, device)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        with torch.autocast("cuda", dtype=torch.float16):
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float())

        (loss / args.grad_accum).backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0:
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f}  lr {lr_sched.get_last_lr()[0]:.2e}")

        if step % SAMPLE_EVERY == 0 or step == args.steps:
            sample_image(vae, text_encoder, tokenizer, unet, scheduler, args, step)
            unet.train()

        if step % SAVE_EVERY == 0 or step == args.steps:
            save_lora(unet, args, step)

    print("训练结束。下一步用第 5 节的方法加载 LoRA 验证效果。")


@torch.no_grad()
def sample_image(vae, text_encoder, tokenizer, unet, scheduler, args, step):
    device = "cuda"
    unet.eval()
    cfg, n_steps = 7.5, 30

    emb_pos = encode_prompts(tokenizer, text_encoder, [args.sample_prompt], device)
    emb_neg = encode_prompts(tokenizer, text_encoder, [""], device)
    emb = torch.cat([emb_neg, emb_pos])

    latents = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float16,
                          generator=torch.Generator(device).manual_seed(args.seed))
    scheduler.set_timesteps(n_steps)

    for t in scheduler.timesteps:
        pred = unet(torch.cat([latents] * 2), t.repeat(2).to(device), emb).sample
        uncond, cond = pred.chunk(2)
        pred = uncond + cfg * (cond - uncond)
        latents = scheduler.step(pred, t, latents).prev_sample

    image = vae.decode(latents / VAE_SCALE).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    out = args.output_dir / f"sample_step{step:05d}.png"
    Image.fromarray((image * 255).round().astype("uint8")).save(out)
    print(f"  采样已保存: {out}")


def save_lora(unet, args, step):
    state = {}
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    path = args.output_dir / f"lora_step{step:05d}.safetensors"
    from safetensors.torch import save_file
    save_file(state, path)
    print(f"  LoRA 已保存: {path} ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

**关键点解释（03 已讲过的部分不再重复）：**

- **可训练参数只有 LoRA**：`optimizer = AdamW(lora_params, ...)` 只吃 LoRA 的 A/B。UNet 其余参数冻结，VAE/CLIP 根本没进优化器。这就是 LoRA 的核心收益——参数量从 8.6 亿降到约 1~2M（rank=16）。
- **LoRA 权重转 fp32**：`p.data = p.data.to(torch.float32)` 让可训练参数用 fp32 存。混合精度训练里，`autocast` 只把前向计算降到 fp16，**参数本身保持 fp32** 是 PyTorch AMP 的标准做法，能显著减少梯度更新时的数值下溢。
- **梯度累积**：`(loss / grad_accum).backward()`，每 `grad_accum` 步才 `optimizer.step()` 一次。这样 batch=1 也能等效 batch=4，8GB 显存下保证能跑，且大等效 batch 让训练更稳。**注意 `loss` 要除以 `grad_accum`**，否则等效于把学习率放大了 4 倍。
- **warmup**：`min(1.0, step/100)` 让学习率前 100 步从 0 线性爬升到目标值。一开始梯度方向很乱，小 lr 更稳。
- **`drop_last=True`**：丢掉最后一个不足 batch 的批次，避免 `randn_like` 和 `timesteps` 的形状对不上（batch=1 时通常无所谓，但好习惯）。
- **`save_lora` 只存 A/B**：不存整个 UNet。rank=16 时保存的文件通常只有几十 MB，方便分享和加载。
- **采样里的 `unet.eval()` / `unet.train()` 切换**：采样时禁用 dropout、batch norm 等训练专用行为，采完要记得切回 `train()`，否则后续训练会错。

---

## 5. 采样验证与加载 LoRA

训练时每 200 步已经自动存了预览图，**判断训练好坏的第一标准就是看这些图**，而不是只看 loss 数字（loss 低但图崩的"过拟合到噪声"也会发生）。

### 5.1 训练后加载 LoRA 生成图

`save_lora` 存的是自定义 `LoRALinear` 的 `lora_A`/`lora_B`。加载时流程和训练**完全对称**：

```python
"""06_load.py —— 加载训练好的 LoRA 并生成（自包含版）
运行：.venv/bin/python src/training/06_load.py --lora src/output/lora/lora_step01500.safetensors
"""
import argparse
from pathlib import Path
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file

MODEL_DIR = Path(__file__).resolve().parents[2] / "src/models/stable-diffusion-v1-5"
VAE_SCALE = 0.18215


class LoRALinear(torch.nn.Module):
    # 与 05_train.py 完全一致
    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, rank))
        self.scaling = alpha / rank
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

    def forward(self, x):
        return self.base(x) + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


def inject_lora(unet, rank=16, alpha=16, target=("to_q", "to_k", "to_v", "to_out")):
    for _, module in unet.named_modules():
        for attr in target:
            child = getattr(module, attr, None)
            if isinstance(child, torch.nn.Linear):
                setattr(module, attr, LoRALinear(child, rank=rank, alpha=alpha))


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompt, device):
    ids = tokenizer([prompt], padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    return text_encoder(ids)[0]


@torch.no_grad()
def sample(vae, text_encoder, tokenizer, unet, scheduler, prompt, out_path, seed=42):
    device = "cuda"
    unet.eval()
    emb_pos = encode_prompt(tokenizer, text_encoder, prompt, device)
    emb_neg = encode_prompt(tokenizer, text_encoder, "", device)
    emb = torch.cat([emb_neg, emb_pos])

    latents = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float16,
                          generator=torch.Generator(device).manual_seed(seed))
    scheduler.set_timesteps(30)
    for t in scheduler.timesteps:
        pred = unet(torch.cat([latents] * 2), t.repeat(2).to(device), emb).sample
        uncond, cond = pred.chunk(2)
        pred = uncond + 7.5 * (cond - uncond)
        latents = scheduler.step(pred, t, latents).prev_sample

    img = vae.decode(latents / VAE_SCALE).sample
    img = (img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
    Image.fromarray((img * 255).round().astype("uint8")).save(out_path)
    print("生成完成:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--prompt", default="a photo of my subject")
    parser.add_argument("--out", default="src/output/loaded_sample.png")
    args = parser.parse_args()

    device = "cuda"
    vae = AutoencoderKL.from_pretrained(MODEL_DIR, subfolder="vae").to(device, torch.float16).eval()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_DIR, subfolder="text_encoder").to(device, torch.float16).eval()
    scheduler = DDPMScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(MODEL_DIR, subfolder="unet").to(device, torch.float16)

    # 重新注入与训练时一模一样的 LoRA 结构，然后加载 A/B 权重
    inject_lora(unet, rank=16, alpha=16)
    state = load_file(args.lora)
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data.copy_(state[f"{name}.lora_A"].to(module.lora_A.dtype, device))
            module.lora_B.data.copy_(state[f"{name}.lora_B"].to(module.lora_B.dtype, device))

    sample(vae, text_encoder, tokenizer, unet, scheduler, args.prompt, args.out)


if __name__ == "__main__":
    main()
```

> 说明：`06_load.py` 是自包含脚本，`LoRALinear` / `inject_lora` / 采样函数与 `05_train.py` 相同（练习时复制即可）。加载的关键点：**重新注入与训练时相同的 LoRA 结构（rank/alpha 必须一致），再把保存的 A/B 权重 `copy_` 进去**。

### 5.2 判断训练结果

- **loss 曲线**：正常应持续下降并趋于平缓。卡在 1.0 附近不降 = 循环有 bug（回到 4.3 排错）。
- **采样图**：前几百步是模糊色块 → 逐渐出现主体轮廓 → 后期细节稳定。如果后期开始出现伪影/噪点，是**过拟合**，减少步数或增大 `dropout`。
- **多样性**：换不同 seed 采样，看主体是否稳定、背景是否合理。只会复现训练图而不会泛化 = 过拟合。

---

## 6. 调参速查

| 超参 | 建议起点 | 说明 |
|------|---------|------|
| rank (r) | 8~16 | 越大容量越大，越容易过拟合；8GB 下 8~32 都行 |
| alpha | = rank | 效果近似 `alpha/r`；`alpha=2r` 时增量整体放大一倍 |
| LoRA 目标层 | `to_q,to_k,to_v,to_out` | 只挂 attention 通常够用，进阶加 FFN 层 |
| 学习率 | 1e-4（AdamW） | LoRA 比全量微调大 1~2 个数量级；`weight_decay=1e-2` |
| batch × 累积 | 1~2 × 4 | 等效 batch 4~8，8GB 稳跑 |
| 步数 | 1000~3000（几十张图） | 配合采样预览判断，不要只看 loss |
| 梯度检查点 | 开 | 8GB 必开，省一半显存换 ~20% 时间 |
| text encoder LoRA | 先不动 | 进阶项，容易引入不稳定 |
| SNR gamma / min-SNR | 5 | 高分辨率/大 rank 时让 loss 加权更稳（进阶） |

**过拟合 vs 欠拟合判断**：

- 欠拟合：采样图还是糊/主体不像 → 加步数、加 rank、加 lr。
- 过拟合：采样图复现训练图但换 seed 就崩、背景怪 → 减步数、减 rank、加 dropout（如 0.1）、混入正则化图。

---

## 7. 常见坑与排错

1. **CUDA OOM（显存不够）**
   - 确认 `enable_gradient_checkpointing()` 已开；
   - 确认 VAE/CLIP 用 fp16 且 `eval()`，UNet 用 fp16；
   - 减小 batch（本来就是 1 就别再减）、减小分辨率到 448/384（进阶，需同步改 VAE 尺寸）;
   - 加 `pipe.vae.enable_slicing()`（加载脚本里已有）。

2. **loss 卡在 1.0 左右不降**
   - 99% 是训练循环 bug：检查 `noise` 是否真的加到 `z0` 上（用 `scheduler.add_noise`）、`t` 是否传进了 UNet、`noise` 和 `latents` 的 device/dtype 是否一致、`z0` 是否乘了 `VAE_SCALE`。

3. **采样图发灰 / 发黑 / 全噪点**
   - 解码前忘了 `latents / VAE_SCALE`；
   - 归一化逆变换写错：正确是 `(x / 2 + 0.5)`，不是 `(x + 1) / 2` 的另一种写法搞混；
   - 采样步数太少（<20）或 CFG 太小/太大。

4. **`import torchvision` 报错**：`uv add torchvision`。

5. **DataLoader 多进程死循环/卡死**：把主逻辑放进 `if __name__ == "__main__":`，或先设 `num_workers=0` 排查。

6. **训练中途 loss 突然变 NaN**
   - lr 太大（先降 10 倍试）；
   - 忘了 `clip_grad_norm_`；
   - LoRA 权重没转 fp32、在 fp16 下直接训练导致下溢。

7. **保存的 LoRA 加载后图不对**
   - 加载时 `rank`/`alpha` 必须和训练时一致；
   - 加载后忘了 `unet.eval()`。

---

## 8. 下一步

按 `docs/ROADMAP.md` 继续：

1. **阶段 5 调参实验**：同一数据集跑 3 组不同 rank/lr，采样对比，写下结论。
2. **阶段 4 完善数据**：WD14 打标、触发词、prior preservation 正则化图。
3. **阶段 6 进阶**：
   - 读 `diffusers/models/attention_processor.py`，脱离 `add_adapter` 完全手撸 attention 注入；
   - min-SNR 加权损失；
   - 用 peft 的 `LoraConfig` + `get_peft_model` 做生产级管理，对比你自己的手写实现；
   - Textual Inversion、ControlNet。

**一句话回顾整个训练流程**：图像进 VAE 变潜变量 → 随机时间步加噪 → UNet（带 LoRA）预测噪声 → 和真噪声算 MSE → 只更新 LoRA 的 A/B。采样时反过来：纯噪声 → UNet 一步步去噪 → VAE 解码回像素。

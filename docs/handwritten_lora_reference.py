"""手写 SD1.5 LoRA 训练参考实现（学习用，先自己写再对照！）

配套 docs/ROADMAP.md 阶段 3。
仅依赖 diffusers/transformers/torch，无 accelerate，无 peft —— 一切逻辑摊开写。
约 300 行，可拆成 01~05 五个练习文件。

运行：
    cd /home/lipenghui/lora
    .venv/bin/python docs/handwritten_lora_reference.py \
        --data-dir src/dataset/my_images \
        --steps 1500 --rank 16 --alpha 16 --lr 1e-4
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

try:
    from torchvision import transforms  # pip install torchvision（若缺）
except ImportError:
    raise SystemExit("缺 torchvision：.venv/bin/pip install torchvision")

# ============================================================
# 0. 常量
# ============================================================

MODEL_DIR = Path(__file__).resolve().parents[1] / "src/models/stable-diffusion-v1-5"

# SD1.5 的 VAE 缩放因子：潜空间数值整体乘上它，让分布更接近 N(0,1)
VAE_SCALE = 0.18215

RESOLUTION = 512
MAX_TOKEN_LEN = 77  # CLIP 的固定 token 长度
TRAINABLE_DTYPE = torch.float32  # LoRA 权重用 fp32 训，稳定；UNet 推理部分 fp16
SAVE_EVERY = 500
SAMPLE_EVERY = 200


# ============================================================
# 1. 数据集：图片目录 + 每图同名 .txt 标注（my_photo.jpg + my_photo.txt）
# ============================================================

class ImageCaptionDataset(Dataset):

    def __init__(self, data_dir: Path, resolution: int = RESOLUTION):
        self.items = sorted(
            p for p in Path(data_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not self.items:
            raise FileNotFoundError(f"{data_dir} 里没有图片")
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),          # [0,1]
            transforms.Normalize([0.5], [0.5]),  # [-1,1]，VAE 的输入约定
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        caption_path = img_path.with_suffix(".txt")
        caption = caption_path.read_text().strip() if caption_path.exists() else ""

        return {"pixel_values": pixel_values, "caption": caption}


# ============================================================
# 2. LoRA 层：纯手写，就两个小矩阵
#    y = W x + (alpha/r) * B(A x)
#    A: r×in  B: out×r  初始化 A~N(0,σ)  B=0 → 训练开始时增量恰好为 0
# ============================================================

class LoRALinear(torch.nn.Module):

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base  # 冻结的原权重
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.lora_A = torch.nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, rank))  # B=0！
        self.scaling = alpha / rank
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

    def forward(self, x):
        # 原始输出（fp16 权重）+ 低秩增量
        return self.base(x) + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


def inject_lora(unet, rank=16, alpha=16, target=("to_q", "to_k", "to_v", "to_out")):
    """遍历 UNet，把名字命中 target 的 Linear 换成 LoRALinear。

    返回可训练参数列表。SD1.5 的 attention 模块里，这些名字覆盖了
    self-attn 和 cross-attn 的全部投影层。
    """
    trainable = []
    for name, module in unet.named_modules():
        for attr in target:
            child = getattr(module, attr, None)
            if isinstance(child, torch.nn.Linear):
                lora = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, attr, lora)
                trainable += [lora.lora_A, lora.lora_B]
    return trainable


def merge_and_unload(unet):
    """训练完把 BA 合并进原权重：W += (alpha/r)·BA，用于导出完整模型或纯 fp16 LoRA。"""
    for module in unet.modules():
        for attr in list(module._modules):
            layer = getattr(module, attr)
            if isinstance(layer, LoRALinear):
                with torch.no_grad():
                    delta = (layer.lora_B @ layer.lora_A) * layer.scaling
                    layer.base.weight += delta.to(layer.base.weight.dtype)
                setattr(module, attr, layer.base)


# ============================================================
# 3. 文本编码：prompt → 77×768（整个训练中 CLIP 冻结，先编码一次再循环可提速）
# ============================================================

@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompts, device):
    text_input = tokenizer(
        prompts,
        padding="max_length",
        max_length=MAX_TOKEN_LEN,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    return text_encoder(text_input)[0]  # last_hidden_state: [B, 77, 768]


# ============================================================
# 4. 训练主循环
# ============================================================

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
    vae = AutoencoderKL.from_pretrained(MODEL_DIR, subfolder="vae", dtype=torch.float16).to(device).eval()
    text_encoder = CLIPTextModel.from_pretrained(MODEL_DIR, subfolder="text_encoder", dtype=torch.float16).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(MODEL_DIR, subfolder="unet", dtype=torch.float16)

    # ---- 注入 LoRA：UNet 里只有 LoRA 的 A/B 需要梯度 ----
    lora_params = inject_lora(unet, rank=args.rank, alpha=args.alpha)
    n_trainable = sum(p.numel() for p in lora_params)
    print(f"LoRA 可训练参数: {n_trainable/1e6:.2f} M (rank={args.rank})")

    # LoRA 权重升到 fp32（AMP 下更稳），并打开梯度
    for p in lora_params:
        p.data = p.data.to(TRAINABLE_DTYPE)
        p.requires_grad_(True)

    unet.to(device)
    unet.train()

    # 8GB 显存的关键：UNet 内部激活值用检查点重算，省一半显存换 ~20% 时间
    unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)
    # 简易 warmup + 恒定；进阶可换 cosine
    def lr_lambda(step):
        return min(1.0, step / 100)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- 数据 ----
    dataset = ImageCaptionDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    loader_iter = iter(loader)

    print(f"数据集: {len(dataset)} 张 | 总步数 {args.steps} | 等效 batch {args.batch_size * args.grad_accum}")

    # ---- 训练循环 ----
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
        prompts = batch["caption"]

        # ① VAE 编码到潜空间（无梯度：VAE 冻结）
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * VAE_SCALE  # [B, 4, 64, 64]

        # ② 文本条件（无梯度：CLIP 冻结）
        encoder_hidden_states = encode_prompts(tokenizer, text_encoder, prompts, device)

        # ③ 随机时间步 + 随机噪声，前向加噪（DDPM 的核心一步）
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # ④ UNet 预测噪声（LoRA 在这里被训练；dtype 混合由 autocast 处理）
        with torch.autocast("cuda", dtype=torch.float16):
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float())  # 用 fp32 算 loss 更稳

        # ⑤ 反传（梯度累积）
        (loss / args.grad_accum).backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)  # 防梯度爆炸
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0:
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f}  lr {lr_sched.get_last_lr()[0]:.2e}")

        # ---- 定期采样预览：判断训练是否在正轨上的唯一标准 ----
        if step % SAMPLE_EVERY == 0 or step == args.steps:
            sample_image(vae, text_encoder, tokenizer, unet, scheduler, args, step)
            unet.train()

        # ---- 保存 ----
        if step % SAVE_EVERY == 0 or step == args.steps:
            save_lora(unet, args, step)

    print("训练结束。用 src/scripts/loder_lora.py 的思路加载 unet+LoRA 验证效果。")


# ============================================================
# 5. 采样（DDPM step 版，简单但慢；进阶换 DPMSolverMultistepScheduler）
# ============================================================

@torch.no_grad()
def sample_image(vae, text_encoder, tokenizer, unet, scheduler, args, step):
    device = "cuda"
    unet.eval()

    prompt = args.sample_prompt
    negative = ""
    cfg = 7.5
    n_steps = 30

    emb_pos = encode_prompts(tokenizer, text_encoder, [prompt], device)
    emb_neg = encode_prompts(tokenizer, text_encoder, [negative], device)
    emb = torch.cat([emb_neg, emb_pos])

    latents = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float16,
                          generator=torch.Generator(device).manual_seed(args.seed))
    scheduler.set_timesteps(n_steps)

    for t in scheduler.timesteps:
        noise_pred = unet(torch.cat([latents] * 2), t.repeat(2).to(device), emb).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + cfg * (noise_cond - noise_uncond)  # classifier-free guidance
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    image = vae.decode(latents / VAE_SCALE).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    out = args.output_dir / f"sample_step{step:05d}.png"
    Image.fromarray((image * 255).round().astype("uint8")).save(out)
    print(f"  采样已保存: {out}")
    return unet


def save_lora(unet, args, step):
    """只保存 LoRA 的 A/B 权重（几 MB~几十 MB），而不是整个 UNet。"""
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

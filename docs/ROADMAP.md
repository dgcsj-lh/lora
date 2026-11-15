# SD1.5 LoRA 微调学习路线（自己动手写训练代码）

> 目标：不看 kohya_ss，纯手写 `diffusers` 训练脚本，在 8GB 显存（RTX 5060）上完成
> SD1.5 的 LoRA 微调，并且理解每一步在做什么。
>
> 使用方法：按阶段推进，每个阶段都有【学什么】和【动手任务】。
> **动手任务做完再进入下一阶段**，代码写在 `src/training/` 下。

---

## 阶段 0：热身（半天）

**学什么**

- 扩散模型直觉：加噪 → 学去噪 → 采样，一文入门（重点看配图和 GIF）：
  - Lilian Weng《What are Diffusion Models?》 <https://lilianweng.github.io/posts/2021-07-11-diffusion-models/>
- SD1.5 的整体结构（你已经用 `loder_lora.py` 见过了）：
  - Text Encoder：CLIP ViT-L/14 文本塔，文本 → 77×768 的条件向量
  - UNet：以时间步 t 和文本条件为输入，预测噪声；LoRA 就装在这里的 attention 层
  - VAE：512×512×3 像素空间 ⇄ 64×64×4 潜空间，**训练在潜空间进行**

**动手任务**

- [ ] 已有 `loder_lora.py`，逐行读懂，能向别人解释：为什么训练在潜空间、DTYPE 为什么用 fp16。

---

## 阶段 1：论文（2~3 天，可穿插后面进行）

按这个顺序读，每篇只抓一个核心思想：

| # | 论文 | 年份 | 要抓的点 |
|---|------|------|---------|
| 1 | DDPM：*Denoising Diffusion Probabilistic Models* <https://arxiv.org/abs/2006.11239> | 2020 | 前向加噪公式、训练目标就是预测噪声 ε |
| 2 | LDM（Stable Diffusion 本体）：*High-Resolution Image Synthesis with Latent Diffusion Models* <https://arxiv.org/abs/2112.10752> | 2022 | 为什么用 VAE 压到潜空间、cross-attention 如何注入文本条件 |
| 3 | Classifier-Free Guidance <https://arxiv.org/abs/2207.12598> | 2022 | 训练时随机 drop 掉文本条件，推理时 `7.5` 这个系数就是它 |
| 4 | LoRA：*Low-Rank Adaptation of Large Language Models* <https://arxiv.org/abs/2106.09685> | 2021 | `W' = W + BA`，只训低秩矩阵；理解 rank / alpha / dropout |
| 5 | DreamBooth <https://arxiv.org/abs/2208.12242> | 2022 | 少样本个性化思想；**prior preservation（正则化图）** 防"语言漂移" |

**动手任务**

- [ ] 用 1 页纸写出：给定 batch，LoRA 一步训练的计算流程（VAE 编码 → 加噪 → UNet 预测 → MSE → 反传，只更新 BA）。

---

## 阶段 2：官方训练入门（2~3 天）

**学什么**

> 本机访问不了 huggingface.co，文档走代理或镜像看。

- Hugging Face 扩散课程（结构最完整，可看中文版）：
  <https://huggingface.co/learn/diffusion-course>（中文：<https://huggingface.co/learn/diffusion-course/zh-CN>）
- Annotated Diffusion（带完整代码逐段讲解）： <https://huggingface.co/blog/annotated-diffusion>
- diffusers 官方训练文档：
  - 文生图训练总览 <https://huggingface.co/docs/diffusers/training/overview>
  - 官方 LoRA 教程（text2image-fine-tune 脚本）<https://huggingface.co/docs/diffusers/training/lora>

**动手任务**

- [ ] 跑通官方 `train_text_to_image_lora.py`（不依赖它写自己的代码，先建立"能跑"的基准）：
  `diffusers/examples/text_to_image/train_text_to_image_lora.py`
- [ ] 记录它用到的所有组件清单（噪声调度器、optimizer、lr_scheduler、AMP、EMA……），
  这就是你之后手写脚本的 TODO 清单。

---

## 阶段 3：手写"最简可训练"脚本（1 周，核心）

**学什么**

- `DDPMScheduler` 的两个关键 API：`add_noise()`（前向加噪）和 `step()`（采样用，训练用不到）。
- 训练循环的完整数据流：

```
图像 512×512
  → VAE.encode()          → 潜变量 4×64×64 × 0.18215
文本 prompt
  → tokenizer(77)         → text_encoder → 77×768 条件向量
随机时间步 t, 随机噪声 ε
  → scheduler.add_noise() → 加噪潜变量 z_t
UNet(z_t, t, 条件向量)     → 预测噪声 ε̂
loss = MSE(ε̂, ε)
loss.backward()           → 只有 LoRA 的 B、A 有梯度
```

- 8GB 显存的核心技巧：**梯度检查点**（`gradient_checkpointing_enable`）、**冻结 VAE/CLIP**、fp16 + AMP、batch 1~2 + 梯度累积、attention slicing / xformers。
- 附件 `docs/handwritten_lora_reference.py`：约 300 行、带逐段注释的参考实现
  （先自己写，卡住了再对照，别直接抄）。

**动手任务（每个都是独立小文件，写进 `src/training/`）**

- [ ] `01_dataset.py`：图片目录 → Dataset（中心裁剪 512、归一化）、DataLoader
- [ ] `02_encode.py`：验证 VAES：图片 ↔ 潜空间，来回转换后画图对比
- [ ] `03_overfit.py`：**对 1 张图过拟合 500 步**，最终 MSE < 0.05 且采样出的图
  就是那张图 → 证明你的训练循环是对的（最重要的一步）
- [ ] `04_lora_inject.py`：给 UNet 的 attention 层注入 LoRA（用 diffusers 内置
  `add_adapter` 或 peft，再试纯手写 `nn.Linear` 版本）
- [ ] `05_train.py`：完整训练脚本：真实数据集（几十张图）+ args + 检查点保存 +
  采样预览

---

## 阶段 4：数据集与提示词工程（贯穿进行）

**学什么**

- 打标：用 BLIP / WD14 tagger 给图自动生成 caption，再人工修正
- 触发词（trigger word）：给目标主体一个专属 token，如 `mypet dog`
- 正则化图像（prior preservation）：与目标同类的通用图 + 类别 prompt，
  按 1:1~1:3 混入，防止模型忘记"狗"的一般概念
- 分辨率桶（aspect ratio bucketing）：同尺寸 batch 训练更稳

**动手任务**

- [ ] 准备一个 20~50 张图的小数据集（自己的照片 / 某个固定角色 / 某种风格），
  打好标，建好 `src/dataset/` 目录结构

---

## 阶段 5：调参与评估（持续）

**学什么**

| 超参 | 建议起点 | 说明 |
|------|---------|------|
| rank (r) | 8~32 | 越大容量越大、越容易过拟合 |
| alpha | = rank 或 2×rank | 缩放系数，效果 ≈ alpha/r |
| LoRA 目标层 | `to_q,to_k,to_v,to_out`（可加 `ff`） | 只挂 attention 通常够用 |
| 学习率 | 1e-4（AdamW） | LoRA 比全量微调大 1~2 个数量级 |
| batch × 累积 | 1~2 × 4 | 等效 batch 4~8 |
| 步数 | 1000~3000（小数据集） | 配合采样预览判断 |
| text encoder LoRA | 先不动 | 进阶项 |
| SNR gamma / min-SNR | 5 | 加权损失，高分辨率/大 rank 时明显更稳 |

- 过拟合 vs 欠拟合的判断：每 200~500 步用固定 seed 采样看图
- 用 fp16 保存/合并 LoRA，用你现有的 `loder_lora.py` 加载验证
- （可选）W&B / TensorBoard 记录 loss 曲线

**动手任务**

- [ ] 对同一数据集跑 3 组不同 rank / lr 的对比实验，各保存检查点并采样对比

---

## 阶段 6：进阶方向（选学）

- **手写 attention 注入**：读 `attention_processor.py`，脱离 `add_adapter` 手撸 LoRA 层 —— 最能加深对 SD/LoRA 的理解
- **ControlNet** <https://arxiv.org/abs/2302.05543>：给扩散模型加可控条件
- **Textual Inversion** <https://arxiv.org/abs/2208.01618>：学一个新 embedding 而不是 LoRA
- **min-SNR 加权** <https://arxiv.org/abs/2303.09556>：平衡不同时间步的损失
- **PEFT 文档** <https://huggingface.co/docs/peft/index>：生产级 LoRA 管理
- WebUI 社区对 LoRA 各参数的经验讨论（搜 "SD LoRA rank alpha guide"）

---

## 附：本项目环境现状

| 组件 | 版本 | 说明 |
|------|------|------|
| torch | 2.11.0+cu128 | RTX 5060 必须 cu128+ |
| diffusers | 0.43.0 | 注意：`pipe.enable_vae_slicing()` 等旧 API 已移除 |
| transformers | 5.23.0 | |
| peft | 0.21.3 | LoRA 注入可用 |
| datasets | 5.2.0 | 数据集加载 |
| bitsandbytes | 0.50.2 | 8bit optimizer（显存紧张时用） |
| accelerate | 1.15.0 | 分布式/混合精度封装 |
| wandb | 0.31.0 | 实验记录（可选） |
| safetensors | 0.8.0 | LoRA 保存格式 |

- SD1.5 全套模型在 `src/models/stable-diffusion-v1-5/`，加载方式见 `src/scripts/loder_lora.py`
- 网络注意：`huggingface.co` 与 PyPI 大文件直连不稳；模型走 `HF_ENDPOINT=https://hf-mirror.com`，
  pip/uv 走本机代理 `127.0.0.1:7897`
- 官方示例脚本在 `diffusers` 包里没有附带，从这里取：
  <https://github.com/huggingface/diffusers/tree/main/examples/text_to_image>

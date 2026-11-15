# LoRA

基于 `diffusers` 的 Stable Diffusion 1.5 LoRA 训练/实验项目。

## 环境

- Python 3.11（`.venv`，由 uv 管理）
- torch 2.11.0+cu128（RTX 5060 / sm_120 需要 cu128 及以上）
- diffusers 0.43.0、transformers 5.23.0、accelerate 1.15.0
- GPU：RTX 5060 8GB

## SD1.5 模型

模型已下载到 `src/models/stable-diffusion-v1-5/`（约 5.2GB，完整 pipeline：
unet / vae / text_encoder / tokenizer / scheduler / safety_checker / feature_extractor）。

来源：`stable-diffusion-v1-5/stable-diffusion-v1-5`（经 hf-mirror 镜像下载）。

## 使用

```bash
# 加载模型 + 查看结构 + 生成测试图（默认读取本地模型）
.venv/bin/python src/scripts/loder_lora.py

# 只用 CPU/低显存运行
.venv/bin/python src/scripts/loder_lora.py --cpu-offload

# 只加载不生成
.venv/bin/python src/scripts/loder_lora.py --no-generate
```

测试图输出为当前目录下的 `sd15_test.png`。

## 备注

- 本机访问不了 `huggingface.co`，下载模型时请用镜像：

  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

- 模型和输出目录已在 `.gitignore` 中忽略，不会被提交。

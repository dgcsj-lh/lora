import torch
import argparse
from pathlib import Path

from diffusers import StableDiffusionPipeline


# =====================================
# 1. 配置
# =====================================

# 本地模型目录（src/models/stable-diffusion-v1-5）
MODEL_ID = str(
    Path(__file__).resolve().parents[1]
    / "models"
    / "stable-diffusion-v1-5"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# =====================================
# 2. 统计模型参数
# =====================================

def count_parameters(model):

    total = sum(
        p.numel() for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def show_model_info(name, model):

    total, trainable = count_parameters(model)

    print(f"\n========== {name} ==========")

    print(f"参数总量: {total:,}")

    print(f"可训练参数: {trainable:,}")

    print(f"参数量: {total / 1e6:.2f} M")

    print(f"数据类型: {next(model.parameters()).dtype}")


# =====================================
# 3. 加载 SD1.5
# =====================================

def load_model(model_id, cpu_offload=False):

    print("\n开始加载 Stable Diffusion 1.5...")

    print("Model:", model_id)

    print("Device:", DEVICE)

    print("Dtype:", DTYPE)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        use_safetensors=True
    )

    # 显存优化
    # diffusers 0.43 起，VAE slicing 直接开在 vae 组件上
    pipe.vae.enable_slicing()

    if DEVICE == "cuda":

        if cpu_offload:

            print("\n启用 CPU Offload")

            pipe.enable_model_cpu_offload()

        else:

            pipe = pipe.to(DEVICE)

    print("\n模型加载成功！")

    return pipe


# =====================================
# 4. 查看模型组件
# =====================================

def inspect_model(pipe):

    print("\n\n========== SD1.5 模型结构 ==========")

    print("VAE:", type(pipe.vae).__name__)

    print("Text Encoder:", type(pipe.text_encoder).__name__)

    print("UNet:", type(pipe.unet).__name__)

    print("Scheduler:", type(pipe.scheduler).__name__)

    print("Tokenizer:", type(pipe.tokenizer).__name__)

    # 统计参数
    show_model_info("VAE", pipe.vae)

    show_model_info("CLIP Text Encoder", pipe.text_encoder)

    show_model_info("U-Net", pipe.unet)

    # 总参数
    total = sum(
        p.numel()
        for p in pipe.vae.parameters()
    )

    total += sum(
        p.numel()
        for p in pipe.text_encoder.parameters()
    )

    total += sum(
        p.numel()
        for p in pipe.unet.parameters()
    )

    print("\n========== 总参数 ==========")

    print(f"总参数量: {total:,}")

    print(f"总参数量: {total / 1e6:.2f} M")


# =====================================
# 5. 测试生成图片
# =====================================

def generate_image(pipe):

    print("\n开始生成图片...")

    prompt = (
        "a beautiful anime girl, "
        "long silver hair, blue eyes, "
        "masterpiece, best quality"
    )

    negative_prompt = (
        "low quality, blurry, "
        "bad anatomy, extra fingers"
    )

    generator = torch.Generator(
        device=DEVICE
    ).manual_seed(42)

    with torch.inference_mode():

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

    image.save("sd15_test.png")

    print("\n图片保存成功: sd15_test.png")


# =====================================
# 6. 主函数
# =====================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="Hugging Face 模型 ID 或本地模型路径"
    )

    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="使用 CPU Offload 节省显存"
    )

    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="只加载模型，不生成图片"
    )

    args = parser.parse_args()

    # 加载模型
    pipe = load_model(
        args.model,
        args.cpu_offload
    )

    # 查看模型
    inspect_model(pipe)

    # 显示显存
    if DEVICE == "cuda":

        print("\n========== GPU 显存 ==========")

        print(
            "已分配:",
            torch.cuda.memory_allocated() / 1024**3,
            "GB"
        )

        print(
            "缓存:",
            torch.cuda.memory_reserved() / 1024**3,
            "GB"
        )

    # 生成图片
    if not args.no_generate:

        generate_image(pipe)

    print("\nSD1.5 实验结束！")


if __name__ == "__main__":

    main()
import argparse
import os
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig


VALID_OPTIONS = {"1", "2", "3", "4", "5"}
DEFAULT_MODEL_DIR = "models/qwen-vl"
PROMPT = """You are solving a deep-learning multiple-choice question shown in the image.
Read the question and all four options carefully. Choose exactly one option.

Return only one digit:
1 for option A
2 for option B
3 for option C
4 for option D
5 if the image is unreadable or you are not confident.

No words, no punctuation, no explanation."""


def get_args():
    parser = argparse.ArgumentParser(description="VQA inference for MCQ images.")
    parser.add_argument("--test_dir", required=True, help="Absolute path to test directory.")
    parser.add_argument("--model_dir", default=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--output_file", default="submission.csv")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    return parser.parse_args()


def find_images_dir(test_dir: Path) -> Path:
    for name in ("images", "image"):
        candidate = test_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No images/ or image/ folder found under {test_dir}")


def normalize_image_name(raw_name: str) -> str:
    path = Path(str(raw_name))
    return path.name


def image_path_for(images_dir: Path, image_name: str) -> Path:
    name = normalize_image_name(image_name)
    direct = images_dir / name
    if direct.exists():
        return direct
    stem = Path(name).stem
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image for {image_name} in {images_dir}")


def should_use_4bit() -> bool:
    env_value = os.environ.get("USE_4BIT", "").strip().lower()
    if env_value in {"1", "true", "yes", "y"}:
        return True
    if env_value in {"0", "false", "no", "n"}:
        return False
    if not torch.cuda.is_available():
        return False
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    total_gb = total_bytes / (1024**3)
    return total_gb < 16


def load_model(model_dir: str):
    from transformers import AutoModelForImageTextToText

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory {model_path} not found. Run setup.bash while internet is available."
        )
    safetensors = list(model_path.glob("*.safetensors"))
    if not safetensors:
        raise FileNotFoundError(
            f"Model weights not found in {model_path}. "
            f"Download is incomplete — wait for setup.bash to finish."
        )

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )

    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)} — running on GPU.")
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if should_use_4bit():
            print("VRAM < 16 GB — loading in 4-bit quantization.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        except ValueError:
            print("WARNING: Not enough GPU VRAM even for 4-bit — falling back to CPU (slow).")
            model_kwargs.pop("quantization_config", None)
            model_kwargs["device_map"] = "cpu"
            model_kwargs["torch_dtype"] = torch.float32
            model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    else:
        print("No GPU detected — running on CPU (slow).")
        model_kwargs = {
            "device_map": "cpu",
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
            "trust_remote_code": True,
        }
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return processor, model


def build_inputs(processor, image: Image.Image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        from qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(text=[text], images=[image], padding=True, return_tensors="pt")


def move_inputs_to_model(inputs, model):
    device = next(model.parameters()).device
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def extract_option(text: str) -> str:
    cleaned = text.strip()
    match = re.search(r"\b([1-5])\b", cleaned)
    if match:
        return match.group(1)

    letter_map = {"A": "1", "B": "2", "C": "3", "D": "4"}
    match = re.search(r"\b([ABCD])\b", cleaned.upper())
    if match:
        return letter_map[match.group(1)]

    return "5"


@torch.inference_mode()
def predict_one(processor, model, image_path: Path, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = build_inputs(processor, image, PROMPT)
    inputs = move_inputs_to_model(inputs, model)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_len:]
    output_text = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    option = extract_option(output_text)
    return option if option in VALID_OPTIONS else "5"


def main():
    args = get_args()
    test_dir = Path(args.test_dir).expanduser().resolve()
    test_csv = test_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found at {test_csv}")

    images_dir = find_images_dir(test_dir)
    test_df = pd.read_csv(test_csv)
    if "image_name" not in test_df.columns:
        raise ValueError("test.csv must contain an image_name column")

    processor, model = load_model(args.model_dir)

    rows = []
    for image_name in test_df["image_name"].astype(str):
        path = image_path_for(images_dir, image_name)
        try:
            option = predict_one(processor, model, path, args.max_new_tokens)
        except Exception as exc:
            print(f"WARNING: failed on {image_name}: {exc}")
            option = "5"
        rows.append({"image_name": image_name, "option": int(option)})

    submission = pd.DataFrame(rows, columns=["image_name", "option"])
    submission.to_csv(args.output_file, index=False)
    print(f"Wrote {args.output_file} with {len(submission)} rows")


if __name__ == "__main__":
    main()

import os
from pathlib import Path
from typing import List, Union, Optional

import torch
from PIL import Image
import cv2

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# Model / processor loading

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Helpers

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _is_image(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def _is_video(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def _sample_video_frames(
    video_path: Union[str, Path],
    max_frames: int = 8,
) -> List[Image.Image]:
    """
    Sample up to `max_frames` frames evenly from a video
    and return them as a list of PIL Images in RGB.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count == 0:
        cap.release()
        raise RuntimeError(f"Video appears to have 0 frames: {video_path}")

    # Choose up to max_frames indices evenly spaced through the video
    indices = list(range(frame_count))
    if frame_count > max_frames:
        step = frame_count / max_frames
        indices = [int(i * step) for i in range(max_frames)]

    frames: List[Image.Image] = []

    for target_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        success, frame = cap.read()
        if not success:
            continue
        # Convert BGR via Opencv  -> RGB -> PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(pil_img)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from video: {video_path}")

    return frames


def _generate_from_images(
    images: List[Image.Image],
    prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """
    Core generation function using a list of PIL images.
    """
    # Build chat-style messages as in the official template
    # (multiple images followed by the user text) :contentReference[oaicite:1]{index=1}
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # Convert messages -> text prompt according to the model's chat template
    chat_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    # Create model inputs
    inputs = processor(
        text=chat_prompt,
        images=images,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    # Decode
    generated = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0]

    return generated


# Endpoint for user call

def generate_from_media(
    path: Union[str, Path],
    prompt: Optional[str] = None,
    max_new_tokens: int = 128,
) -> str:
    """
    Run SmolVLM-Instruct on an image OR video and return generated text.

    Args:
        path: Path to an image or video file.
        prompt:
            - For images: e.g. "Describe this image." or "What is happening here?"
            - For videos (handled via sampled frames): e.g. "Describe this video."
            If None, a reasonable default is used based on media type.
        max_new_tokens: Max tokens to generate.

    Returns:
        The generated text (full assistant turn, including "Assistant:" prefix).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if _is_image(path):
        if prompt is None:
            prompt = "Describe this image in detail."
        # Use HF utility to load image
        image = load_image(str(path))
        images = [image]
        return _generate_from_images(images, prompt, max_new_tokens=max_new_tokens)

    if _is_video(path):
        if prompt is None:
            prompt = "Describe this video in detail."
        frames = _sample_video_frames(path, max_frames=8)
        return _generate_from_images(frames, prompt, max_new_tokens=max_new_tokens)

    raise ValueError(
        f"Unsupported file extension: {path.suffix}. "
        f"Supported image types: {sorted(IMAGE_EXTS)}; "
        f"supported video types: {sorted(VIDEO_EXTS)}"
    )
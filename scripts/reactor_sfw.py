import time
from transformers import pipeline
from PIL import Image
import io
import logging
import os
import comfy.model_management as model_management
from reactor_utils import download
from scripts.reactor_logger import logger

MODEL_EXISTS = False

def ensure_nsfw_model(nsfwdet_model_path):
    """Download NSFW detection model if it doesn't exist"""
    global MODEL_EXISTS
    downloaded = 0
    nd_urls = [
        "https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/config.json",
        "https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/model.safetensors",
        "https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/preprocessor_config.json",
    ]
    for model_url in nd_urls:
        model_name = os.path.basename(model_url)
        model_path = os.path.join(nsfwdet_model_path, model_name)
        if not os.path.exists(model_path):
            if not os.path.exists(nsfwdet_model_path):
                os.makedirs(nsfwdet_model_path)
            download(model_url, model_path, model_name)
        if os.path.exists(model_path):
            downloaded += 1
    MODEL_EXISTS = True if downloaded == 3 else False
    return MODEL_EXISTS

SCORE = 0.96  # Default threshold

logging.getLogger("transformers").setLevel(logging.ERROR)

def validate_nsfw_threshold(threshold: float) -> float:
    """
    Validate and normalize NSFW threshold parameter.

    Args:
        threshold: Threshold value to validate

    Returns:
        float: Validated threshold value

    Raises:
        ValueError: If threshold is outside valid range
    """
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"NSFW threshold must be a number, got {type(threshold)}")

    threshold = float(threshold)

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"NSFW threshold must be between 0.0 and 1.0, got {threshold}")

    return threshold

def nsfw_image(img_data, model_path: str, threshold: float = 0.96):
    """
    Detect NSFW content in an image using a configurable threshold.

    Args:
        img_data: Image data as bytes
        model_path: Path to the NSFW detection model
        threshold: Detection threshold (0.0-1.0), default 0.96

    Returns:
        bool: True if NSFW content detected above threshold, False otherwise
    """
    start_time = time.time()

    # Validate threshold parameter
    threshold = validate_nsfw_threshold(threshold)

    if not MODEL_EXISTS:
        logger.status("Ensuring NSFW detection model exists...")
        if not ensure_nsfw_model(model_path):
            return True
    device = model_management.get_torch_device()
    with Image.open(io.BytesIO(img_data)) as img:
        if "cpu" in str(device):
            predict = pipeline("image-classification", model=model_path)
        else:
            device_id = 0
            if "cuda" in str(device):
                device_id = int(str(device).split(":")[1])
            predict = pipeline("image-classification", model=model_path, device=device_id)

        result = predict(img)
        nsfw_score = result[0]["score"] if result[0]["label"] == "nsfw" else (1.0 - result[0]["score"])
        is_nsfw = result[0]["label"] == "nsfw" and nsfw_score > threshold

        detection_time = time.time() - start_time

        logger.status(f'NSFW detection: score={nsfw_score:.3f}, threshold={threshold:.3f} - {"NSFW" if is_nsfw else "SFW"} ({detection_time:.2f}s)')

        if is_nsfw:
            logger.status(f'NSFW content detected with score={nsfw_score:.3f} (threshold: {threshold:.3f}), skipping...')
            return True
        return False

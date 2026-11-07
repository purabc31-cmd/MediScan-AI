from io import BytesIO
import base64
import numpy as np
import cv2
from PIL import Image


def png_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def segmentation_overlay(original: Image.Image, mask: np.ndarray) -> Image.Image:
    base = original.convert("RGB").resize((mask.shape[1], mask.shape[0]))
    arr = np.asarray(base).copy()
    active = mask > 0.5
    # Muted teal overlay; this is a visualization of model output, not a contour diagnosis.
    tint = np.array([65, 145, 128], dtype=np.uint8)
    arr[active] = (0.55 * arr[active] + 0.45 * tint).astype(np.uint8)
    return Image.fromarray(arr)


def gradcam_overlay(original: Image.Image, cam: np.ndarray) -> Image.Image:
    width, height = original.size
    heat = cv2.resize(cam, (width, height))
    heat = np.clip(heat * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    base = np.asarray(original.convert("RGB"))
    overlay = (0.58 * base + 0.42 * colored).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

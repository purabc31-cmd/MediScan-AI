from io import BytesIO
import numpy as np
from PIL import Image
import pydicom


def decode_medical_image(raw: bytes, filename: str):
    """Decode PNG/JPEG or a single-slice DICOM into a displayable grayscale PIL image."""
    lower = (filename or "").lower()
    if lower.endswith(".dcm") or raw[128:132] == b"DICM":
        dataset = pydicom.dcmread(BytesIO(raw), force=True)
        pixels = dataset.pixel_array
        if pixels.ndim == 3:
            pixels = pixels[pixels.shape[0] // 2]
        pixels = pixels.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        pixels = pixels * slope + intercept
        low, high = np.percentile(pixels, [1, 99])
        if high <= low:
            high = low + 1.0
        pixels = np.clip((pixels - low) / (high - low), 0, 1)
        array = (pixels * 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")

    image = Image.open(BytesIO(raw)).convert("L")
    return image

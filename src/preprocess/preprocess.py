import cv2
import numpy as np
from skimage.filters import frangi
from skimage import exposure
from typing import Tuple


def load_image(path: str) -> np.ndarray:
    # Supports URI placeholders in demo; for real paths use cv2.imread
    if path.startswith("placeholder://"):
        # return a synthetic image for demo
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(img, (256, 256), 100, (255, 255, 255), -1)
        return img
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def clahe_color(img: np.ndarray) -> np.ndarray:
    # Convert to LAB and apply CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return result


def frangi_vessel_enhance(gray: np.ndarray) -> np.ndarray:
    # skimage frangi expects float in [0,1]
    imgf = gray.astype(np.float32) / 255.0
    fr = frangi(imgf)
    # normalize to 0-255
    frn = (255 * (fr - fr.min()) / (fr.max() - fr.min() + 1e-9)).astype(np.uint8)
    return frn


def resize_image(img: np.ndarray, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def preprocess_image(path: str) -> dict:
    img = load_image(path)
    resized = resize_image(img)
    clahe = clahe_color(resized)
    gray = cv2.cvtColor(clahe, cv2.COLOR_BGR2GRAY)
    vessels = frangi_vessel_enhance(gray)
    # Compute a simple quality metric: variance of Laplacian (focus)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    quality = {
        "quality_score": min(100, int(np.clip((fm / 100.0) * 100, 0, 100))),
        "gradable": fm > 50,
        "focus_measure": float(fm),
    }
    return {
        "preprocessed_image": clahe,
        "vessel_map": vessels,
        **quality,
    }

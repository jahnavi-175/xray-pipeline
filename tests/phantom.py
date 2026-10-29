"""Synthetic chest-like image with known lung masks (for unit tests / smoke tests)."""
import cv2
import numpy as np


def make_phantom(seed=0, noise=4.0, variant="centered", size=256):
    """variant: 'centered' (black margin all round), 'wide' (body fills the frame sideways, as in
    tightly cropped X-rays), 'touch' (lungs reach the image border), 'tight' (real-data-like tight crop)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    gt = np.zeros((size, size), np.uint8)
    if variant == "tight":      # tight crop: body fills the frame, lungs touch left/right edges,
        img[:] = 165            # dark air in the top corners, gas bubble under the left lung
        cv2.fillPoly(img, [np.array([[0, 0], [70, 0], [0, 60]]), np.array([[size, 0], [size - 70, 0], [size, 60]])], 5)
        for cx in (60, size - 60):
            cv2.ellipse(img, (cx, 125), (58, 90), 0, 0, 360, 70, -1)
            cv2.ellipse(gt, (cx, 125), (58, 90), 0, 0, 360, 1, -1)
        cv2.ellipse(img, (80, 240), (26, 14), 0, 0, 360, 60, -1)
    else:
        body_rx = {"centered": 96, "wide": 150, "touch": 170}[variant]
        cv2.ellipse(img, (size // 2, size // 2 + 10), (body_rx, 118), 0, 0, 360, 165, -1)   # body
        dx = {"centered": 48, "wide": 78, "touch": 100}[variant]
        for cx in (size // 2 - dx, size // 2 + dx):
            cv2.ellipse(img, (cx, size // 2), (34, 72), 0, 0, 360, 70, -1)                # lungs (dark)
            cv2.ellipse(gt, (cx, size // 2), (34, 72), 0, 0, 360, 1, -1)
    cv2.rectangle(img, (size // 2 - 8, 30), (size // 2 + 8, size - 20), 200, -1)       # spine
    for y in range(70, 190, 22):                                                         # ribs over lungs
        m = np.zeros_like(img); cv2.line(m, (20, y), (size - 20, y + 14), 1, 5)
        img += m * 22 * (gt > 0)
    img = cv2.GaussianBlur(img, (0, 0), 2.0) + rng.normal(0, noise, img.shape)
    img = np.clip(img * rng.uniform(0.85, 1.1), 0, 255).astype(np.uint8)
    return img, gt

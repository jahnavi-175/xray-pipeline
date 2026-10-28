"""Figures for the reports."""
import numpy as np
import matplotlib.pyplot as plt


def before_after_grid(original, variants, hist=True, title=None, path=None):
    """Row of images (original + each variant); optional row of histograms below."""
    items = [("original", original)] + list(variants.items())
    n = len(items)
    rows = 2 if hist else 1
    fig, ax = plt.subplots(rows, n, figsize=(2.8 * n, 2.8 * rows + 0.3), squeeze=False)
    for j, (name, im) in enumerate(items):
        ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=255)
        ax[0, j].set_title(name, fontsize=10); ax[0, j].axis("off")
        if hist:
            ax[1, j].hist(im.ravel(), bins=64, range=(0, 255), color="steelblue")
            ax[1, j].set_yticks([]); ax[1, j].set_xlim(0, 255)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def log_spectrum(img):
    F = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    return np.log1p(np.abs(F))


def spectra_grid(original, variants, path=None):
    items = [("original", original)] + list(variants.items())
    fig, ax = plt.subplots(1, len(items), figsize=(2.8 * len(items), 3), squeeze=False)
    for j, (name, im) in enumerate(items):
        ax[0, j].imshow(log_spectrum(im), cmap="magma"); ax[0, j].set_title(name, fontsize=10)
        ax[0, j].axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def zoom_crops(images: dict, box, path=None):
    """Compare fine detail: crop the same box (r0, r1, c0, c1) from each image."""
    r0, r1, c0, c1 = box
    fig, ax = plt.subplots(1, len(images), figsize=(2.8 * len(images), 3), squeeze=False)
    for j, (name, im) in enumerate(images.items()):
        ax[0, j].imshow(im[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=255)
        ax[0, j].set_title(name, fontsize=10); ax[0, j].axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig

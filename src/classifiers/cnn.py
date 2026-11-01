"""ResNet18 transfer-learning classifier (needs PyTorch + torchvision). Trained by scripts/train_cnn.py."""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision


class Net(nn.Module):
    """ResNet18 on 1-channel [0,1] images (replicated to 3 channels + ImageNet normalisation inside)."""

    def __init__(self, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.m = torchvision.models.resnet18(weights=w)
        self.m.fc = nn.Linear(512, 2)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.m((x.repeat(1, 3, 1, 1) - self.mean) / self.std)


class CNNPredictor:
    """Inference wrapper: uint8 grayscale image -> probability of 'abnormal'."""

    def __init__(self, model, size, enh, device="cpu"):
        self.model, self.size, self.enh, self.device = model.to(device).eval(), size, enh, device

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu")
        model = Net(pretrained=False)                 # weights come from the checkpoint, no download
        model.load_state_dict(ck["state"])
        return cls(model, ck["size"], ck["enh"], device)

    @torch.no_grad()
    def predict_prob(self, img):
        x = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        t = torch.from_numpy(x)[None, None].to(self.device)
        return float(torch.softmax(self.model(t), 1)[0, 1].cpu())

import torch.nn as nn

def build_model(num_classes=6):
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 96 -> 48
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 48 -> 24
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(32, num_classes),
    )
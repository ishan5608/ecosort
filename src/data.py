from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT/"data"/"public"

def get_loaders(batch_size = 64):
    transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_set = datasets.ImageFolder(DATA_DIR/"train",transform = transform)
    val_set = datasets.ImageFolder(DATA_DIR/"val",transform = transform)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False)
    return train_loader, val_loader
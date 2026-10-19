import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from src.model import build_model

ROOT_DATA = "data/public/val"
CKPT = "checkpoints/cnn_v2_norm_best.pt"
ONNX = "models/model.onnx"

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
val_set = datasets.ImageFolder(ROOT_DATA, transform=transform)
loader = DataLoader(val_set, batch_size=64, shuffle=False)

# 1. PyTorch model
model = build_model()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

labels, pt_preds = [], []
with torch.no_grad():
    for X, y in loader:
        pt_preds.append(model(X).argmax(1).numpy())
        labels.append(y.numpy())
labels = np.concatenate(labels)
pt_preds = np.concatenate(pt_preds)
print("PyTorch accuracy:", (pt_preds == labels).mean())

# 2. Per-class accuracy and confusion matrix
classes = val_set.classes
cm = np.zeros((6, 6), dtype=int)
for t, p in zip(labels, pt_preds):
    cm[t, p] += 1
for i, name in enumerate(classes):
    print(f"{name:>10}: {cm[i, i] / cm[i].sum():.3f}  ({cm[i].sum()} images)")
print("Confusion matrix (rows = true, cols = predicted):")
print(cm)

# 3. ONNX model on the same images
session = ort.InferenceSession(ONNX)
onnx_preds = []
for X, _ in loader:
    out = session.run(None, {session.get_inputs()[0].name: X.numpy()})[0]
    onnx_preds.append(out.argmax(1))
onnx_preds = np.concatenate(onnx_preds)
print("ONNX accuracy:", (onnx_preds == labels).mean())
print("Predictions that differ between PyTorch and ONNX:", int((onnx_preds != pt_preds).sum()))
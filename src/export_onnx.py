"""
Export the best checkpoint to ONNX.

Run from the project root:
    python -m src.export_onnx --checkpoint checkpoints/cnn_v1_best.pt
"""

import argparse
from pathlib import Path

import torch

from src.model import build_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="models/model.onnx")
    args = parser.parse_args()

    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy_input = torch.zeros(1, 3, 96, 96)
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        external_data=False,
    )
    print(f"Exported to {output_path}")


if __name__ == "__main__":
    main()
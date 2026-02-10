import argparse
import os
import sys
import torch
from firesam.models import LimFUNetFire


def export_student_onnx(args):
    if args.dtype == "fp16" and not torch.cuda.is_available():
        raise SystemExit("FP16 export requires CUDA (PyTorch FP16 ops on CPU are often unsupported).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LimFUNetFire(in_channels=6, num_classes=1)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    if args.dtype == "fp16":
        model = model.half()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dummy_input = torch.randn(1, 6, args.height, args.width, device=device, dtype=torch.float16 if args.dtype == "fp16" else torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"Exported {args.dtype.upper()} ONNX model to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export LimFUNet-Fire student to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Student checkpoint path.")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX file path.")
    parser.add_argument("--height", type=int, default=416)
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_student_onnx(args)

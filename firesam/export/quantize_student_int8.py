import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from firesam.data.datasets import FireSegmentationDataset
from firesam.utils.prompts import sample_prompts_from_torch_mask, rasterize_prompts
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)


class FirePromptCalibReader(CalibrationDataReader):
    def __init__(self, split_file: str, height: int, width: int, num_calib: int, seed: int):
        ds = FireSegmentationDataset(split_file, image_size=(height, width), augment=False)
        rng = np.random.default_rng(seed)
        k = min(num_calib, len(ds))
        indices = rng.choice(len(ds), size=k, replace=False)
        self._inputs = []
        for i in indices:
            sample = ds[int(i)]
            img = sample["image"]          
            mask = sample["mask"]          
            # Prompts from GT for calibration
            box, pos_pts, neg_pts = sample_prompts_from_torch_mask(mask, num_pos=2, num_neg=2)
            prompts = rasterize_prompts(height, width, box, pos_pts, neg_pts)
            x = torch.cat([img, prompts], dim=0).unsqueeze(0).numpy().astype(np.float32)
            self._inputs.append({"input": x})
        self._iter = iter(self._inputs)
    def get_next(self):
        return next(self._iter, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="FP32 ONNX input path")
    ap.add_argument("--output", required=True, help="INT8 ONNX output path")
    ap.add_argument("--calib_split", required=True, help="Split txt: 'img_rel mask_rel' per line")
    ap.add_argument("--num_calib", type=int, default=200)
    ap.add_argument("--height", type=int, default=416)
    ap.add_argument("--width", type=int, default=608)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dr = FirePromptCalibReader(
        split_file=args.calib_split,
        height=args.height,
        width=args.width,
        num_calib=args.num_calib,
        seed=args.seed,
    )

    quantize_static(
        model_input=args.input,
        model_output=args.output,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"Exported INT8 QDQ ONNX to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import torch
import coremltools as ct

from perception_models.tools.convert import CoreMLConverter


def main():
    out_dir = Path("coreml_models")
    out_dir.mkdir(exist_ok=True)
    model_name = "PE-Core-T16-384"

    print("🚀 TorchScript→CoreML Conversion (bypassing ONNX)")
    print(f"Model: {model_name}")

    conv = CoreMLConverter()
    model = conv.load_pe_core_model(model_name)
    if model is None:
        raise SystemExit("Failed to load PE Core model")

    mobile_model = conv.create_mobile_wrapper(model, model_name)
    mobile_model.eval()

    example_input = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        y = mobile_model(example_input)
        print("Wrapper output shape:", tuple(y.shape))

    print("Tracing TorchScript...")
    traced = torch.jit.trace(mobile_model, example_input)

    print("Converting to CoreML (mlprogram, fp16, ALL compute units)...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=example_input.shape, name="image")],
        outputs=[ct.TensorType(name="features")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    out_path = out_dir / "pe_core_t16_fp16.mlpackage"
    print("Saving:", out_path)
    mlmodel.save(str(out_path))

    # Quick validation run via CoreML
    try:
        print("Validating CoreML model with random input...")
        inp = np.random.randn(1, 3, 384, 384).astype(np.float32)
        # coremltools MLModel predict expects dict of input name to array
        out = mlmodel.predict({"image": inp})
        print("Prediction keys:", list(out.keys()))
        feats = out.get("features")
        if feats is not None:
            print("Output shape:", getattr(feats, "shape", None))
            print("✅ CoreML model runs")
        else:
            print("⚠️ CoreML output 'features' not found in prediction")
    except Exception as e:
        print("⚠️ CoreML validation run failed:", e)


if __name__ == "__main__":
    main()


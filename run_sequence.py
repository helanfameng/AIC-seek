import argparse
import json
from pathlib import Path

import torch
from temporal_model.core.model import BboxTubeTemporalModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    sequence_dir = Path(args.sequence)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not sequence_dir.is_dir():
        raise SystemExit(f"Sequence directory not found: {sequence_dir}")
    if not model_path.is_file():
        raise SystemExit(f"Model package not found: {model_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check the PyTorch installation.")

    extensions = {".jpg", ".jpeg", ".png"}
    frames = sorted(
        p for p in sequence_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not frames:
        raise SystemExit("No JPG/PNG images found.")

    print("Frames:", len(frames))
    print("First:", frames[0].name)
    print("Last:", frames[-1].name)
    print("Device:", args.device)

    model = BboxTubeTemporalModel.from_package(
        model_path, device=args.device
    )
    result = model.predict_sequence(frames)

    kept = result.details.get("tubes", {}).get("kept", [])
    probabilities = [
        float(t["probability"])
        for t in kept
        if t.get("probability") is not None
    ]

    report = {
        "sequence": str(sequence_dir),
        "frame_names": [p.name for p in frames],
        "is_smoke": bool(result.is_positive),
        "max_kept_probability": max(probabilities) if probabilities else None,
        "kept_tube_count": len(kept),
        "details": result.details,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("Smoke verdict:", report["is_smoke"])
    print("Max kept probability:", report["max_kept_probability"])
    print("Kept tubes:", report["kept_tube_count"])
    print("Saved:", output_path)


if __name__ == "__main__":
    main()

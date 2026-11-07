# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "pnnx",
#     "spandrel",
#     "requests",
#     "numpy",
# ]
# ///
"""
Batch convert all PyTorch (.pth) models in the current directory to NCNN (.param + .bin)
for use with Upscayl / upscayl-bin.exe.

Uses Spandrel for universal architecture & checkpoint loading (ESRGAN, ESRGAN+, Compact, DAT, etc.).

Usage:
    uv run batch_convert.py
"""

import gc
import sys
from pathlib import Path

import torch


def rename_io_blobs(param_path: Path, input_name: str = "data", output_name: str = "output") -> None:
    lines = param_path.read_text().splitlines()
    if not lines or len(lines) < 3:
        return
    header, counts = lines[0], lines[1]
    layer_lines = [l for l in lines[2:] if l.strip()]

    parsed = []
    for line in layer_lines:
        tokens = line.split()
        ltype, lname, n_in, n_out = tokens[0], tokens[1], int(tokens[2]), int(tokens[3])
        n_blob = n_in + n_out
        blobs = tokens[4:4 + n_blob]
        params = tokens[4 + n_blob:]
        parsed.append([ltype, lname, n_in, n_out, blobs, params])

    if not parsed:
        return

    old_input_blob = parsed[0][4][-1]
    old_output_blob = parsed[-1][4][-1]
    rename_map = {old_input_blob: input_name, old_output_blob: output_name}

    for entry in parsed:
        entry[4] = [rename_map.get(b, b) for b in entry[4]]

    new_lines = [header, counts]
    for ltype, lname, n_in, n_out, blobs, params in parsed:
        new_lines.append(" ".join([ltype, lname, str(n_in), str(n_out), *blobs, *params]))

    param_path.write_text("\n".join(new_lines) + "\n")


def clean_intermediates(stem: str):
    sanitized = stem.replace("-", "_")
    for pattern in [f"{stem}*", f"{sanitized}*"]:
        for p in Path(".").glob(pattern):
            if p.suffix in [".pt", ".onnx", ".py"] or ".pnnx." in p.name or ".ncnn." in p.name:
                if p.suffix not in [".param", ".bin"] or ".ncnn." in p.name:
                    p.unlink(missing_ok=True)


def convert_model(pth_path: Path, trace_size: int = 32) -> None:
    stem = pth_path.stem
    param_out = Path(f"{stem}.param")
    bin_out = Path(f"{stem}.bin")

    print(f"\n==================================================")
    print(f" Processing: {pth_path.name}")
    print(f"==================================================")

    from spandrel import ModelLoader

    loader = ModelLoader(device=torch.device("cpu"))
    descriptor = loader.load_from_file(str(pth_path))
    
    model = descriptor.model.eval().cpu()
    arch_name = descriptor.architecture
    scale = descriptor.scale

    print(f" Detected Architecture: {arch_name} | Scale: {scale}x")
    print(f" Exporting to NCNN FP16 via PNNX...")

    import pnnx

    dummy = torch.rand(1, 3, trace_size, trace_size)
    traced_pt = Path(f"{stem}.pt")

    clean_intermediates(stem)

    pnnx.export(
        model,
        str(traced_pt),
        inputs=(dummy,),
        ncnnparam=f"{stem}.ncnn.param",
        ncnnbin=f"{stem}.ncnn.bin",
        ncnnpy=f"{stem}_ncnn.py",
        fp16=True,
    )

    ncnn_param = Path(f"{stem}.ncnn.param")
    ncnn_bin = Path(f"{stem}.ncnn.bin")

    if not ncnn_param.exists() or not ncnn_bin.exists():
        print(f"ERROR: Failed to produce NCNN files for {pth_path.name}")
        return

    ncnn_param.rename(param_out)
    ncnn_bin.rename(bin_out)
    rename_io_blobs(param_out)
    clean_intermediates(stem)

    del model, descriptor
    gc.collect()

    print(f" SUCCESS: Wrote {param_out.name} & {bin_out.name}")


def main() -> None:
    current_dir = Path(".")
    pth_files = sorted(current_dir.glob("*.pth"))

    if not pth_files:
        print("No .pth files found in current directory.")
        return

    print(f"Found {len(pth_files)} .pth model(s) to batch convert.")
    
    successful = 0
    failed = []

    for idx, pth in enumerate(pth_files, 1):
        print(f"\n[{idx}/{len(pth_files)}]", end="")
        try:
            convert_model(pth)
            successful += 1
        except Exception as e:
            print(f"\nFAILED to convert {pth.name}: {e}")
            failed.append(pth.name)

    print(f"\n" + "=" * 50)
    print(f"BATCH CONVERSION SUMMARY")
    print(f"=" * 50)
    print(f" Successfully converted: {successful}/{len(pth_files)}")
    if failed:
        print(f" Failed models ({len(failed)}):")
        for f_name in failed:
            print(f"   - {f_name}")
    print(f"\nDone! You can now use these models in Upscayl.")


if __name__ == "__main__":
    main()
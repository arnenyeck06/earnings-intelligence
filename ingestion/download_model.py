import os
import shutil
from pathlib import Path

DEST = Path("models/Xenova/all-MiniLM-L6-v2")
REPO = "Xenova/all-MiniLM-L6-v2"
ONNX_CANDIDATES = ["onnx/model.onnx", "onnx/encoder_model.onnx", "model.onnx"]


def download():
    from huggingface_hub import hf_hub_download, list_repo_files
    DEST.mkdir(parents=True, exist_ok=True)
    files = list(list_repo_files(repo_id=REPO))
    onnx_file = next((c for c in ONNX_CANDIDATES if c in files), None)
    if not onnx_file:
        raise FileNotFoundError(f"No ONNX model in {REPO}")
    for remote, local in [("tokenizer.json", "tokenizer.json"), (onnx_file, "model.onnx")]:
        dst = DEST / local
        if dst.exists():
            print(f"  exists: {dst}")
            continue
        src = hf_hub_download(repo_id=REPO, filename=remote)
        shutil.copy2(src, dst)
        print(f"  saved:  {dst}")
    print(f"\nModel ready at {DEST}")


if __name__ == "__main__":
    download()

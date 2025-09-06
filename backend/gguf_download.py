# download_gguf.py
import os
from huggingface_hub import snapshot_download

REPO_ID = "MaziyarPanahi/gemma-2b-GGUF"   # Example repo
FILENAME_PATTERN = "gemma-2b.Q4_K_M.gguf" # Exact GGUF filename
DEST_DIR = "backend/models"

os.makedirs(DEST_DIR, exist_ok=True)

print(f"Downloading {FILENAME_PATTERN} from {REPO_ID} ...")
local_path = snapshot_download(
    repo_id=REPO_ID,
    allow_patterns=FILENAME_PATTERN,
    local_dir=DEST_DIR,
    local_dir_use_symlinks=False,
)

print(f"✅ GGUF file downloaded to: {local_path}")

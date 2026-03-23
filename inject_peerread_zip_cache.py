from __future__ import annotations

import json
import shutil
from pathlib import Path

from datasets import config
from datasets.utils.file_utils import fsspec_head, hash_url_to_filename

# 只改这里：本地下载好的 PeerRead 压缩包路径
ZIP_PATH = Path("~/Download/PeerRead-master.zip")

# PeerRead 脚本里使用的固定下载地址（用于计算缓存文件名）
PEERREAD_URL = "https://github.com/allenai/PeerRead/archive/master.zip"


def main() -> None:
    src = ZIP_PATH.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Zip not found: {src}")

    info = fsspec_head(PEERREAD_URL)
    etag = info.get("ETag") or info.get("etag")

    cache_dir = Path(config.HF_DATASETS_CACHE) / "downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Write multiple cache-key variants to maximize cache hit reliability:
    # 1) key computed with current ETag
    # 2) key without ETag
    # 3) any already-created *.incomplete stem for this URL prefix
    base_key = hash_url_to_filename(PEERREAD_URL, None)
    key_with_etag = hash_url_to_filename(PEERREAD_URL, etag)

    target_keys: set[str] = {base_key, key_with_etag}
    for incomplete in cache_dir.glob(f"{base_key}*.incomplete"):
        target_keys.add(incomplete.name.removesuffix(".incomplete"))

    for key in sorted(target_keys):
        dst = cache_dir / key
        meta = cache_dir / f"{key}.json"
        shutil.copy2(src, dst)
        meta.write_text(json.dumps({"url": PEERREAD_URL, "etag": etag}), encoding="utf-8")
        print(f"Injected zip into cache: {dst}")
        print(f"Metadata file: {meta}")

    print(f"HF_DATASETS_CACHE: {config.HF_DATASETS_CACHE}")


if __name__ == "__main__":
    main()

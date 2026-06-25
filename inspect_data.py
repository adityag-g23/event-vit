"""Run this on one sample after downloading PEOD to reveal the exact file formats.

Usage:
    python inspect_data.py --data_root /path/to/PEOD
"""

import argparse
import os
import glob
import struct
import numpy as np


def find_first(root, pattern):
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return hits[0] if hits else None


def inspect_png(path):
    from PIL import Image
    img = Image.open(path)
    print(f"  PNG  shape={img.size}  mode={img.mode}")


def inspect_npy(path):
    arr = np.load(path, allow_pickle=True)
    print(f"  NPY  shape={arr.shape}  dtype={arr.dtype}")
    print(f"       first row: {arr[0] if len(arr) else 'empty'}")


def inspect_txt(path):
    with open(path) as f:
        lines = f.readlines()[:5]
    print(f"  TXT  {len(lines)} lines (first 3):")
    for l in lines[:3]:
        print(f"       {l.rstrip()}")


def inspect_dat(path):
    size = os.path.getsize(path)
    print(f"  DAT  file size = {size} bytes")
    with open(path, "rb") as f:
        raw = f.read(min(512, size))

    # --- Try ASCII header (Prophesee RAW format starts with % lines) ---
    header_end = 0
    if raw[:1] == b"%":
        lines = raw.split(b"\n")
        for i, line in enumerate(lines):
            if not line.startswith(b"%"):
                header_end = sum(len(l) + 1 for l in lines[:i])
                break
        print(f"  DAT  Prophesee-style ASCII header ({header_end} bytes):")
        for l in raw[:header_end].split(b"\n")[:6]:
            print(f"       {l.decode(errors='replace')}")
    else:
        print(f"  DAT  No ASCII header — first 32 raw bytes (hex):")
        print(f"       {raw[:32].hex()}")
        print(f"  DAT  first 32 bytes as uint8:  {np.frombuffer(raw[:32], dtype=np.uint8)}")

    body = raw[header_end:]
    print(f"  DAT  body starts at byte {header_end}, body sample hex: {body[:32].hex()}")

    # Try common event struct sizes
    for event_bytes, desc, dtype in [
        (8,  "t(u4) x(u2) y(u2)",          np.dtype([("t","<u4"),("x","<u2"),("y","<u2")])),
        (8,  "x(u2) y(u2) t(u4)",          np.dtype([("x","<u2"),("y","<u2"),("t","<u4")])),
        (12, "t(u4) x(u2) y(u2) p(u1)+pad",np.dtype([("t","<u4"),("x","<u2"),("y","<u2"),("p","u1"),("_","u1"),("__","u2")])),
    ]:
        body_size = size - header_end
        n_events = body_size // event_bytes
        remainder = body_size % event_bytes
        print(f"  DAT  if {event_bytes}-byte events ({desc}): n={n_events}, remainder={remainder} bytes")

    # Try numpy load directly
    try:
        arr = np.fromfile(path, dtype=np.uint8)
        print(f"  DAT  raw uint8 array length: {len(arr)}")
    except Exception as e:
        print(f"  DAT  numpy fromfile failed: {e}")


def inspect_annotations(ann_root):
    print("\n── Annotations ──")
    all_files = []
    for ext in ("*.npy", "*.json", "*.txt", "*.xml", "*.csv"):
        all_files += glob.glob(os.path.join(ann_root, "**", ext), recursive=True)
    all_files = sorted(all_files)[:5]
    if not all_files:
        print("  No annotation files found under", ann_root)
        return
    for p in all_files:
        print(f"\n  File: {p}")
        ext = os.path.splitext(p)[1].lower()
        if ext == ".npy":
            inspect_npy(p)
        elif ext in (".txt", ".csv"):
            inspect_txt(p)
        elif ext == ".json":
            import json
            with open(p) as f:
                d = json.load(f)
            keys = list(d.keys()) if isinstance(d, dict) else f"list of {len(d)}"
            print(f"  JSON keys/len: {keys}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to downloaded PEOD folder")
    args = parser.parse_args()
    root = args.data_root

    print("=" * 60)
    print("PEOD DATA INSPECTOR")
    print("=" * 60)

    # --- RGB ---
    print("\n── RGB image ──")
    png = find_first(os.path.join(root, "rgb"), "*.png")
    if png:
        print(f"  File: {png}")
        inspect_png(png)
    else:
        print("  No PNG found under rgb/")

    # --- Event DAT ---
    print("\n── Event .dat file ──")
    dat = find_first(os.path.join(root, "event"), "*.dat")
    if dat:
        print(f"  File: {dat}")
        inspect_dat(dat)
    else:
        print("  No .dat found under event/")

    # --- Timestamps ---
    print("\n── Timestamp files ──")
    ts = find_first(os.path.join(root, "timestamp"), "*")
    if ts:
        print(f"  File: {ts}")
        ext = os.path.splitext(ts)[1].lower()
        if ext == ".npy":
            inspect_npy(ts)
        else:
            inspect_txt(ts)
    else:
        print("  No timestamp files found")

    # --- Annotations ---
    inspect_annotations(os.path.join(root, "annotations"))

    # --- Top-level folder structure ---
    print("\n── Top-level folder tree (2 levels) ──")
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            children = sorted(os.listdir(full))[:6]
            print(f"  {entry}/  ({len(os.listdir(full))} items)")
            for c in children:
                cc = os.path.join(full, c)
                n = len(os.listdir(cc)) if os.path.isdir(cc) else ""
                print(f"    {c}{'/' if os.path.isdir(cc) else ''}  {n}")
        else:
            print(f"  {entry}")

    print("\nPaste the output above back to Claude to get dataset.py updated.")


if __name__ == "__main__":
    main()

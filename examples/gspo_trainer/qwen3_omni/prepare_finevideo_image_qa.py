# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build an image-only multiple-choice RL parquet from FineVideo (phase 1).

Each sample = N uniformly sampled frames + a 4-way multiple-choice question over
the video's category label. The answer is a single letter (verifiable, exact
match), and the task requires looking at the frames — good for the input-ablation
correctness check.

Usage:
  # 1) First inspect the real schema (no build), confirm the label field:
  python prepare_finevideo_image_qa.py --inspect
  # 2) Then build:
  python prepare_finevideo_image_qa.py --output-dir ~/data/finevideo_img \
      --max-samples 4000 --num-frames 4 --label-field content_parent_category
"""

import argparse
import io
import json
import os
import random
import tempfile

LETTERS = ["A", "B", "C", "D"]
# Candidate label fields, most-verifiable first; overridden by --label-field.
_LABEL_CANDIDATES = ["content_parent_category", "content_fine_category", "content_type"]


def _get_metadata(sample):
    """FineVideo stores metadata under 'json' (dict) alongside the 'mp4' bytes."""
    for key in ("json", "metadata"):
        meta = sample.get(key)
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                continue
        if isinstance(meta, dict):
            return meta
    return None


def _find_label(meta, label_field):
    """Return the category label string, searching top-level then content_metadata."""
    fields = [label_field] if label_field else _LABEL_CANDIDATES
    for f in fields:
        if f in meta and isinstance(meta[f], str) and meta[f].strip():
            return meta[f].strip()
        cm = meta.get("content_metadata")
        if isinstance(cm, dict) and isinstance(cm.get(f), str) and cm[f].strip():
            return cm[f].strip()
    return None


def _get_mp4_bytes(sample):
    v = sample.get("mp4")
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, dict) and isinstance(v.get("bytes"), (bytes, bytearray)):
        return bytes(v["bytes"])
    return None


def _sample_frames(mp4_bytes, num_frames, size):
    """Uniformly sample num_frames PIL images from the mp4 bytes via decord."""
    from decord import VideoReader

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(mp4_bytes)
        tmp_path = tmp.name
    try:
        vr = VideoReader(tmp_path)
        n = len(vr)
        if n == 0:
            return None
        import numpy as np
        from PIL import Image

        idxs = np.linspace(0, n - 1, num_frames).round().astype(int).tolist()
        arr = vr.get_batch(idxs).asnumpy()  # (num_frames, H, W, 3)
        frames = []
        for f in arr:
            img = Image.fromarray(f).convert("RGB")
            img.thumbnail((size, size))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            frames.append({"bytes": buf.getvalue()})
        return frames
    finally:
        os.unlink(tmp_path)


def _build_prompt(num_frames, label, distractors):
    options = [label] + distractors
    random.shuffle(options)
    correct_idx = options.index(label)
    placeholders = "".join("<image>\n" for _ in range(num_frames))
    opt_lines = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    question = (
        f"{placeholders}"
        "These frames are sampled from a video. Which category best describes it?\n"
        f"{opt_lines}\n"
        "Answer with the letter only."
    )
    return [{"role": "user", "content": question}], LETTERS[correct_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=os.path.expanduser("~/data/finevideo_img"))
    ap.add_argument("--repo", default="HuggingFaceFV/finevideo")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--num-frames", type=int, default=4)
    ap.add_argument("--frame-size", type=int, default=448)
    ap.add_argument("--label-field", default=None, help="e.g. content_parent_category")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect", action="store_true", help="Print first sample schema and exit.")
    args = ap.parse_args()
    random.seed(args.seed)

    from datasets import load_dataset

    ds = load_dataset(args.repo, split=args.split, streaming=True)

    if args.inspect:
        s = next(iter(ds))
        print("=== top-level keys ===")
        for k, v in s.items():
            print(f"  {k}: {type(v).__name__}")
        meta = _get_metadata(s)
        print("=== metadata keys ===")
        print(" ", list(meta.keys()) if isinstance(meta, dict) else meta)
        for f in _LABEL_CANDIDATES:
            print(f"  label candidate '{f}':", _find_label(meta, f) if meta else None)
        return

    # Buffer samples (frames + label), collect the label vocab for distractors.
    buffered = []
    for s in ds:
        if len(buffered) >= args.max_samples:
            break
        meta = _get_metadata(s)
        mp4 = _get_mp4_bytes(s)
        if meta is None or mp4 is None:
            continue
        label = _find_label(meta, args.label_field)
        if not label:
            continue
        try:
            frames = _sample_frames(mp4, args.num_frames, args.frame_size)
        except Exception as e:  # decode failures: skip, don't abort the whole build
            print(f"skip (decode failed): {e}")
            continue
        if not frames:
            continue
        buffered.append({"label": label, "frames": frames})
        if len(buffered) % 200 == 0:
            print(f"buffered {len(buffered)} / {args.max_samples}")

    vocab = sorted({b["label"] for b in buffered})
    if len(vocab) < len(LETTERS):
        raise SystemExit(f"Need >= {len(LETTERS)} distinct labels for MC, got {len(vocab)}: {vocab}")
    print(f"collected {len(buffered)} samples over {len(vocab)} categories")

    rows = []
    for i, b in enumerate(buffered):
        pool = [c for c in vocab if c != b["label"]]
        distractors = random.sample(pool, len(LETTERS) - 1)
        prompt, gt = _build_prompt(args.num_frames, b["label"], distractors)
        rows.append(
            {
                "data_source": "HuggingFaceFV/finevideo",
                "prompt": prompt,
                "images": b["frames"],
                "ability": "video_category_mc",
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {"index": i, "answer_text": b["label"]},
            }
        )

    random.shuffle(rows)
    n_test = max(1, int(len(rows) * args.test_frac))
    test_rows, train_rows = rows[:n_test], rows[n_test:]

    import pandas as pd

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(os.path.join(args.output_dir, "train.parquet"))
    pd.DataFrame(test_rows).to_parquet(os.path.join(args.output_dir, "test.parquet"))
    print(f"wrote {len(train_rows)} train / {len(test_rows)} test to {args.output_dir}")
    print("sample ground_truth values:", [r["reward_model"]["ground_truth"] for r in rows[:10]])


if __name__ == "__main__":
    main()

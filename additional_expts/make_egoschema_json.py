"""
make_egoschema_json.py -- convert the EgoSchema HF parquet (Subset split) into the exact
MVBench record format our scripts consume, so EVERY existing experiment runs on EgoSchema
unchanged via `--tasks EgoSchema` (see the DATA_LIST["EgoSchema"] entry in inference.py).

EgoSchema Subset schema (one row per question, 500 rows):
    question_idx : str            e.g. "00000"
    question     : str
    video_idx    : str (uuid)     -> the clip is <video_idx>.mp4
    option       : list[str]      each ALREADY prefixed, e.g. "A. C is cooking."
    answer       : str            a 0-BASED index into option, e.g. "3"

The MVBench record our scripts want (inference.build_prompt):
    question   : str
    candidates : list[str]        bare option text (the "A. " prefix stripped)
    answer     : str              the correct option TEXT (build_prompt does candidates.index(answer))
    video      : str              filename passed to official_frames

Emits <out_root>/json/<out_name> (default egoschema.json). The videos must be reachable at
<out_root>/video/<video> -- symlink the extracted dir once:
    ln -s <out_root>/videos/videos <out_root>/video

Run:
    python make_egoschema_json.py \
        --parquet ~/Experiments/EgoSchema/Subset/test-00000-of-00001.parquet \
        --out_root ~/Experiments/EgoSchema
"""
import argparse
import json
import os
import re

import pandas as pd

_PREFIX = re.compile(r"^\s*[A-Za-z]\.\s+")   # strips a single leading option label "A. " / "b. "


def main():
    ap = argparse.ArgumentParser(description="EgoSchema parquet -> MVBench-format egoschema.json")
    ap.add_argument("--parquet", required=True, help="EgoSchema split parquet (use the Subset split; it has real answers).")
    ap.add_argument("--out_root", required=True, help="EgoSchema root; the json lands in <out_root>/json/.")
    ap.add_argument("--out_name", default="egoschema.json")
    ap.add_argument("--video_ext", default=".mp4")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    need = {"question_idx", "question", "video_idx", "option", "answer"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"parquet is missing columns {missing}; got {list(df.columns)}")

    records, dup = [], 0
    for _, row in df.iterrows():
        candidates = [_PREFIX.sub("", str(o)).strip() for o in list(row["option"])]
        gt = int(row["answer"])
        if not 0 <= gt < len(candidates):
            raise ValueError(f"answer {row['answer']!r} out of range for q {row['question_idx']}")
        answer = candidates[gt]
        if candidates.count(answer) > 1:      # .index() in build_prompt would hit the FIRST match
            dup += 1
        records.append({
            "question_idx": str(row["question_idx"]),
            "question": str(row["question"]),
            "candidates": candidates,
            "answer": answer,
            "video": f"{row['video_idx']}{args.video_ext}",
        })

    out_dir = os.path.join(os.path.expanduser(args.out_root), "json")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)
    with open(out_path, "w") as fh:
        json.dump(records, fh, indent=2)

    print(f"wrote {len(records)} records -> {out_path}")
    if dup:
        print(f"WARNING: {dup} record(s) have a duplicated correct-option string after prefix "
              f"stripping; build_prompt's .index() will map to the first occurrence.")
    print("example record:\n" + json.dumps(records[0], indent=2)[:500])
    print("\nnext:\n"
          f"  ln -s {os.path.join(os.path.expanduser(args.out_root), 'videos', 'videos')} "
          f"{os.path.join(os.path.expanduser(args.out_root), 'video')}\n"
          "  # then run any experiment with:  --data_root <out_root> --tasks EgoSchema")


if __name__ == "__main__":
    main()

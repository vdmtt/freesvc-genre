#!/usr/bin/env python3
"""Build a source->target pair list for inference / zero-shot SVC evaluation.

Reads a split CSV (e.g. dataset_custom/test.csv) whose rows are:

    <abs_wav_path>|<lang>|<singer>

and emits an inference metadata CSV that inference_online_spk.py understands:

    src|lang_src|speaker_src|speaker_tgt|lang_tgt

Each SOURCE utterance is paired with N other singers as TARGET identities
(cross-speaker conversions), never the source's own singer. The target speaker
reference wavs are resolved by inference_online_spk.py from
  --spk-ref-base-dir/<lang_tgt>/<speaker_tgt>/*.wav
so point that flag at dataset_custom/audio.

Example:
    python make_inference_pairs.py \
        --split-csv dataset_custom/test.csv \
        --out dataset_custom/eval_pairs.csv \
        --max-src-per-singer 3 \
        --targets-per-src 2 \
        --seed 1806
"""
import argparse
import csv
import os
import random
from collections import defaultdict


def load_split(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            wav, lang, singer = parts[0], parts[1], parts[2]
            rows.append((wav, lang, singer))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", required=True, help="e.g. dataset_custom/test.csv")
    ap.add_argument("--out", required=True, help="output pair CSV")
    ap.add_argument("--max-src-per-singer", type=int, default=3,
                    help="how many source utterances to take per source singer")
    ap.add_argument("--targets-per-src", type=int, default=2,
                    help="how many distinct target singers per source utterance")
    ap.add_argument("--seed", type=int, default=1806)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_split(args.split_csv)
    if not rows:
        raise SystemExit(f"No rows parsed from {args.split_csv}")

    by_singer = defaultdict(list)
    lang_of = {}
    for wav, lang, singer in rows:
        by_singer[singer].append(wav)
        lang_of[singer] = lang

    singers = sorted(by_singer.keys())
    if len(singers) < 2:
        raise SystemExit(
            f"Need >=2 singers for cross-speaker pairs, found {len(singers)}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_pairs = 0
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        # Header row: inference_online_spk.py skips it by default
        # (--ignore-metadata-header True).
        f.write("src|lang_src|speaker_src|speaker_tgt|lang_tgt\n")
        for src_singer in singers:
            src_wavs = sorted(by_singer[src_singer])
            rng.shuffle(src_wavs)
            src_wavs = src_wavs[: args.max_src_per_singer]

            other_singers = [s for s in singers if s != src_singer]
            for wav in src_wavs:
                k = min(args.targets_per_src, len(other_singers))
                tgts = rng.sample(other_singers, k)
                for tgt in tgts:
                    f.write(
                        f"{wav}|{lang_of[src_singer]}|{src_singer}"
                        f"|{tgt}|{lang_of[tgt]}\n"
                    )
                    n_pairs += 1

    print(f"[pairs] wrote {n_pairs} source->target pairs to {args.out}")
    print(f"[pairs] {len(singers)} singers, "
          f"{sum(len(v) for v in by_singer.values())} utterances available")


if __name__ == "__main__":
    main()
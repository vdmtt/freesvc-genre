#!/usr/bin/env python3
"""Lightweight checks for FreeSVC CSVs and singer-level split leakage."""

from __future__ import annotations

import argparse
import csv
import random
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--max-audio-check", type=int, default=500)
    return p.parse_args()


def read_csv(path: Path) -> List[Tuple[Path, str, str]]:
    rows = []
    if not path.exists():
        print(f"[WARN] missing {path}")
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.reader(f, delimiter="|"), start=1):
            if len(row) != 3:
                raise ValueError(f"{path}:{line_no} must have 3 columns path|language|speaker, got {len(row)}")
            wav = Path(row[0])
            if not wav.is_absolute():
                wav = path.parent / wav
            rows.append((wav, row[1], row[2]))
    return rows


def check_audio(wav: Path, sample_rate: int) -> str | None:
    if not wav.exists():
        return f"missing: {wav}"
    try:
        with wave.open(str(wav), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            frames = wf.getnframes()
            if sr != sample_rate:
                return f"bad_sr {sr}: {wav}"
            if ch != 1:
                return f"bad_channels {ch}: {wav}"
            if frames <= 0:
                return f"empty: {wav}"
    except wave.Error as e:
        return f"not_pcm_wav: {wav} ({e})"
    return None


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    split_paths = {
        "train": dataset_dir / "train.csv",
        "valid": dataset_dir / "valid.csv",
        "test": dataset_dir / "test.csv",
    }
    rows_by_split = {k: read_csv(v) for k, v in split_paths.items()}

    print("[CSV SUMMARY]")
    singers_by_split: Dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        speakers = {spk for _, _, spk in rows}
        langs = Counter(lang for _, lang, _ in rows)
        singers_by_split[split] = speakers
        print(f"  {split:5s}: {len(rows):7d} wavs | {len(speakers):4d} singers | langs={dict(langs)}")

    leaks = []
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        overlap = singers_by_split[a] & singers_by_split[b]
        if overlap:
            leaks.append((a, b, sorted(overlap)))
    if leaks:
        for a, b, overlap in leaks:
            print(f"[LEAK] {a}/{b}: {overlap[:20]}")
        raise SystemExit(2)
    print("[OK] no singer overlap across train/valid/test")

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    rng = random.Random(1806)
    rng.shuffle(all_rows)
    errors = []
    for wav, _, _ in all_rows[: args.max_audio_check]:
        err = check_audio(wav, args.sample_rate)
        if err:
            errors.append(err)
            if len(errors) >= 20:
                break
    if errors:
        print("[AUDIO ERRORS]")
        for err in errors:
            print("  " + err)
        raise SystemExit(3)
    print(f"[OK] checked up to {min(len(all_rows), args.max_audio_check)} wav files")


if __name__ == "__main__":
    main()

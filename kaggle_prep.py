#!/usr/bin/env python3
"""Chuan bi data cho Kaggle — thay the buoc 2 (tai + sort) va 3 (split).

Tinh huong: data tren /kaggle/input DA sort san theo folder ca si, nhung con 48 kHz.
/kaggle/input la read-only nen phai resample ra /kaggle/working.

Script nay:
  1. Chon subset ca si (can bang, khong lay het 17 GB)
  2. Resample 48k -> 24k mono PCM16 vao working
  3. Sinh train.csv / valid.csv, giu ca si held-out rieng

Vi du:
    python kaggle_prep.py \
        --src  /kaggle/input/my-svc-data/audio/english \
        --dst  /kaggle/working/freesvc/dataset_custom \
        --num-singers 10 --files-per-singer 150 --held-out 2

Chay lai duoc: file da convert se bi bo qua.
"""
import argparse
import os
import shutil
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="Thu muc chua <Singer>/*.wav tren /kaggle/input")
    p.add_argument("--dst", required=True,
                   help="dataset_custom tren /kaggle/working")
    p.add_argument("--language", default="english")
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--num-singers", type=int, default=10,
                   help="Tong so ca si lay (gom ca held-out)")
    p.add_argument("--files-per-singer", type=int, default=150)
    p.add_argument("--held-out", type=int, default=2,
                   help="So ca si danh cho valid, khong xuat hien trong train")
    p.add_argument("--min-duration", type=float, default=5.0,
                   help="Bo file ngan hon nguong nay (giay)")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--genre-map", default="",
                   help="Duong dan genre_map.csv de copy sang dst (tuy chon)")
    return p.parse_args()


def wav_info(path):
    """Doc header wav bang stdlib, tra ve (sr, channels, duration) hoac None."""
    try:
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            return sr, w.getnchannels(), w.getnframes() / sr if sr else 0.0
    except Exception:
        return None


def main():
    args = parse_args()
    src = Path(args.src)
    dst_root = Path(args.dst)
    audio_dst = dst_root / "audio" / args.language

    if not src.is_dir():
        sys.exit(f"[ERROR] Khong thay {src}")
    if shutil.which("ffmpeg") is None:
        sys.exit("[ERROR] Khong co ffmpeg trong PATH")

    # ---- 1. Chon ca si -----------------------------------------------------
    singers = {}
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        wavs = sorted(d.glob("*.wav"))
        if wavs:
            singers[d.name] = wavs

    if len(singers) < args.num_singers:
        print(f"[WARN] Chi co {len(singers)} ca si, yeu cau {args.num_singers}")
        args.num_singers = len(singers)
    if args.num_singers <= args.held_out:
        sys.exit("[ERROR] num-singers phai lon hon held-out")

    # Uu tien ca si co nhieu file nhat -> du de cat lay files-per-singer,
    # dong thoi tranh ca si qua it file lam mat can bang.
    chosen = sorted(singers, key=lambda s: -len(singers[s]))[: args.num_singers]
    chosen.sort()   # sort lai theo ten cho on dinh giua cac lan chay

    print(f"[INFO] {len(singers)} ca si co san, chon {len(chosen)}:")
    for s in chosen:
        print(f"         {s:35s} {len(singers[s]):5d} file")

    # ---- 2. Loc + resample -------------------------------------------------
    jobs, skipped_short, skipped_bad = [], 0, 0
    for s in chosen:
        taken = 0
        for w in singers[s]:
            if taken >= args.files_per_singer:
                break
            info = wav_info(w)
            if info is None:
                skipped_bad += 1
                continue
            _, _, dur = info
            if dur < args.min_duration:
                skipped_short += 1
                continue
            jobs.append((w, audio_dst / s / w.name))
            taken += 1

    print(f"\n[INFO] {len(jobs)} file se dung "
          f"(bo {skipped_short} file < {args.min_duration}s, {skipped_bad} file loi)")

    todo = [(a, b) for a, b in jobs if not b.exists()]
    print(f"[INFO] {len(jobs) - len(todo)} file da convert tu truoc, "
          f"can convert {len(todo)}")

    def convert(pair):
        s_, d_ = pair
        d_.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(s_),
             "-ac", "1", "-ar", str(args.sample_rate), "-c:a", "pcm_s16le", str(d_)],
            capture_output=True)
        return d_ if r.returncode == 0 else None

    if todo:
        ok = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for i, res in enumerate(ex.map(convert, todo), 1):
                ok += res is not None
                if i % 100 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)
        print(f"[INFO] convert xong {ok}/{len(todo)}")

    # ---- 3. Kiem tra ket qua ----------------------------------------------
    bad = []
    for _, d_ in jobs[: min(50, len(jobs))]:
        info = wav_info(d_)
        if info is None or info[0] != args.sample_rate or info[1] != 1:
            bad.append((d_, info))
    if bad:
        print(f"\n[ERROR] {len(bad)} file khong dung chuan sau convert, vi du:")
        for d_, info in bad[:3]:
            print(f"    {d_}: {info}")
        sys.exit(1)
    print(f"[OK] mau kiem tra: {args.sample_rate} Hz mono PCM16")

    # ---- 4. Sinh CSV -------------------------------------------------------
    held_out = chosen[-args.held_out:]
    train_spk = chosen[: -args.held_out]

    def lines_for(spks):
        out = []
        for s in spks:
            for w in sorted((audio_dst / s).glob("*.wav")):
                out.append(f"{w}|{args.language}|{s}")
        return out

    tr, va = lines_for(train_spk), lines_for(held_out)
    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "train.csv").write_text("\n".join(tr) + "\n")
    (dst_root / "valid.csv").write_text("\n".join(va) + "\n")

    if args.genre_map and Path(args.genre_map).is_file():
        shutil.copy(args.genre_map, dst_root / "genre_map.csv")
        print(f"[INFO] da copy genre_map.csv")

    total_gb = sum(f.stat().st_size for f in audio_dst.rglob("*.wav")) / 1e9

    print(f"\n{'=' * 62}")
    print(f"train.csv : {len(tr):6d} dong | {len(train_spk)} ca si")
    print(f"valid.csv : {len(va):6d} dong | {len(held_out)} ca si held-out: {held_out}")
    print(f"dung luong: {total_gb:.2f} GB trong {audio_dst}")
    print(f"{'=' * 62}")
    print("\nBuoc tiep theo — cache pitch (lam 1 lan):")
    print(f"  python scripts/preprocess_pitch.py \\")
    print(f"      --in-dir {dst_root}/audio \\")
    print(f"      --out-dir {dst_root}/pitch_features \\")
    print(f"      --pitch-predictor rmvpe --skip-existing")

if __name__ == "__main__":
    main()
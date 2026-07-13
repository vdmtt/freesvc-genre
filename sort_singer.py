#!/usr/bin/env python3
"""Sort raw singer audio into FreeSVC layout: <output>/<language>/<singer>/<file>.wav

Singer name can be derived from:
  - parent   : the immediate parent folder of each audio file  (chunk_01/Son Tung/clip.wav -> "Son Tung")
  - filename : the part of the filename before the trailing _<digits> segment id
               (Son Tung_00001.wav -> "Son Tung", My Singer_chunk_012.wav -> "My Singer")
  - auto     : try filename first; if it can't be parsed, fall back to parent folder

Default is 'parent' because chunk datasets are almost always organised as
.../<SingerName>/<clip>.wav. Output filenames are made unique per singer with a
short deterministic hash of the source path, so re-running is safe (resumable)
and clips coming from different chunks never overwrite each other.
"""
import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}

# trailing "_<digits>" or "_chunk_<digits>" / "-segment_<digits>" etc.
_SEG_SUFFIX = re.compile(
    r"[\s\-_]*(?:chunk|seg|segment|part|clip)?[\s\-_]*\d+$",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)  # dataset_custom/audio
    parser.add_argument("--language", default="english")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--copy-only", action="store_true")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--singer-source",
        choices=["parent", "filename", "auto"],
        default="parent",
        help="where to read the singer name from (default: parent folder)",
    )
    return parser.parse_args()


def normalize_singer_name(name: str) -> str:
    name = name.strip()
    # collapse spaces/hyphens/underscores
    name = re.sub(r"[\s\-_]+", "_", name)
    # keep letters, numbers and underscore only (drop accents/punctuation)
    name = re.sub(r"[^A-Za-z0-9_]", "", name)
    name = name.strip("_")
    if not name:
        raise ValueError("Empty singer name after normalization")
    return name


def singer_from_filename(path: Path) -> str:
    stem = path.stem
    raw = _SEG_SUFFIX.sub("", stem)
    if not raw or not raw.strip("_- "):
        raise ValueError(f"Cannot parse singer from filename: {path.name}")
    return normalize_singer_name(raw)


def singer_from_parent(path: Path, input_dir: Path) -> str:
    parent = path.parent
    # never use the input root itself or generic chunk/extract folders as singer
    bad = {input_dir.name.lower(), "audio", "wav", "wavs", "data"}
    name = parent.name
    if (
        parent == input_dir
        or name.lower() in bad
        or re.fullmatch(r"(extracted_)?[\s\-_]*chunk[\s\-_]*\d*", name, re.IGNORECASE)
    ):
        raise ValueError(
            f"Parent folder '{name}' does not look like a singer name for {path}"
        )
    return normalize_singer_name(name)


def resolve_singer(path: Path, input_dir: Path, mode: str) -> str:
    if mode == "filename":
        return singer_from_filename(path)
    if mode == "parent":
        return singer_from_parent(path, input_dir)
    # auto: filename first, then parent
    try:
        return singer_from_filename(path)
    except ValueError:
        return singer_from_parent(path, input_dir)


def unique_out_name(src: Path, singer: str) -> str:
    h = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:6]
    stem = re.sub(r"[^A-Za-z0-9]+", "", src.stem) or "clip"
    return f"{singer}_{stem}_{h}.wav"


def convert_audio(src: Path, dst: Path, sample_rate: int, copy_only: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_only and src.suffix.lower() == ".wav":
        shutil.copy2(src, dst)
        return
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    lang_dir = output_dir / args.language

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    audio_files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        print(f"[WARN] No audio files found under: {input_dir}")
        return

    manifest_rows = []
    singer_counts = {}
    processed = skipped = failed = 0

    for src in audio_files:
        try:
            singer = resolve_singer(src, input_dir, args.singer_source)
            out_name = unique_out_name(src, singer)
            dst = lang_dir / singer / out_name

            if dst.exists() and not args.overwrite:
                skipped += 1
                singer_counts[singer] = singer_counts.get(singer, 0) + 1
                continue

            convert_audio(src, dst, args.sample_rate, args.copy_only)

            singer_counts[singer] = singer_counts.get(singer, 0) + 1
            processed += 1
            manifest_rows.append([str(src), str(dst), args.language, singer])
        except Exception as e:
            failed += 1
            print(f"[FAIL] {src}: {e}")

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not manifest_path.exists()
        with manifest_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["source_path", "output_path", "language", "singer"])
            writer.writerows(manifest_rows)

    print("\n[SORT SUMMARY]")
    print(f"  input dir   : {input_dir}")
    print(f"  output dir  : {lang_dir}")
    print(f"  singer src  : {args.singer_source}")
    print(f"  sample rate : {args.sample_rate}")
    print(f"  processed   : {processed}")
    print(f"  skipped     : {skipped}")
    print(f"  failed      : {failed}")
    print("\n[SINGER COUNTS]")
    for singer, count in sorted(singer_counts.items()):
        print(f"  {singer}: {count}")
    if failed and processed == 0:
        raise SystemExit(
            "All files failed to sort. Check --singer-source "
            "(parent/filename/auto) and your folder/file naming."
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Thong ke so BAI (songid) va so SEGMENT theo genre.

Con so quan trong nhat la SO BAI, khong phai so segment:
cac segment cung 1 songid den tu cung 1 bai/ca si/ban phoi -> tuong quan rat manh,
gan nhu khong phai mau doc lap. Genre chi co vai bai thi genre embedding se hoc
thanh "dac trung cua may bai do" (overfit) chu khong phai "dac trung the loai".

Dung:
  # thong ke tren toan bo genre_map (chay duoc o local, khong can train.csv)
  python genre_stats.py --genre-map dataset_custom/genre_map.csv

  # chi tinh tren cac segment THUC SU dung trong train (sau khi co train.csv)
  python genre_stats.py --genre-map dataset_custom/genre_map.csv \
      --csv dataset_custom/train.csv
"""
import argparse
import os
import re
import sys
from collections import defaultdict


SEG_RE = re.compile(r"(\d{8})_[0-9a-fA-F]{4,}$")


def parse_segment_id(path):
    """Lay segment id (xxxxyyyy) tu ten file THUC TE.

    Format that: {Ca_Si}_{CaSiVietLien}{songid}{segid}_{hash}.wav
    VD: Sons_Of_The_East_SonsOfTheEast13250001_44d218.wav -> '13250001'
    KHONG dung rsplit('_',1)[-1] -> se ra hash '44d218'!
    """
    stem = os.path.basename(path)
    if stem.endswith(".wav"):
        stem = stem[:-4]
    m = SEG_RE.search(stem)
    if m:
        return m.group(1)
    parts = stem.split("_")
    if len(parts) >= 2:
        m = re.search(r"(\d{8})$", parts[-2])
        if m:
            return m.group(1)
    hits = re.findall(r"\d{8}", stem)
    return hits[-1] if hits else None


def parse_genre_map(path, sep=","):
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or sep not in line:
                continue
            seg, genre = line.split(sep, 1)
            seg, genre = seg.strip(), genre.strip()
            if seg:
                mapping[seg] = genre
    return mapping


def segments_from_csv(csv_paths, sep="|"):
    """Lay {segment_id: singer} tu metadata csv."""
    segs = {}
    unparsed = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"[WARN] khong thay {csv_path}, bo qua")
            continue
        with open(csv_path, encoding="utf-8") as f:
            for row in f:
                row = row.strip()
                if not row:
                    continue
                cols = row.split(sep)
                path = cols[0]
                seg = parse_segment_id(path)
                if seg is None:
                    unparsed.append(path)
                    continue
                singer = cols[2] if len(cols) > 2 else os.path.basename(os.path.dirname(path))
                segs[seg] = singer
    if unparsed:
        print(f"[WARN] {len(unparsed)} file khong parse duoc segment id, vd: {unparsed[:2]}")
    return segs


def cross_tab(genre_map, split_files, csv_sep, seg_len):
    """Bang genre x split: phat hien genre vang mat khoi mot split."""
    splits = {}
    for label, path in split_files:
        if not os.path.exists(path):
            print(f"[WARN] khong thay {path}, bo qua")
            continue
        segs = segments_from_csv([path], csv_sep)
        splits[label] = segs

    if not splits:
        return

    genres = sorted(set(genre_map.values()))
    labels = list(splits.keys())

    print()
    print("=" * 78)
    print("CROSS-TAB: SO BAI moi genre theo split  (0 = NGUY HIEM)")
    print("-" * 78)
    hdr = f"{'GENRE':<14}"
    for lb in labels:
        hdr += f"{lb.upper():>12}"
    print(hdr)
    print("-" * 78)

    problems = []
    for g in genres:
        line = f"{g:<14}"
        counts = {}
        for lb in labels:
            songs = set()
            for seg in splits[lb]:
                if genre_map.get(seg) == g:
                    songs.add(seg[:-seg_len] if len(seg) > seg_len else seg)
            counts[lb] = len(songs)
            line += f"{len(songs):>12}"
        print(line)
        if counts.get("train", 0) == 0 and any(counts.get(lb, 0) > 0 for lb in labels if lb != "train"):
            problems.append((g, "KHONG co trong TRAIN nhung CO o split khac"))
        elif counts.get("train", 0) > 0 and counts.get("train", 0) < 3:
            problems.append((g, f"chi {counts['train']} bai trong TRAIN"))
    print("=" * 78)

    print("\nDANH GIA CROSS-TAB:")
    if not problems:
        print("  [OK] moi genre deu co mat trong train. Ablation genre se co nghia.")
    for g, msg in problems:
        print(f"  [FAIL] '{g}': {msg}")
        print(f"         -> genre_emb['{g}'] KHONG duoc train = random vector.")
        print(f"         -> luc eval no cong NHIEU vao x, keo tut ket qua run ablation")
        print(f"            vi ly do SAI (khong phai vi nhan genre vo dung).")
        print(f"         -> FIX: doi seed split, hoac ep ca si cua '{g}' vao train")
        print(f"            (TRAIN_SINGERS_FILE trong config.sh), hoac bo genre nay.")


def bar(frac, width=28):
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genre-map", required=True)
    ap.add_argument("--map-sep", default=",")
    ap.add_argument("--csv", nargs="*", default=[],
                    help="metadata csv. Co -> chi thong ke segment thuc su dung trong do")
    ap.add_argument("--csv-sep", default="|")
    ap.add_argument("--seg-len", type=int, default=4,
                    help="so ky tu cua segment id o CUOI (default 4). songid = phan con lai")
    ap.add_argument("--min-songs", type=int, default=10,
                    help="canh bao genre co it hon so bai nay")
    ap.add_argument("--splits", nargs="*", default=[], metavar="LABEL=PATH",
                    help="cross-tab genre x split, vd: "
                         "--splits train=dataset_custom/train.csv valid=dataset_custom/valid.csv "
                         "test=dataset_custom/test.csv")
    args = ap.parse_args()

    mapping = parse_genre_map(args.genre_map, args.map_sep)
    if not mapping:
        print("[FAIL] genre_map rong hoac sai separator")
        sys.exit(1)

    singers_of = {}
    if args.csv:
        used = segments_from_csv(args.csv, args.csv_sep)
        missing = [s for s in used if s not in mapping]
        before = len(mapping)
        mapping = {s: g for s, g in mapping.items() if s in used}
        singers_of = used
        print(f"Loc theo csv: {before} -> {len(mapping)} segment"
              f" ({len(missing)} segment trong csv KHONG co genre)")
        if missing:
            print(f"  [WARN] vd thieu: {missing[:5]}")

    # gom theo genre
    seg_count = defaultdict(int)
    songs = defaultdict(set)
    singers = defaultdict(set)
    for seg, genre in mapping.items():
        seg_count[genre] += 1
        songid = seg[:-args.seg_len] if len(seg) > args.seg_len else seg
        songs[genre].add(songid)
        if seg in singers_of:
            singers[genre].add(singers_of[seg])

    total_seg = sum(seg_count.values())
    total_song = len(set().union(*songs.values())) if songs else 0

    has_singer = bool(singers_of)
    print()
    print("=" * 78)
    hdr = f"{'GENRE':<14}{'SEGMENT':>9}{'%':>7}  {'BAI':>5}{'%':>7}  {'seg/bai':>8}"
    if has_singer:
        hdr += f"{'CA SI':>7}"
    print(hdr)
    print("-" * 78)

    for genre in sorted(seg_count, key=lambda g: -len(songs[g])):
        ns, nsong = seg_count[genre], len(songs[genre])
        line = (f"{genre:<14}{ns:>9}{ns/total_seg*100:>6.1f}%"
                f"  {nsong:>5}{nsong/total_song*100:>6.1f}%  {ns/nsong:>8.1f}")
        if has_singer:
            line += f"{len(singers[genre]):>7}"
        print(line)
    print("-" * 78)
    print(f"{'TONG':<14}{total_seg:>9}{'':>7}  {total_song:>5}")
    print("=" * 78)

    # bieu do theo SO BAI
    print("\nPhan bo theo SO BAI (con so quan trong):")
    mx = max(len(v) for v in songs.values())
    for genre in sorted(songs, key=lambda g: -len(songs[g])):
        n = len(songs[genre])
        print(f"  {genre:<14} {bar(n/mx)} {n}")

    # canh bao
    print("\nDANH GIA:")
    weak = [g for g in songs if len(songs[g]) < args.min_songs]
    if weak:
        for g in weak:
            print(f"  [WARN] '{g}' chi co {len(songs[g])} bai -> genre_emb['{g}'] se hoc thanh")
            print(f"         'dac trung cua may bai nay' chu khong phai the loai (overfit).")
            print(f"         -> can nhac gop vao genre khac, hoac bo va giam num_genres.")
    ratio = max(len(v) for v in songs.values()) / min(len(v) for v in songs.values())
    print(f"  Ti le mat can bang (theo bai): {ratio:.1f}x")
    if ratio > 20:
        print("  -> Lech RAT nang. weighted sampling se lap genre hiem rat nhieu lan/epoch")
        print("     => doi tu underfit sang OVERFIT, khong phai fix that su.")
    elif ratio > 5:
        print("  -> Lech dang ke. Co the thu weighted sampling (nen dung sqrt inverse freq).")
    else:
        print("  -> Tuong doi can bang. Chay baseline khong weighted truoc.")
    if not has_singer and not args.splits:
        print("\n  [INFO] Chay lai voi --csv dataset_custom/train.csv de biet so CA SI moi genre")
        print("         (nhieu bai nhung cung 1 ca si van la tin hieu overfit).")

    if args.splits:
        full_map = parse_genre_map(args.genre_map, args.map_sep)
        split_files = []
        for item in args.splits:
            if "=" not in item:
                print(f"[WARN] --splits sai dinh dang: {item!r}, can LABEL=PATH")
                continue
            lb, path = item.split("=", 1)
            split_files.append((lb, path))
        cross_tab(full_map, split_files, args.csv_sep, args.seg_len)


if __name__ == "__main__":
    main()
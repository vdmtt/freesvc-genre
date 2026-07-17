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
import sys
from collections import defaultdict


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
                stem = os.path.basename(path).replace(".wav", "")
                seg = stem.rsplit("_", 1)[-1]        # xxxxyyyy
                singer = cols[2] if len(cols) > 2 else stem.rsplit("_", 1)[0]
                segs[seg] = singer
    return segs


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
    if not has_singer:
        print("\n  [INFO] Chay lai voi --csv dataset_custom/train.csv de biet so CA SI moi genre")
        print("         (nhieu bai nhung cung 1 ca si van la tin hieu overfit).")


if __name__ == "__main__":
    main()
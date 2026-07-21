#!/usr/bin/env python3
"""Kiem tra genre_map.csv truoc khi train FreeSVC + genre embedding.

Check 3 chieu:
  1) COVERAGE : moi segment trong train/valid/test.csv deu co trong genre_map
  2) GENRE SET: moi genre trong genre_map deu co trong genre2id cua config
  3) ID SANITY: genre2id la int, lien tuc tu 0, va num_genres > max(id)

Dung:
  python verify_genre_map.py \
      --genre-map dataset_custom/genre_map.csv \
      --config configs/config_genre.yaml \
      --csv dataset_custom/train.csv dataset_custom/valid.csv
"""
import argparse
import os
import re
import sys

import yaml


SEG_RE = re.compile(r"(\d{8})_[0-9a-fA-F]{4,}$")


def parse_segment_id(path):
    """Lay segment id (xxxxyyyy) tu ten file DA SORT (co hash md5 o cuoi).

    sort_singer.py tao ten: {Ca_Si}_{TenGocBoKyTuDacBiet}_{hash6}.wav
    VD: Sons_Of_The_East_SonsOfTheEast13250001_44d218.wav -> '13250001'
    KHONG dung rsplit('_',1)[-1] -> ra hash '44d218'!
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
    """Doc file map -> {segment_id: genre}. Bao loi trung/leading-zero."""
    mapping = {}
    dup = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if sep not in line:
                print(f"  [WARN] dong {lineno} khong co separator '{sep}': {line!r}")
                continue
            seg, genre = line.split(sep, 1)
            seg, genre = seg.strip(), genre.strip()
            if seg in mapping and mapping[seg] != genre:
                dup.append(seg)
            mapping[seg] = genre
    if dup:
        print(f"  [WARN] {len(dup)} segment bi trung voi genre khac nhau: {dup[:5]}")
    return mapping


def segments_from_csv(csv_path, sep="|", skip_header=False):
    """Lay segment id tu cot dau (path wav) cua metadata csv."""
    segs = {}
    with open(csv_path, encoding="utf-8") as f:
        rows = f.readlines()
    if skip_header:
        rows = rows[1:]
    for row in rows:
        row = row.strip()
        if not row:
            continue
        path = row.split(sep)[0]
        seg = parse_segment_id(path)
        if seg is None:
            print(f"  [WARN] khong parse duoc segment id: {os.path.basename(path)}")
            continue
        segs[seg] = path
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genre-map", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--csv", nargs="*", default=[],
                help="metadata csv (train/valid/test). Bo trong -> bo qua check coverage "
                     "(dung khi chua chay 3_split.sh)")
    ap.add_argument("--map-sep", default=",", help="separator trong genre_map (default ',')")
    ap.add_argument("--csv-sep", default="|", help="separator trong metadata csv (default '|')")
    ap.add_argument("--skip-header", action="store_true")
    args = ap.parse_args()

    ok = True

    # ---- doc config ----
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = cfg.get("data", {}) or {}
    genre2id = data.get("genre2id")
    num_genres = data.get("num_genres")
    genre_dim = data.get("genre_dim")

    print("=" * 60)
    print("[1/4] CONFIG")
    if genre2id is None:
        print("  [FAIL] config.data.genre2id KHONG TON TAI "
              "(sai ten key? phai la 'genre2id', khong phai 'genre_id')")
        sys.exit(1)
    print(f"  genre2id   : {genre2id}")
    print(f"  num_genres : {num_genres}")
    print(f"  genre_dim  : {genre_dim}  (phai = hidden_channels, thuong 192)")

    # ---- 3) id sanity ----
    print("[2/4] ID SANITY")
    bad_type = {k: v for k, v in genre2id.items() if not isinstance(v, int)}
    if bad_type:
        print(f"  [FAIL] id khong phai int: {bad_type}  (vd 'rock: 1s' -> phai la 'rock: 1')")
        ok = False
    else:
        ids = sorted(genre2id.values())
        if ids != list(range(len(ids))):
            print(f"  [WARN] id khong lien tuc tu 0: {ids}")
        if num_genres is None or max(ids) >= num_genres:
            print(f"  [FAIL] num_genres={num_genres} phai > max(id)={max(ids)} "
                  f"(nn.Embedding se index out of range)")
            ok = False
        else:
            print(f"  [OK] id int, lien tuc, max={max(ids)} < num_genres={num_genres}")

    # ---- doc genre map ----
    print("[3/4] GENRE MAP")
    if not os.path.exists(args.genre_map):
        print(f"  [FAIL] khong thay file: {args.genre_map}")
        sys.exit(1)
    mapping = parse_genre_map(args.genre_map, args.map_sep)
    print(f"  segment trong map : {len(mapping)}")
    genres_in_map = sorted(set(mapping.values()))
    print(f"  genre trong map   : {genres_in_map}")

    # chieu QUAN TRONG: genre trong map phai co trong genre2id
    unknown = sorted({g for g in mapping.values() if g not in genre2id})
    if unknown:
        print(f"  [FAIL] genre co trong map nhung THIEU trong genre2id: {unknown}")
        print(f"         -> them vao config, hoac sua ten cho khop")
        ok = False
    else:
        print("  [OK] moi genre trong map deu co trong genre2id")

    # chieu nay chi la thong tin, KHONG phai loi
    unused = sorted({g for g in genre2id if g not in set(mapping.values())})
    if unused:
        print(f"  [INFO] genre khai trong config nhung dataset chua co: {unused}")
        print("         -> khong sao, embedding do chi khong duoc train")

    # ---- 1) coverage ----
    print("[4/4] COVERAGE (quan trong nhat)")
    if not args.csv:
        print("  [SKIP] chua truyen --csv -> BO QUA check coverage.")
        print("         Cac check tren van co gia tri (config + genre set).")
        print("         SAU khi chay 2_ va 3_ (co train.csv/valid.csv), chay lai voi:")
        print("           --csv dataset_custom/train.csv dataset_custom/valid.csv")
        print("=" * 60)
        print("KET QUA:", "PASS (phan config) - CHUA check coverage"
              if ok else "FAIL - sua cac loi tren truoc")
        sys.exit(0 if ok else 1)
    for csv_path in args.csv:
        if not os.path.exists(csv_path):
            print(f"  [WARN] khong thay {csv_path}, bo qua")
            continue
        segs = segments_from_csv(csv_path, args.csv_sep, args.skip_header)
        missing = {s: p for s, p in segs.items() if s not in mapping}
        tag = "OK" if not missing else "FAIL"
        print(f"  [{tag}] {csv_path}: {len(segs)} segment, thieu genre: {len(missing)}")
        if missing:
            ok = False
            for s, p in list(missing.items())[:5]:
                print(f"        - seg {s!r} tu {os.path.basename(p)}")
            if len(missing) > 5:
                print(f"        ... va {len(missing)-5} segment nua")
            print("        -> nhung sample nay se bi collate BO QUA AM THAM (KeyError)")

    print("=" * 60)
    print("KET QUA:", "PASS - san sang train" if ok else "FAIL - sua cac loi tren truoc")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--language", default="english")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1806)

    parser.add_argument("--min-files-per-singer", type=int, default=1)

    parser.add_argument("--train-singers-file", default="")
    parser.add_argument("--valid-singers-file", default="")
    parser.add_argument("--test-singers-file", default="")

    return parser.parse_args()


def read_singer_file(path):
    if not path:
        return None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Singer file not found: {p}")

    singers = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            singers.append(line)

    return singers


def list_singers(audio_root, language, min_files):
    lang_dir = Path(audio_root) / language

    if not lang_dir.exists():
        raise FileNotFoundError(f"Language audio dir not found: {lang_dir}")

    singer_to_wavs = {}

    for singer_dir in sorted(lang_dir.iterdir()):
        if not singer_dir.is_dir():
            continue

        wavs = sorted(singer_dir.glob("*.wav"))

        if len(wavs) < min_files:
            print(
                f"[SKIP] {singer_dir.name}: only {len(wavs)} wavs "
                f"< min_files_per_singer={min_files}"
            )
            continue

        singer_to_wavs[singer_dir.name] = wavs

    if not singer_to_wavs:
        raise RuntimeError(f"No valid singers found under: {lang_dir}")

    return singer_to_wavs


def auto_split_singers(singers, valid_ratio, test_ratio, seed):
    singers = list(singers)

    if len(singers) < 3:
        raise RuntimeError(
            f"Need at least 3 singers for disjoint train/valid/test split, "
            f"but found {len(singers)}"
        )

    rng = random.Random(seed)
    rng.shuffle(singers)

    n_total = len(singers)
    n_test = max(1, round(n_total * test_ratio))
    n_valid = max(1, round(n_total * valid_ratio))

    if n_test + n_valid >= n_total:
        n_test = 1
        n_valid = 1

    test_singers = singers[:n_test]
    valid_singers = singers[n_test:n_test + n_valid]
    train_singers = singers[n_test + n_valid:]

    if not train_singers:
        raise RuntimeError("Train split is empty. Reduce valid/test ratios.")

    return train_singers, valid_singers, test_singers


def manual_split_singers(all_singers, train_file, valid_file, test_file):
    train_singers = read_singer_file(train_file)
    valid_singers = read_singer_file(valid_file)
    test_singers = read_singer_file(test_file)

    if train_singers is None or valid_singers is None or test_singers is None:
        raise RuntimeError(
            "Manual split requires all three files: "
            "--train-singers-file, --valid-singers-file, --test-singers-file"
        )

    all_singers = set(all_singers)

    for split_name, split_singers in [
        ("train", train_singers),
        ("valid", valid_singers),
        ("test", test_singers),
    ]:
        missing = sorted(set(split_singers) - all_singers)
        if missing:
            raise RuntimeError(
                f"{split_name} singer file contains unknown singers: {missing}"
            )

    return train_singers, valid_singers, test_singers


def assert_disjoint(train_singers, valid_singers, test_singers):
    train_set = set(train_singers)
    valid_set = set(valid_singers)
    test_set = set(test_singers)

    overlap_train_valid = train_set & valid_set
    overlap_train_test = train_set & test_set
    overlap_valid_test = valid_set & test_set

    if overlap_train_valid or overlap_train_test or overlap_valid_test:
        raise RuntimeError(
            "Singer leakage detected:\n"
            f"  train ∩ valid: {sorted(overlap_train_valid)}\n"
            f"  train ∩ test : {sorted(overlap_train_test)}\n"
            f"  valid ∩ test : {sorted(overlap_valid_test)}"
        )


def write_csv(path, singers, singer_to_wavs, language):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0

    with path.open("w", encoding="utf-8", newline="") as f:
        for singer in singers:
            wavs = singer_to_wavs[singer]

            for wav in wavs:
                f.write(f"{wav.resolve()}|{language}|{singer}\n")
                n_rows += 1

    return n_rows


def write_singer_list(path, singers):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(singers) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    split_dir = out_dir / "singer_splits"

    singer_to_wavs = list_singers(
        audio_root=args.audio_root,
        language=args.language,
        min_files=args.min_files_per_singer,
    )

    all_singers = sorted(singer_to_wavs.keys())

    manual_mode = bool(
        args.train_singers_file
        or args.valid_singers_file
        or args.test_singers_file
    )

    if manual_mode:
        train_singers, valid_singers, test_singers = manual_split_singers(
            all_singers=all_singers,
            train_file=args.train_singers_file,
            valid_file=args.valid_singers_file,
            test_file=args.test_singers_file,
        )
    else:
        train_singers, valid_singers, test_singers = auto_split_singers(
            singers=all_singers,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    assert_disjoint(train_singers, valid_singers, test_singers)

    n_train = write_csv(
        out_dir / "train.csv",
        train_singers,
        singer_to_wavs,
        args.language,
    )
    n_valid = write_csv(
        out_dir / "valid.csv",
        valid_singers,
        singer_to_wavs,
        args.language,
    )
    n_test = write_csv(
        out_dir / "test.csv",
        test_singers,
        singer_to_wavs,
        args.language,
    )

    all_rows = 0
    with (out_dir / "all.csv").open("w", encoding="utf-8", newline="") as f:
        for singer in all_singers:
            for wav in singer_to_wavs[singer]:
                f.write(f"{wav.resolve()}|{args.language}|{singer}\n")
                all_rows += 1

    write_singer_list(split_dir / "train_singers.txt", train_singers)
    write_singer_list(split_dir / "valid_singers.txt", valid_singers)
    write_singer_list(split_dir / "test_singers.txt", test_singers)

    print("[SPLIT SUMMARY]")
    print(f"  audio root : {Path(args.audio_root).resolve() / args.language}")
    print(f"  language   : {args.language}")
    print(f"  total singers: {len(all_singers)}")
    print(f"  total wavs   : {all_rows}")
    print(f"  train: {len(train_singers):4d} singers | {n_train:7d} wavs")
    print(f"  valid: {len(valid_singers):4d} singers | {n_valid:7d} wavs")
    print(f"  test : {len(test_singers):4d} singers | {n_test:7d} wavs")
    print()

    print("[OUTPUT]")
    print(f"  {out_dir / 'train.csv'}")
    print(f"  {out_dir / 'valid.csv'}")
    print(f"  {out_dir / 'test.csv'}")
    print(f"  {out_dir / 'all.csv'}")
    print(f"  {split_dir / 'train_singers.txt'}")
    print(f"  {split_dir / 'valid_singers.txt'}")
    print(f"  {split_dir / 'test_singers.txt'}")


if __name__ == "__main__":
    main()
## Training
```bash
bash run_prepare.sh
bash 4_extract_pitch.sh
bash 5_train.sh
```
## Inference
```bash
python make_inference_pairs.py \
  --split-csv dataset_custom/test.csv \
  --out dataset_custom/eval_pairs.csv \
  --max-src-per-singer #3 \
  --targets-per-src #2 \
  --seed 1806
```
```bash
python scripts/inference_online_spk.py \
 --hpfile dataset_custom/runs/officialstyle_freevcGD/.hydra/config.yaml\
 --ptfile dataset_custom/checkpoints/G_XXXXX_XXXXXXX.pth \
 --input-base-dir / \
 --metadata-path path/to/eval_pairs.csv \
 --spk-mode online \
 --spk-ref-base-dir dataset_custom/audio \
 --num-ref-wavs 10 \
 --pitch-predictor rmvpe \
 --out-dir gen-samples/
```


### Audio Conversion

This section explains how to use the FreeSVC model for audio conversion.

```bash
python scripts/inference.py --hpfile path/to/config.yaml \
                   --ptfile path/to/checkpoint.pth \
                   --input-base-dir path/to/input/directory \
                   --metadata-path path/to/metadata.csv \
                   --spk-emb-base-dir path/to/speaker/embeddings \
                   --out-dir path/to/output_directory \
                   [--use-vad] \
                   [--use-timestamp] \
                   [--concat-audio] \
                   [--pitch-factor PITCH_FACTOR]
```

**Parameters:**
- --hpfile: Path to the configuration YAML file
- --ptfile: Path to the model checkpoint file
- --input-base-dir: Base directory containing source audio files
- --metadata-path: Path to the CSV metadata file
- --spk-emb-base-dir: Directory containing speaker embeddings
- --out-dir: Output directory for converted audio (default: "gen-samples/")

**Optional Parameters:**
- --pitch-predictor: Pitch predictor model type (default: "rmvpe")
- --use-vad: Enable Voice Activity Detection for better segmentation
- --use-timestamp: Add timestamps to output filenames
- --concat-audio: Concatenate all converted segments into a single file
- --pitch-factor: Adjust pitch modification factor (default: 0.9544)
- --ignore-metadata-header: Skip first row of metadata CSV (default: True)

**Metadata CSV Format:**
```
source_path|source_lang|source_speaker|target_speaker|target_lang
./audio/source1.wav|en|speaker1|speaker2|ja
./audio/source2.wav|zh|speaker3|speaker4|en
```

Required columns:

- source_path: Path to source audio file (relative to input-base-dir)
- source_lang: Source language code (e.g., 'en', 'ja', 'zh')
- source_speaker: Source speaker identifier
- target_speaker: Target speaker identifier
- target_lang: Target language code
- transcript: (Optional) Text transcript of the audio

**Output Directory Structure**
The converted audio files will be organized in the following structure:
```
output_dir/
└── metadata_name/
    └── source_lang/
        └── target_lang/
            └── source_speaker/
                └── target_speaker/
                    └── converted_audio.wav
```


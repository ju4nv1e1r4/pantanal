"""
How to use:
    python -m src.data.explore_audio
    python -m src.data.explore_audio --output data/audio_stats.csv --format csv
    python -m src.data.explore_audio --output data/audio_stats.json --format json
    python -m src.data.explore_audio --species blchaw1 67252 555123
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

TRAIN_CSV = "data/train.csv"
TAXONOMY = "data/taxonomy.csv"
AUDIO_DIR = "data/train_audio"


def extract_audio_stats(file_path: str, sr: int = 32000) -> dict:
    """
    Loads an audio file and extracts statistics relevant to
    understanding signal quality and frequency content.
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)

    duration = len(y) / sr

    # Waveform
    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    silence_ratio = float(np.mean(np.abs(y) < 0.01))  # fraction below noise floor

    # Mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=224,
        fmin=50.0,
        fmax=16000.0,
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    spec_min = float(S_dB.min())
    spec_max = float(S_dB.max())
    spec_mean = float(S_dB.mean())
    spec_std = float(S_dB.std())

    # Dominant frequency band
    # Sum energy across time for each mel band, find the peak band
    mel_freqs = librosa.mel_frequencies(n_mels=224, fmin=50.0, fmax=16000.0)
    band_energy = S.mean(axis=1)  # [n_mels]
    dominant_band = int(np.argmax(band_energy))
    dominant_freq = float(mel_freqs[dominant_band])

    # Energy in three frequency zones
    low_mask = mel_freqs < 1000
    mid_mask = (mel_freqs >= 1000) & (mel_freqs < 4000)
    high_mask = mel_freqs >= 4000

    total_energy = band_energy.sum() + 1e-9
    energy_low_pct = float(band_energy[low_mask].sum() / total_energy)
    energy_mid_pct = float(band_energy[mid_mask].sum() / total_energy)
    energy_high_pct = float(band_energy[high_mask].sum() / total_energy)

    return {
        "duration_s": round(duration, 2),
        "rms": round(rms, 5),
        "peak_amplitude": round(peak, 5),
        "silence_ratio": round(silence_ratio, 4),
        "spec_min_db": round(spec_min, 2),
        "spec_max_db": round(spec_max, 2),
        "spec_mean_db": round(spec_mean, 2),
        "spec_std_db": round(spec_std, 2),
        "dominant_freq_hz": round(dominant_freq, 1),
        "energy_low_pct": round(energy_low_pct, 4),  # < 1 kHz
        "energy_mid_pct": round(energy_mid_pct, 4),  # 1–4 kHz
        "energy_high_pct": round(energy_high_pct, 4),  # > 4 kHz
    }


def plot_species_audio(data_dir, train_df, species_code, n_samples=1, rating=5.0):
    """
    Plot waveform + mel spectrogram for a species.
    Not called by default — use for manual inspection.
    """
    import librosa.display
    import matplotlib.pyplot as plt

    mask = (train_df["primary_label"] == species_code) & (train_df["rating"] == rating)
    species_samples = train_df[mask].head(n_samples)

    if species_samples.empty:
        print(
            f"Warning: No samples rated {rating} for {species_code}. Using best available..."
        )
        species_samples = (
            train_df[train_df["primary_label"] == species_code]
            .sort_values("rating", ascending=False)
            .head(n_samples)
        )
    if species_samples.empty:
        print(f"Error: No sample found for species: {species_code}")
        return

    for _, row in species_samples.iterrows():
        file_path = os.path.join(data_dir, "train_audio", row["filename"])
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        y, sr = librosa.load(file_path, sr=32000)
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        librosa.display.waveshow(y, sr=sr, ax=ax[0], color="blue")
        ax[0].set_title(f"Waveform: {species_code} (Rating: {row['rating']})")
        ax[0].set_ylabel("Amplitude")

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=224, fmin=50.0, fmax=16000.0
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_dB, sr=sr, x_axis="time", y_axis="mel", fmax=16000, ax=ax[1]
        )
        ax[1].set_title(f"Mel-Spectrogram: {species_code}")
        fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
        plt.tight_layout()
        plt.savefig(f"plot_{species_code}.png")
        print(f"Saved plot_{species_code}.png")


def build_stats_table(
    train_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    audio_dir: str,
    species_filter: list[str] | None = None,
    max_files_per_species: int | None = None,
) -> pd.DataFrame:
    """
    Iterates over training audio files and returns a DataFrame with
    per-file statistics plus metadata (class_name, common_name, rating).
    """
    # Merge metadata
    meta_cols = ["primary_label", "common_name", "class_name"]
    available = [c for c in meta_cols if c in taxonomy_df.columns]
    df = train_df.copy()
    if available:
        df = df.merge(
            taxonomy_df[
                ["primary_label"] + [c for c in available if c != "primary_label"]
            ].drop_duplicates("primary_label"),
            on="primary_label",
            how="left",
        )

    if species_filter:
        df = df[df["primary_label"].isin(species_filter)]
        print(f"Filtered to {len(species_filter)} species: {species_filter}")

    if max_files_per_species:
        df = (
            df.groupby("primary_label", group_keys=False)
            .apply(lambda g: g.head(max_files_per_species))
            .reset_index(drop=True)
        )

    records = []
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting audio stats"):
        file_path = os.path.join(audio_dir, row["filename"])
        if not os.path.exists(file_path):
            errors += 1
            continue

        try:
            stats = extract_audio_stats(file_path)
        except Exception as e:
            errors += 1
            continue

        record = {
            "filename": row["filename"],
            "primary_label": row["primary_label"],
            "rating": row.get("rating", None),
        }
        for col in ["common_name", "class_name"]:
            if col in row:
                record[col] = row[col]

        record.update(stats)
        records.append(record)

    if errors:
        print(f"[WARNING] {errors} file(s) skipped due to errors or missing files.")

    return pd.DataFrame(records)


def main(args):
    print("=== DeepWetlands — Audio Exploration ===")

    train_df = pd.read_csv(args.train_csv)
    taxonomy_df = pd.read_csv(args.taxonomy)

    print(f"Total samples in train.csv : {len(train_df)}")
    print(f"Unique species             : {train_df['primary_label'].nunique()}")

    species_filter = args.species if args.species else None

    df_stats = build_stats_table(
        train_df=train_df,
        taxonomy_df=taxonomy_df,
        audio_dir=args.audio_dir,
        species_filter=species_filter,
        max_files_per_species=args.max_files,
    )

    print(f"\nStats computed for {len(df_stats)} files.")

    summary = (
        df_stats.groupby("primary_label")
        .agg(
            n_files=("filename", "count"),
            mean_duration_s=("duration_s", "mean"),
            mean_rms=("rms", "mean"),
            mean_silence=("silence_ratio", "mean"),
            mean_spec_mean=("spec_mean_db", "mean"),
            mean_spec_std=("spec_std_db", "mean"),
            dominant_freq=("dominant_freq_hz", "mean"),
            energy_low=("energy_low_pct", "mean"),
            energy_mid=("energy_mid_pct", "mean"),
            energy_high=("energy_high_pct", "mean"),
        )
        .reset_index()
        .sort_values("n_files")
    )

    # Save per-file stats
    if args.format == "csv":
        df_stats.to_csv(args.output, index=False)
        summary_path = args.output.replace(".csv", "_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Saved: {args.output}")
        print(f"Saved: {summary_path}")
    else:
        out_json = {
            "per_file": df_stats.to_dict(orient="records"),
            "per_species_summary": summary.to_dict(orient="records"),
        }
        with open(args.output, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"Saved: {args.output}")

    # Print quick summary of species with unusual frequency profiles
    print(
        "\n--- Species with dominant frequency below 1000 Hz (amphibians/low signal) ---"
    )
    low_freq = summary[summary["dominant_freq"] < 1000].sort_values("dominant_freq")
    print(
        low_freq[
            ["primary_label", "n_files", "dominant_freq", "energy_low", "energy_mid"]
        ].to_string(index=False)
    )

    print("\n--- Species with high silence ratio (poor quality recordings) ---")
    silent = summary[summary["mean_silence"] > 0.3].sort_values(
        "mean_silence", ascending=False
    )
    print(
        silent[["primary_label", "n_files", "mean_silence", "mean_rms"]].to_string(
            index=False
        )
    )

    print("\n--- Rarest species (fewest files) ---")
    print(
        summary.head(10)[
            ["primary_label", "n_files", "dominant_freq", "energy_low"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract audio statistics from training set"
    )
    parser.add_argument("--train_csv", default=TRAIN_CSV)
    parser.add_argument("--taxonomy", default=TAXONOMY)
    parser.add_argument("--audio_dir", default=AUDIO_DIR)
    parser.add_argument("--output", default="data/audio_stats.csv")
    parser.add_argument("--format", default="csv", choices=["csv", "json"])
    parser.add_argument(
        "--species",
        nargs="+",
        default=None,
        help="Filter to specific species labels (e.g. --species blchaw1 67252)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Max files per species (use for quick testing)",
    )
    args = parser.parse_args()
    main(args)

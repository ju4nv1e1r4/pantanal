import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F_audio
from tqdm import tqdm

warnings.filterwarnings("ignore")


# Kaggle default paths  (overridden by CLI args for local runs)
KAGGLE_SOUNDSCAPES = Path("/kaggle/input/birdclef-2026/test_soundscapes")
KAGGLE_MODEL       = Path("/kaggle/input/deepwetlands-model/best_model.pth")
KAGGLE_TAXONOMY    = Path("/kaggle/input/birdclef-2026/taxonomy.csv")
KAGGLE_SUBMISSION  = Path("/kaggle/input/birdclef-2026/sample_submission.csv")
KAGGLE_OUTPUT      = Path("/kaggle/working/submission.csv")


# Audio constants (must match training exactly)
TARGET_SR      = 32000
WINDOW_SECONDS = 5
WINDOW_SAMPLES = TARGET_SR * WINDOW_SECONDS   # 160 000 samples per window


# Mel spectrogram (must match GPUAudioTransform in audio_transform.py)
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 224
F_MIN      = 50.0
F_MAX      = 16000.0

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate    = TARGET_SR,
    n_fft          = N_FFT,
    hop_length     = HOP_LENGTH,
    n_mels         = N_MELS,
    f_min          = F_MIN,
    f_max          = F_MAX,
    power          = 2.0,
)

amplitude_to_db = torchaudio.transforms.AmplitudeToDB()


class DeepWetlandsModel(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0",
                 num_classes: int = 234):
        super().__init__()
        import timm
        self.model = timm.create_model(
            model_name,
            pretrained  = False,
            num_classes = num_classes,
            in_chans    = 1,
            drop_rate   = 0.2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(num_classes: int,
                model_name: str = "efficientnet_b0") -> nn.Module:
    return DeepWetlandsModel(model_name=model_name, num_classes=num_classes)

def load_soundscape(path: Path) -> torch.Tensor:
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    y, sr = sf.read(str(path), dtype="float32", always_2d=False)

    # stereo -> mono
    if y.ndim == 2:
        y = y.mean(axis=1)

    # resample if needed
    if sr != TARGET_SR:
        d = gcd(sr, TARGET_SR)
        y = resample_poly(y.astype("float64"),
                          up=TARGET_SR // d,
                          down=sr // d).astype("float32")

    waveform = torch.from_numpy(y).unsqueeze(0)  # [1, N]
    return waveform

def slice_soundscape(
    waveform: torch.Tensor,
    overlap: float = 0.5,
) -> tuple[list[torch.Tensor], list[int]]:
    total_samples = waveform.shape[1]
    hop_samples   = int(WINDOW_SAMPLES * (1.0 - overlap))
    hop_seconds   = WINDOW_SECONDS * (1.0 - overlap)

    windows   = []
    end_times = []
    start     = 0
    t_end     = float(WINDOW_SECONDS)

    while start + WINDOW_SAMPLES <= total_samples:
        chunk = waveform[:, start : start + WINDOW_SAMPLES]
        windows.append(chunk)
        end_times.append(t_end)
        start += hop_samples
        t_end += hop_seconds

    remainder = total_samples - start
    if remainder > 0:
        chunk = waveform[:, start:]
        chunk = torch.nn.functional.pad(chunk, (0, WINDOW_SAMPLES - remainder))
        windows.append(chunk)
        end_times.append(t_end)

    return windows, end_times


def waveform_to_spec(waveform: torch.Tensor) -> torch.Tensor:
    spec = mel_transform(waveform)   # [1, N_MELS, T]
    spec = amplitude_to_db(spec)     # [1, N_MELS, T]
    spec = spec.unsqueeze(0)         # [1, 1, N_MELS, T]
    return spec

@torch.no_grad()
def run_inference(
    soundscape_dir: Path,
    model:          nn.Module,
    label_columns:  list[str],
    sample_sub:     pd.DataFrame,
    batch_size:     int = 32,
    local_test:     bool = False,
    overlap: float = 0.5,
) -> pd.DataFrame:
    device = torch.device("cpu")
    model.eval()
    model.to(device)

    required_row_ids = set() if local_test else set(sample_sub["row_id"].values)

    results = []

    soundscape_files = sorted(soundscape_dir.glob("*.ogg"))
    if not soundscape_files:
        soundscape_files = sorted(soundscape_dir.glob("*.wav"))

    print(f"Found {len(soundscape_files)} soundscape(s) to process.")
    t_start = time.time()

    for sc_path in tqdm(soundscape_files, desc="Soundscapes"):
        sc_stem = sc_path.stem

        try:
            waveform = load_soundscape(sc_path)
        except Exception as e:
            print(f"  [WARNING] Could not load {sc_path.name}: {e}")
            continue

        windows, end_times = slice_soundscape(waveform, overlap=overlap)

        all_probs = []
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i : i + batch_size]
            specs = torch.cat([waveform_to_spec(w) for w in batch_windows], dim=0)
            specs = specs.to(device)

            logits = model(specs)                       # [B, num_classes]
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)  # [N_windows, num_classes]

        from collections import defaultdict
        pooled: dict[int, np.ndarray] = defaultdict(
            lambda: np.full(len(label_columns), -np.inf)
        )
        for j, t_end in enumerate(end_times):
            official_t = max(WINDOW_SECONDS,
                             round(t_end / WINDOW_SECONDS) * WINDOW_SECONDS)
            pooled[official_t] = np.maximum(pooled[official_t], all_probs[j])

        for official_t, probs in pooled.items():
            row_id = f"{sc_stem}_{official_t}"
            if required_row_ids and row_id not in required_row_ids:
                continue
            row = {"row_id": row_id}
            for k, col in enumerate(label_columns):
                row[col] = float(probs[k])
            results.append(row)

    elapsed = time.time() - t_start
    print(f"Inference complete in {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"Generated {len(results)} rows.")

    df_out = pd.DataFrame(results)
    return df_out

def main(args):
    print("=== DeepWetlands — BirdCLEF+ 2026 Inference ===")
    print(f"Soundscapes : {args.soundscapes}")
    print(f"Model       : {args.model}")
    print(f"Taxonomy    : {args.taxonomy}")
    print(f"Output      : {args.output}")

    taxonomy  = pd.read_csv(args.taxonomy)
    classes   = sorted(taxonomy["primary_label"].unique())
    label_map = {label: i for i, label in enumerate(classes)}
    print(f"Classes     : {len(classes)}")

    sample_sub    = pd.read_csv(args.submission)
    label_columns = [c for c in sample_sub.columns if c != "row_id"]
    print(f"Label cols  : {len(label_columns)} (from sample_submission.csv)")

    missing = set(label_columns) - set(classes)
    if missing:
        print(f"[WARNING] {len(missing)} columns in submission not found in taxonomy: {missing}")

    model = build_model(num_classes=len(classes))
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print("Model loaded successfully.")

    df_preds = run_inference(
        soundscape_dir = Path(args.soundscapes),
        model          = model,
        label_columns  = label_columns,
        sample_sub     = sample_sub,
        batch_size     = args.batch_size,
        local_test     = args.local_test,
        overlap        = args.overlap,
    )

    if args.local_test:
        df_preds.to_csv(args.output, index=False)
        print(f"Saved (local test): {args.output}  ({len(df_preds)} rows × {len(df_preds.columns)} cols)")
    else:
        if df_preds.empty:
            print("[WARNING] No predictions generated... submitting uniform prior.")
            df_out = sample_sub.copy()
        else:
            df_out = sample_sub.copy()
            df_out = df_out.set_index("row_id")
            df_preds = df_preds.set_index("row_id")
            df_out.update(df_preds)
            df_out = df_out.reset_index()
        df_out.to_csv(args.output, index=False)
        print(f"Saved: {args.output}  ({len(df_out)} rows × {len(df_out.columns)} cols)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BirdCLEF+ 2026 inference — generates submission.csv"
    )

    parser.add_argument(
        "--soundscapes",
        default=str(KAGGLE_SOUNDSCAPES),
    )
    parser.add_argument(
        "--model",
        default=str(KAGGLE_MODEL),
    )
    parser.add_argument(
        "--taxonomy",
        default=str(KAGGLE_TAXONOMY),
    )
    parser.add_argument(
        "--submission",
        default=str(KAGGLE_SUBMISSION),
    )
    parser.add_argument(
        "--output",
        default=str(KAGGLE_OUTPUT),
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=32,
        help="Windows per batch (reduce if OOM)"
    )
    parser.add_argument(
        "--local_test",
        action="store_true",
        help="Skip row_id filter, save raw predictions"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Window overlap fraction (0.0=none, 0.5=50%%)"
    )

    args = parser.parse_args()
    main(args)

import argparse
import random
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm_gui, tqdm


SPECIES_CONFIG = {
    "67252": {
        "name": "Milk Frog",
        "signal_hz": "700-4000 Hz",
        "snr": "high",
        "enable_pitch_shift": True,
        "pitch_steps": [-2, -1, 1, 2],
        "enable_time_stretch": True,
        "stretch_rates": [0.80, 0.90, 1.10, 1.20],
        "enable_add_noise": True,
        "noise_levels": [0.003, 0.005],
        "enable_window_crop": False,
    },
    "1595929": {
        "name": "Uruguay Harlequin Frog",
        "signal_hz": "~4000 Hz (narrow band) + <100 Hz (pulses)",
        "snr": "high",
        "enable_pitch_shift": False,
        "pitch_steps": [],
        "enable_time_stretch": True,
        "stretch_rates": [0.85, 0.92, 1.08, 1.15],
        "enable_add_noise": False,
        "enable_window_crop": True,
        "window_duration_s": 8.0,
        "window_hop_s": 4.0,
    },
}

TARGET_SR = 32000
MAX_DURATION_S = 30.0

def load_audio(path: Path, target_sr: int = TARGET_SR, max_duration_s: float | None = MAX_DURATION_S) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)

    if y.ndim == 2:
        y = y.mean(axis=1)  # stereo -> mono

    if max_duration_s is not None:
        max_samples = int(max_duration_s * sr)
        if len(y) > max_samples:
            y = y[:max_samples]

    y = np.ascontiguousarray(y, dtype=np.float32)

    if sr != target_sr:
        d = gcd(sr, target_sr)
        y64 = y.astype(np.float64)
        y64 = resample_poly(y64, up=target_sr // d, down=sr // d)
        y = np.ascontiguousarray(y64, dtype=np.float32)

    return y, target_sr

def save_audio(y: np.ndarray, sr: int, out_path: Path, dry_run: bool = False):
    if dry_run:
        print(f"  [dry_run] would write -> {out_path.name}")
        return
    y_out = np.ascontiguousarray(y, dtype=np.float32)
    try:
        sf.write(str(out_path), y_out, sr, format="OGG", subtype="VORBIS")
    except Exception:
        # fallback to WAV
        wav_path = out_path.with_suffix(".wav")
        sf.write(str(wav_path), y_out, sr, format="WAV", subtype="PCM_16")

def augment_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    y_in = np.ascontiguousarray(y, dtype=np.float64)

    precision = 1000
    p = round(rate * precision)
    q = precision
    d = gcd(p, q)
    p, q = p // d, q // d

    y_out = resample_poly(y_in, up=q, down=p)

    n = len(y)
    if len(y_out) >= n:
        y_out = y_out[:n]
    else:
        y_out = np.pad(y_out, (0, n - len(y_out)))

    return np.ascontiguousarray(y_out, dtype=np.float32)

def augment_pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    factor = 2 ** (n_steps / 12.0)

    precision = 1000
    p = round(factor * precision)
    q = precision
    d = gcd(p, q)
    p, q = p // d, q // d

    y_in = np.ascontiguousarray(y, dtype=np.float64)
    y_resampled = resample_poly(y_in, up=q, down=p)

    n = len(y)
    if len(y_resampled) != n:
        d2 = gcd(n, len(y_resampled))
        y_out = resample_poly(y_resampled, up=n // d2, down=len(y_resampled) // d2)
    else:
        y_out = y_resampled

    if len(y_out) >= n:
        y_out = y_out[:n]
    else:
        y_out = np.pad(y_out, (0, n - len(y_out)))

    return np.ascontiguousarray(y_out, dtype=np.float32)

def augment_add_noise(y: np.ndarray, noise_level: float) -> np.ndarray:
    noise = np.random.randn(len(y)).astype(np.float32)
    return np.ascontiguousarray(y + noise_level * noise, dtype=np.float32)

def extract_windows(
    y: np.ndarray,
    sr: int,
    window_s: float,
    hop_s: float
) -> list[np.ndarray]:

    win_samples = int(window_s * sr)
    hop_samples = int(hop_s * sr)
    windows = []
    start = 0
    while start + win_samples <= len(y):
        win = np.ascontiguousarray(y[start: start + win_samples], dtype=np.float32)
        windows.append(win)
        start += hop_samples
    return windows

def augment_class(
    label: str,
    train_audio_dir: Path,
    target_count: int,
    dry_run: bool
):

    cfg = SPECIES_CONFIG[label]
    species_dir = train_audio_dir / label
 
    if not species_dir.exists():
        print(f"\n[WARNING] Directory not found: {species_dir}")
        return
 
    # Originals are files that do NOT start with "aug_".
    originals = sorted(
        f for f in species_dir.glob("*.ogg") if not f.name.startswith("aug_")
    )
    if not originals:
        originals = sorted(
            f for f in species_dir.glob("*.wav") if not f.name.startswith("aug_")
        )
    if not originals:
        print(f"\n[WARNING] No original audio files found in {species_dir}")
        return
 
    existing = (list(species_dir.glob("aug_*.ogg"))
                + list(species_dir.glob("aug_*.wav")))
    current_count = len(originals) + len(existing)
 
    print(f"\n{'─' * 60}")
    print(f"Species      : {label} -- {cfg['name']}")
    print(f"Originals    : {len(originals)}  |  Existing augmented: {len(existing)}")
    print(f"Current total: {current_count}  ->  Target: {target_count}")
 
    if current_count >= target_count:
        print("  Already reached target_count. Nothing to do.")
        return
 
    generated = 0
    needed = target_count - current_count
    pbar = tqdm(total=needed, desc=f"  {label}", unit="sample")
 
    orig_cycle = list(originals) * 10
    random.shuffle(orig_cycle)
    orig_iter = iter(orig_cycle)
 
    def next_original():
        try:
            return next(orig_iter)
        except StopIteration:
            return random.choice(originals)
 
    if cfg.get("enable_window_crop") and generated < needed:
        win_s = cfg["window_duration_s"]
        hop_s = cfg["window_hop_s"]
        for src in originals:
            if generated >= needed:
                break
            y, sr = load_audio(src, max_duration_s=None)
            if len(y) < win_s * sr:
                continue  # recording too short for this window size
            windows = extract_windows(y, sr, win_s, hop_s)
            for i, win in enumerate(windows):
                if generated >= needed:
                    break
                out_name = f"aug_{src.stem}_win{i:02d}.ogg"
                out_path = species_dir / out_name
                if out_path.exists():
                    continue
                save_audio(win, sr, out_path, dry_run)
                generated += 1
                pbar.update(1)

    if cfg.get("enable_time_stretch") and generated < needed:
        for rate in cfg["stretch_rates"]:
            if generated >= needed:
                break
            src = next_original()
            y, sr = load_audio(src)
            y_aug = augment_time_stretch(y, rate)
            out_name = f"aug_{src.stem}_ts{int(rate * 100):03d}.ogg"
            out_path = species_dir / out_name
            if out_path.exists():
                generated += 1
                pbar.update(1)
                continue
            save_audio(y_aug, sr, out_path, dry_run)
            generated += 1
            pbar.update(1)
 
    if cfg.get("enable_pitch_shift") and generated < needed:
        for step in cfg["pitch_steps"]:
            if generated >= needed:
                break
            src = next_original()
            y, sr = load_audio(src)
            y_aug = augment_pitch_shift(y, sr, step)
            sign = "p" if step > 0 else "m"
            out_name = f"aug_{src.stem}_ps{sign}{abs(step)}.ogg"
            out_path = species_dir / out_name
            if out_path.exists():
                generated += 1
                pbar.update(1)
                continue
            save_audio(y_aug, sr, out_path, dry_run)
            generated += 1
            pbar.update(1)
 
    if cfg.get("enable_add_noise") and generated < needed:
        for level in cfg["noise_levels"]:
            if generated >= needed:
                break
            src = next_original()
            y, sr = load_audio(src)
            y_aug = augment_add_noise(y, level)
            out_name = f"aug_{src.stem}_noise{int(level * 1000):03d}.ogg"
            out_path = species_dir / out_name
            if out_path.exists():
                generated += 1
                pbar.update(1)
                continue
            save_audio(y_aug, sr, out_path, dry_run)
            generated += 1
            pbar.update(1)
 
    if generated < needed:
        for _ in range(needed - generated):
            src = next_original()
            y, sr = load_audio(src)
            rate = random.choice(cfg["stretch_rates"])
            steps = (random.choice(cfg["pitch_steps"])
                     if cfg.get("enable_pitch_shift") else 0)
            y_aug = augment_time_stretch(y, rate)
            if steps != 0:
                y_aug = augment_pitch_shift(y_aug, sr, steps)
            rnd = random.randint(1000, 9999)
            out_name = f"aug_{src.stem}_combo{rnd}.ogg"
            out_path = species_dir / out_name
            save_audio(y_aug, sr, out_path, dry_run)
            generated += 1
            pbar.update(1)
 
    pbar.close()

    if not dry_run:
        total_final = (len(originals)
                       + len(list(species_dir.glob("aug_*.ogg")))
                       + len(list(species_dir.glob("aug_*.wav"))))
        print(f"  Generated: {generated}  |  New total: {total_final}")

def sanity_check(label: str, train_audio_dir: Path):
    species_dir = train_audio_dir / label
    aug_files = sorted(
        list(species_dir.glob("aug_*.ogg"))
        + list(species_dir.glob("aug_*.wav"))
    )
    broken = []
    for f in tqdm(aug_files, desc=f"  Sanity check {label}", unit="file"):
        try:
            info = sf.info(str(f))
            duration = info.frames / info.samplerate
            if duration < 0.5:
                broken.append((f.name, f"duration {duration:.2f}s < 0.5s"))
        except Exception as e:
            broken.append((f.name, str(e)))
    if broken:
        print(f"\n  [WARNING] {len(broken)} file(s) with issues:")
        for name, reason in broken:
            print(f"    {name}: {reason}")
    else:
        print(f"  All {len(aug_files)} augmented files OK.")

def main():
    parser = argparse.ArgumentParser(
        description="Offline augmentation for rare-class species in BirdCLEF+ 2026"
    )
    parser.add_argument(
        "--data_dir", type=Path, default=Path("data"),
        help="Dataset root directory (contains train_audio/)"
    )
    parser.add_argument(
        "--labels", nargs="+", default=["67252", "1595929"],
        help="Labels to augment (default: 67252 1595929)"
    )
    parser.add_argument(
        "--target_count", type=int, default=25,
        help="Total samples per class after augmentation (originals + synthetic)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Simulate without writing any files"
    )
    parser.add_argument(
        "--skip_sanity", action="store_true",
        help="Skip the post-generation sanity check"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_audio_dir = args.data_dir / "train_audio"
    if not train_audio_dir.exists():
        print(f"[ERROR] {train_audio_dir} not found. Check --data_dir.")
        return

    print(f"\nDeepWetlands -- Offline Augmentation for Rare Classes")
    print(f"{'=' * 60}")
    print(f"data_dir     : {args.data_dir.resolve()}")
    print(f"train_audio  : {train_audio_dir.resolve()}")
    print(f"labels       : {args.labels}")
    print(f"target_count : {args.target_count} samples per class")
    print(f"dry_run      : {args.dry_run}")
    print(f"seed         : {args.seed}")

    for label in args.labels:
        if label not in SPECIES_CONFIG:
            print(f"\n[WARNING] {label} is not in SPECIES_CONFIG -- skipping.")
            continue
        augment_class(label, train_audio_dir, args.target_count, args.dry_run)

    if not args.dry_run and not args.skip_sanity:
        print(f"\n{'─' * 60}")
        print("Running post-generation sanity check...")
        for label in args.labels:
            if label in SPECIES_CONFIG:
                sanity_check(label, train_audio_dir)

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print("Dry run complete. Re-run without --dry_run to write files.")
    else:
        print("Augmentation complete.")
        print("Next step: run training as normal -- the DataLoader will pick up")
        print("the new aug_*.ogg files automatically.")


if __name__ == "__main__":
    main()

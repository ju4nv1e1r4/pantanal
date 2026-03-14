import random
import shutil
from pathlib import Path
import argparse
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


SPECIES_CONFIG = {
    "67252": {
        "name": "Milk Frog",
        "signal_hz": "700–4000 Hz",
        "snr": "alto",
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
        "signal_hz": "~4000 Hz (narrowband) + <100 Hz (pulses)",
        "snr": "alto",
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

def load_audio(path: Path, target_sr: int = TARGET_SR):
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y, sr

def save_audio(y: np.ndarray, sr: int, out_path: Path, dry_run: bool = False):
    if dry_run:
        print(f"  [dry_run] would write -> {out_path.name}")
        return
    sf.write(str(out_path), y, sr, format="OGG", subtype="VORBIS")

def augment_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate=rate)

def augment_pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def augment_add_noise(y: np.ndarray, noise_level: float) -> np.ndarray:
    noise = np.random.randn(len(y)).astype(np.float32)
    return y + noise_level * noise

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
        windows.append(y[start : start + win_samples])
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
        print(f"\n[WARNING] Dir not found: {species_dir}")
        return

    originals = sorted(species_dir.glob("*.ogg"))
    if not originals:
        originals = sorted(species_dir.glob("*.wav"))
    if not originals:
        print(f"\n[WARNING] These is no audio in {species_dir}")
        return

    existing = list(species_dir.glob("aug_*.ogg")) + list(species_dir.glob("aug_*.wav"))
    current_count = len(originals) + len(existing)

    print(f"\n{'-'*60}")
    print(f"Species: {label} -- {cfg['name']}")
    print(f"Originals: {len(originals)} | Existing Augmented: {len(existing)}")
    print(f"Current total: {current_count} -> target: {target_count}")

    if current_count >= target_count:
        print("Already reached target_count. Nothing to do.")
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
            y, sr = load_audio(src)
            if len(y) < win_s * sr:
                continue  # very short audio on window
            windows = extract_windows(y, sr, win_s, hop_s)
            for i, win in enumerate(windows):
                if generated >= needed:
                    break
                out_name = f"aug_{src.stem}_win{i:02d}.ogg"
                out_path = species_dir / out_name
                if out_path.exists():
                    continue
                save_audio(win.astype(np.float32), sr, out_path, dry_run)
                generated += 1
                pbar.update(1)

    if cfg.get("enable_time_stretch") and generated < needed:
        for rate in cfg["stretch_rates"]:
            if generated >= needed:
                break
            src = next_original()
            y, sr = load_audio(src)
            y_aug = augment_time_stretch(y, rate)
            out_name = f"aug_{src.stem}_ts{int(rate*100):03d}.ogg"
            out_path = species_dir / out_name
            if out_path.exists():
                generated += 1
                pbar.update(1)
                continue
            save_audio(y_aug.astype(np.float32), sr, out_path, dry_run)
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
            save_audio(y_aug.astype(np.float32), sr, out_path, dry_run)
            generated += 1
            pbar.update(1)

    if cfg.get("enable_add_noise") and generated < needed:
        for level in cfg["noise_levels"]:
            if generated >= needed:
                break
            src = next_original()
            y, sr = load_audio(src)
            y_aug = augment_add_noise(y, level)
            out_name = f"aug_{src.stem}_noise{int(level*1000):03d}.ogg"
            out_path = species_dir / out_name
            if out_path.exists():
                generated += 1
                pbar.update(1)
                continue
            save_audio(y_aug.astype(np.float32), sr, out_path, dry_run)
            generated += 1
            pbar.update(1)

    if generated < needed:
        for _ in range(needed - generated):
            src = next_original()
            y, sr = load_audio(src)
            rate  = random.choice(cfg["stretch_rates"])
            steps = random.choice(cfg["pitch_steps"]) if cfg.get("enable_pitch_shift") else 0
            y_aug = augment_time_stretch(y, rate)
            if steps != 0:
                y_aug = augment_pitch_shift(y_aug, sr, steps)
            rnd = random.randint(1000, 9999)
            out_name = f"aug_{src.stem}_combo{rnd}.ogg"
            out_path = species_dir / out_name
            save_audio(y_aug.astype(np.float32), sr, out_path, dry_run)
            generated += 1
            pbar.update(1)

    pbar.close()
    total_final = len(originals) + len(list(species_dir.glob("aug_*.ogg")))
    if not dry_run:
        print(f"Generated: {generated} | Final: {total_final}")

def sanity_check(label: str, train_audio_dir: Path):
    species_dir = train_audio_dir / label
    aug_files = sorted(species_dir.glob("aug_*.ogg"))
    broken = []
    for f in tqdm(aug_files, desc=f"Sanity check -- {label}", unit="file"):
        try:
            y, sr = librosa.load(str(f), sr=None, mono=True, duration=5.0)
            if len(y) < sr * 0.5:  # less than 0.5s = suspicious
                broken.append((f.name, "duration < 0.5s"))
        except Exception as e:
            broken.append((f.name, str(e)))
    if broken:
        print(f"\n  [WARNING] {len(broken)} there is a file or too many files with problems:")
        for name, reason in broken:
            print(f"Name: {name} -> Possible reason: {reason}")
    else:
        print(f"  All {len(aug_files)} files augmented OK.")

def main():
    parser = argparse.ArgumentParser(
        description="Augmentation offline for rare classes of BirdCLEF+ 2026"
    )
    parser.add_argument(
        "--data_dir", type=Path, default=Path("data"),
        help="Dataset root (contains train_audio/)"
    )
    parser.add_argument(
        "--labels", nargs="+", default=["67252", "555123"],
        help="Labels to augment (default: 67252 555123)"
    )
    parser.add_argument(
        "--target_count", type=int, default=25,
        help="Total sample per class augm. (originais + sintéticos)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Simulation, but without record any file"
    )
    parser.add_argument(
        "--skip_sanity", action="store_true",
        help="Step up sanity check post gen."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed (reproducibility)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_audio_dir = args.data_dir / "train_audio"
    if not train_audio_dir.exists():
        print(f"[ERROR] {train_audio_dir} not found. Check --data_dir.")
        return

    print(f"\nDeepWetlands — Offline augmentation for rare classes")
    print(f"{'='*60}")
    print(f"data_dir     : {args.data_dir.resolve()}")
    print(f"train_audio  : {train_audio_dir.resolve()}")
    print(f"labels       : {args.labels}")
    print(f"target_count : {args.target_count} samples per class")
    print(f"dry_run      : {args.dry_run}")
    print(f"seed         : {args.seed}")

    for label in args.labels:
        if label not in SPECIES_CONFIG:
            print(f"\n[WARNING] {label} not in SPECIES_CONFIG — steping.")
            continue
        augment_class(label, train_audio_dir, args.target_count, args.dry_run)

    if not args.dry_run and not args.skip_sanity:
        print(f"\n{'─'*60}")
        print("Sanity check for all generated files...")
        for label in args.labels:
            if label in SPECIES_CONFIG:
                sanity_check(label, train_audio_dir)

    print(f"\n{'='*60}")
    if args.dry_run:
        print("dry_run is d'one. Run without --dry_run now to start the record.")
    else:
        print("Augmentação is done.")
        print("Next step: run the train — DataLoader")
        print("will find all files in aug_*.ogg automatically.")


if __name__ == "__main__":
    main()

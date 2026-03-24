import os
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.ml.inference import build_model, run_inference


def main(args):
    print("=== DeepWetlands — Local Soundscape Validation ===")

    taxonomy = pd.read_csv(args.taxonomy)
    classes = sorted(taxonomy["primary_label"].unique())

    cache_file = "preds_cache.csv"
    print(f"Taxonomy loaded: {len(classes)} classes.")

    print(f"Loading model: {args.model}")
    if os.path.exists(cache_file):
        print(f"Predictions cache found -> ({cache_file})!")
        df_preds = pd.read_csv(cache_file)
    else:
        print(f"Building model: {args.model}")
        model = build_model(num_classes=len(classes), model_name=args.model_name)

        state = torch.load(args.model, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

        print("\nExtracting predictions...")
        dummy_sub = pd.DataFrame(columns=["row_id"] + classes)

        df_preds = run_inference(
            soundscape_dir=Path(args.soundscapes),
            model=model,
            label_columns=classes,
            sample_sub=dummy_sub,
            batch_size=args.batch_size,
            local_test=True,
            overlap=0.5
        )
        df_preds.to_csv(cache_file, index=False)
        print(f"Predictions saved to cache file {cache_file}!")

    df_true_raw = pd.read_csv(args.labels)
    print(f"\nOriginal Ground Truth columns: {list(df_true_raw.columns)}")

    col_audio = "filename"
    col_time = "end"
    col_labels = "primary_label"

    if "row_id" not in df_true_raw.columns:
        df_true_raw["clean_filename"] = df_true_raw[col_audio].apply(lambda x: Path(str(x).strip()).stem)

        def parse_time_to_seconds(t):
            t = str(t).strip()
            if ':' in t:
                parts = t.split(':')
                return int(sum(float(x) * 60 ** i for i, x in enumerate(reversed(parts))))
            return int(float(t))

        df_true_raw["clean_time"] = df_true_raw[col_time].apply(parse_time_to_seconds).astype(str)
        df_true_raw["row_id"] = df_true_raw["clean_filename"] + "_" + df_true_raw["clean_time"]

    print(f"\n[DEBUG] Example row_id in Predictions : '{df_preds['row_id'].iloc[0]}'")
    print(f"[DEBUG] Example row_id in Ground Truth  : '{df_true_raw['row_id'].iloc[0]}'")

    df_true = pd.DataFrame({"row_id": df_true_raw["row_id"].unique()})
    for c in classes:
        df_true[c] = 0.0

    for _, row in df_true_raw.iterrows():
        r_id = row["row_id"]
        clean_str = str(row[col_labels]).replace('[', ' ').replace(']', ' ').replace("'", ' ').replace('"', ' ').replace(',', ' ')
        birds = clean_str.split()
        for b in birds:
            if b in classes:
                df_true.loc[df_true["row_id"] == r_id, b] = 1.0

    df_merged = pd.merge(df_preds, df_true, on="row_id", suffixes=("_pred", "_true"))
    if len(df_merged) == 0:
        print("ERROR: No common windows found!")
        return
    else:
        print(f"Merge complete! {len(df_merged)} common windows found.")

    y_pred = df_merged[[c + "_pred" for c in classes]].values
    y_true = df_merged[[c + "_true" for c in classes]].values

    mask = y_true.sum(axis=0) > 0
    print(f"\n[DEBUG] Classes with at least 1 positive audio in the {len(df_merged)} windows: {mask.sum()} of {len(classes)}")

    if mask.sum() == 0:
        print("FATAL ERROR: The Ground Truth is 100% zero. No class matched!")
        print(f"[Info] Example from 'primary_label' column: {df_true_raw[col_labels].unique()[:5]}")
        return

    if np.isnan(y_pred).any():
        print("FATAL ERROR: Your model is predicting NaN! The weights might have exploded during training.")
        return

    print("\n" + "="*50)
    mask = y_true.sum(axis=0) > 0

    y_true_present = y_true[:, mask]
    y_pred_present = y_pred[:, mask]

    macro_auc = roc_auc_score(y_true_present, y_pred_present, average="macro")

    print(f"LOCAL SCORE (Present Classes - Macro-AUC): {macro_auc:.4f}")
    print(f"   (Calculated over the {mask.sum()} classes mapped in the soundscape)")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Soundscape Validation")
    parser.add_argument("--soundscapes", default="data/train_soundscapes")
    parser.add_argument("--labels", default="data/train_soundscapes_labels.csv")
    parser.add_argument("--taxonomy", default="data/taxonomy.csv")
    parser.add_argument("--model", required=True, help="Path to the .pth model")
    parser.add_argument("--model_name", default="efficientnet_b3", help="Name of the timm architecture")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    main(args)

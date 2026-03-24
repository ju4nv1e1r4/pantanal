import os
import json
import time
import logging
import datetime
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from src.ml.constants import (
    HTML_TEMPLATE,
    ROW_TEMPLATE,
    CLASS_ROW_TEMPLATE,
    WORST_CLASSES_SECTION_TEMPLATE,
    BADGE_HTML
)

TOP_K_ERRORS = 15  # labels to show
PLOT_DPI     = 150

COLORS = {
    "bg":        "#0f1117",
    "surface":   "#1a1d27",
    "border":    "#2e3147",
    "train":     "#4f9cf9",
    "val":       "#f97316",
    "accent":    "#a78bfa",
    "good":      "#34d399",
    "bad":       "#f87171",
    "text":      "#e2e8f0",
    "subtext":   "#94a3b8",
}


class TrainingLogger:
    def __init__(self, label_map: dict, output_dir: str = "logs/run", model_name = "efficientnet_b3"):
        self.model = model_name
        self.label_map    = label_map
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.output_dir   = Path(output_dir)

        if self.output_dir.exists():
            name = self.output_dir.name
            parent = self.output_dir.parent
            match = re.search(r"(.*?)(\d+)$", name)
            if match:
                prefix, num_str = match.groups()
                num = int(num_str)
                while self.output_dir.exists():
                    num += 1
                    self.output_dir = parent / f"{prefix}{num:0{len(num_str)}d}"
            else:
                i = 1
                while self.output_dir.exists():
                    self.output_dir = parent / f"{name}_{i}"
                    i += 1

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = [] # [{epoch, train_loss, val_loss, macro_auc, per_class_auc, epoch_time, gpu_mb}]
        self.start_time = time.time()

        self._setup_file_logger()
        if HAS_RICH:
            self.console = Console()

        self.log(f"TrainingLogger started -> {os.path.relpath(self.output_dir)}")
        self.log(f"Classes: {len(label_map)} | Timestamp: {datetime.datetime.now().isoformat()} | Model: {self.model}")

    # called at the end of each epoch on train script
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_preds: np.ndarray,   # [N, num_classes] — logits (our probs)
        val_targets: np.ndarray, # [N, num_classes] — one-hot
        epoch_time: float,       # seconds
    ):
        gpu_mb = self._gpu_memory_mb()

        macro_auc, per_class_auc = self._compute_auc(val_preds, val_targets)

        record = {
            "model_name":    self.model,
            "epoch":         epoch,
            "train_loss":    round(train_loss, 6),
            "val_loss":      round(val_loss, 6),
            "macro_auc":     round(macro_auc, 6) if macro_auc is not None else None,
            "per_class_auc": per_class_auc,
            "epoch_time":    round(epoch_time, 2),
            "gpu_mb":        gpu_mb,
        }
        self.history.append(record)

        self._print_epoch_summary(record)
        self._append_to_log(record)

    # called when the training has finished and generate all artifacts
    def finalize(self):
        total_time = time.time() - self.start_time
        self.log(f"\nTraining finished in {self._fmt_time(total_time)} for {self.model}")

        self._plot_loss_curve()
        self._plot_top_k_errors()
        self._plot_gpu_memory()
        self._generate_html(total_time)

        if HAS_RICH:
            self._print_final_summary(total_time)

    def _compute_auc(self, preds, targets):
        if not HAS_SKLEARN or preds is None or targets is None:
            return None, {}

        # convert logits to probabilities (if necessary)
        if preds.max() > 1.0 or preds.min() < 0.0:
            preds = 1 / (1 + np.exp(-preds))  # sigmoid

        per_class_auc = {}
        valid_aucs    = []

        for i in range(targets.shape[1]):
            y_true = targets[:, i]
            y_score = preds[:, i]
            if y_true.sum() == 0:
                continue  # no positives, undefined AUC
            try:
                auc = roc_auc_score(y_true, y_score)
                label = self.idx_to_label.get(i, str(i))
                per_class_auc[label] = round(float(auc), 4)
                valid_aucs.append(auc)
            except Exception:
                pass

        macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None
        return macro_auc, per_class_auc

    def _plot_loss_curve(self):
        epochs     = [r["epoch"] for r in self.history]
        train_loss = [r["train_loss"] for r in self.history]
        val_loss   = [r["val_loss"]   for r in self.history]
        macro_auc  = [r["macro_auc"]  for r in self.history if r["macro_auc"] is not None]

        has_auc = len(macro_auc) == len(epochs)
        fig = plt.figure(figsize=(14, 5), facecolor=COLORS["bg"])
        gs  = gridspec.GridSpec(1, 2 if has_auc else 1, figure=fig)

        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(COLORS["surface"])
        ax1.plot(epochs, train_loss, color=COLORS["train"], lw=2.5, marker="o", ms=5, label="Train Loss")
        ax1.plot(epochs, val_loss,   color=COLORS["val"],   lw=2.5, marker="o", ms=5, label="Val Loss")
        ax1.fill_between(epochs, train_loss, val_loss, alpha=0.08, color=COLORS["accent"])
        ax1.set_title(f"Loss per Epoch [{self.model}]", color=COLORS["text"], fontsize=13, pad=12)
        ax1.set_xlabel("Epoch", color=COLORS["subtext"])
        ax1.set_ylabel("Loss", color=COLORS["subtext"])
        ax1.legend(facecolor=COLORS["border"], labelcolor=COLORS["text"])
        ax1.tick_params(colors=COLORS["subtext"])
        for spine in ax1.spines.values():
            spine.set_edgecolor(COLORS["border"])

        if has_auc:
            ax2 = fig.add_subplot(gs[1])
            ax2.set_facecolor(COLORS["surface"])
            ax2.plot(epochs, macro_auc, color=COLORS["good"], lw=2.5, marker="o", ms=5, label="Macro-AUC")
            ax2.set_ylim(0, 1)
            ax2.set_title(f"Macro-AUC per Epoch [{self.model}]", color=COLORS["text"], fontsize=13, pad=12)
            ax2.set_xlabel("Epoch", color=COLORS["subtext"])
            ax2.set_ylabel("AUC", color=COLORS["subtext"])
            ax2.legend(facecolor=COLORS["border"], labelcolor=COLORS["text"])
            ax2.tick_params(colors=COLORS["subtext"])
            for spine in ax2.spines.values():
                spine.set_edgecolor(COLORS["border"])

        fig.tight_layout(pad=2.0)
        path = self.output_dir / f"{self.model}_loss_curve.png"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", facecolor=COLORS["bg"])
        plt.close(fig)
        self.log(f"Plot saved on: {path}")

    def _plot_top_k_errors(self):
        last = self.history[-1]
        per_class = last.get("per_class_auc", {})
        if not per_class:
            return

        sorted_classes = sorted(per_class.items(), key=lambda x: x[1])
        worst_k = sorted_classes[:TOP_K_ERRORS]
        best_k  = sorted_classes[-5:]

        labels = [c[0] for c in worst_k]
        aucs   = [c[1] for c in worst_k]
        bar_colors = [COLORS["bad"] if a < 0.7 else COLORS["val"] if a < 0.85 else COLORS["good"] for a in aucs]

        fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS["bg"])
        ax.set_facecolor(COLORS["surface"])

        bars = ax.barh(labels, aucs, color=bar_colors, edgecolor=COLORS["border"], height=0.65)
        ax.axvline(x=last["macro_auc"] or 0.5, color=COLORS["accent"], lw=1.5, linestyle="--", label=f'Macro-AUC: {last["macro_auc"]:.4f}')
        ax.set_xlim(0, 1)
        ax.set_title(f"Top-{TOP_K_ERRORS} Worst AUC per labels [{self.model}] (Epoch {last['epoch']})", color=COLORS["text"], fontsize=13, pad=12)
        ax.set_xlabel("AUC", color=COLORS["subtext"])
        ax.legend(facecolor=COLORS["border"], labelcolor=COLORS["text"])
        ax.tick_params(colors=COLORS["subtext"])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])

        for bar, auc in zip(bars, aucs):
            ax.text(auc + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{auc:.3f}", va="center", color=COLORS["text"], fontsize=8)

        fig.tight_layout(pad=2.0)
        path = self.output_dir / f"{self.model}_top_k_errors.png"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", facecolor=COLORS["bg"])
        plt.close(fig)
        self.log(f"Plot saved on: {path}")

    def _plot_gpu_memory(self):
        gpu_data = [(r["epoch"], r["gpu_mb"]) for r in self.history if r["gpu_mb"] is not None]
        if not gpu_data:
            return

        epochs, mbs = zip(*gpu_data)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=COLORS["bg"])
        ax.set_facecolor(COLORS["surface"])
        ax.fill_between(epochs, mbs, alpha=0.3, color=COLORS["accent"])
        ax.plot(epochs, mbs, color=COLORS["accent"], lw=2.5, marker="o", ms=5)
        ax.set_title(f"GPU memory usage per Epoch [{self.model}]", color=COLORS["text"], fontsize=13, pad=12)
        ax.set_xlabel("Epoch", color=COLORS["subtext"])
        ax.set_ylabel("MB allocated", color=COLORS["subtext"])
        ax.tick_params(colors=COLORS["subtext"])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])

        fig.tight_layout(pad=2.0)
        path = self.output_dir / f"{self.model}_gpu_memory.png"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", facecolor=COLORS["bg"])
        plt.close(fig)
        self.log(f"Plot saved on: {path}")


    def _generate_html(self, total_time: float):
        best_epoch   = min(self.history, key=lambda r: r["val_loss"])
        last         = self.history[-1]
        timestamp    = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

        rows = ""
        for r in self.history:
            is_best = r["epoch"] == best_epoch["epoch"]
            style   = 'style="background:#1e2a1e;"' if is_best else ""
            auc_str = f"{r['macro_auc']:.4f}" if r["macro_auc"] is not None else "—"
            best_badge = BADGE_HTML if is_best else ""
            gpu_str = f"{r['gpu_mb'] or '—'} MB"

            rows += ROW_TEMPLATE.format(
                style=style,
                epoch=r['epoch'],
                best_badge=best_badge,
                train_loss=r['train_loss'],
                val_loss=r['val_loss'],
                auc_str=auc_str,
                epoch_time=self._fmt_time(r['epoch_time']),
                gpu_mb=gpu_str
            )

        per_class = last.get("per_class_auc", {})
        worst_classes_content = ""
        worst_classes_section = ""

        if per_class:
            worst_10 = sorted(per_class.items(), key=lambda x: x[1])[:10]
            for label, auc in worst_10:
                color = "#f87171" if auc < 0.7 else "#fb923c" if auc < 0.85 else "#34d399"
                pct   = int(auc * 100)
                worst_classes_content += CLASS_ROW_TEMPLATE.format(
                    label=label,
                    pct=pct,
                    color=color,
                    auc=auc
                )
            worst_classes_section = WORST_CLASSES_SECTION_TEMPLATE.format(content=worst_classes_content)

        best_train_loss_val = min(r['train_loss'] for r in self.history)
        best_train_epoch_val = min(self.history, key=lambda r: r['train_loss'])['epoch']

        macro_aucs = [r['macro_auc'] for r in self.history if r['macro_auc'] is not None]
        best_macro_auc_val = f"{max(macro_aucs):.4f}" if macro_aucs else "—"

        html = HTML_TEMPLATE.format(
            timestamp=timestamp,
            num_epochs=len(self.history),
            num_classes=len(self.label_map),
            total_time=self._fmt_time(total_time),
            best_train_loss=f"{best_train_loss_val:.4f}",
            best_train_epoch=best_train_epoch_val,
            best_val_loss=f"{best_epoch['val_loss']:.4f}",
            best_val_epoch=best_epoch['epoch'],
            best_macro_auc=best_macro_auc_val,
            avg_time_epoch=self._fmt_time(total_time / len(self.history)),
            rows=rows,
            worst_classes_section=worst_classes_section,
            model_name=self.model,
            img_loss_curve=f"{self.model}_loss_curve.png",
            img_top_k=f"{self.model}_top_k_errors.png",
            img_gpu=f"{self.model}_gpu_memory.png"
        )

        path = self.output_dir / f"{self.model}_report.html"
        path.write_text(html, encoding="utf-8")
        self.log(f"HTML Report saved: {path}")


    def _print_epoch_summary(self, record):
        if not HAS_RICH:
            print(f"[Epoch {record['epoch']}] train={record['train_loss']:.4f} val={record['val_loss']:.4f} "
                  f"auc={record['macro_auc'] or '—'} t={self._fmt_time(record['epoch_time'])} gpu={record['gpu_mb']}MB")
            return

        auc_str = f"{record['macro_auc']:.4f}" if record["macro_auc"] is not None else "[dim]—[/]"
        gpu_str = f"{record['gpu_mb']} MB" if record["gpu_mb"] else "[dim]—[/]"
        self.console.print(
            f"  [bold cyan]Epoch {record['epoch']}[/]  "
            f"train=[bold blue]{record['train_loss']:.4f}[/]  "
            f"val=[bold yellow]{record['val_loss']:.4f}[/]  "
            f"auc=[bold green]{auc_str}[/]  "
            f"t=[magenta]{self._fmt_time(record['epoch_time'])}[/]  "
            f"gpu=[dim]{gpu_str}[/]"
        )

    def _print_final_summary(self, total_time):
        best = min(self.history, key=lambda r: r["val_loss"])
        last = self.history[-1]

        table = Table(title="Final Training Summary", box=box.SIMPLE_HEAVY,
                      style="dim", title_style="bold white")
        table.add_column("Metric",    style="cyan", width=28)
        table.add_column("Value",      style="bold white", justify="right")

        table.add_row("Architecture",          self.model)
        table.add_row("Trained Epochs",        str(len(self.history)))
        table.add_row("Best Val Loss",          f"{best['val_loss']:.4f}  (epoch {best['epoch']})")
        table.add_row("Last Macro-AUC",         f"{last['macro_auc']:.4f}" if last["macro_auc"] else "—")
        table.add_row("Total Time",              self._fmt_time(total_time))
        table.add_row("Avg Time / Epoch",      self._fmt_time(total_time / len(self.history)))
        table.add_row("Output Dir",               os.path.relpath(self.output_dir))

        self.console.print()
        self.console.print(Panel(table, border_style="bright_black"))

    def _setup_file_logger(self):
        log_path = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
            ]
        )
        self.file_logger = logging.getLogger("pantanal")

    def log(self, msg: str):
        self.file_logger.info(msg)
        if HAS_RICH:
            self.console.print(f"[dim]{msg}[/]")
        else:
            print(msg)

    def _append_to_log(self, record: dict):
        self.file_logger.info(json.dumps(record))

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = int(seconds)
        h, rem  = divmod(seconds, 3600)
        m, s    = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    @staticmethod
    def _gpu_memory_mb():
        if not HAS_TORCH:
            return None
        try:
            if torch.cuda.is_available():
                return round(torch.cuda.memory_allocated() / 1024 ** 2, 1)
        except Exception:
            pass
        return None

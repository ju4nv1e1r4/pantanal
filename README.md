DeepWetlands for Kaggle BirdCLEF+ 2026

This repository is dedicated to the challenge of biodiversity monitoring in the Brazilian Pantanal. The goal is to develop Deep Learning models capable of identifying 234 species (birds, mammals, reptiles, etc.) through passive audio.

## Technology Stack

**Framework:** PyTorch & PyTorch Lightning (via timm for vision models)  
**Audio:** soundfile (I/O) and torchaudio (GPU Transforms)  
**Hardware:** Trained locally on NVIDIA RTX 4050 (6GB VRAM) / Intel Core i7 (Raptor Lake)  
**Logging:** Custom system generating HTML, Matplotlib plots, and detailed reports per architecture

---

## Logbook and Engineering Solutions

The development of our first Baseline (EfficientNet-B0) required solving several chronic audio processing bottlenecks in Deep Learning.

### 1. PyTorch Backend Issue (Audio I/O)

**The Challenge:**  
The torchaudio version combined with PyTorch 2.10+ and CUDA 12.8 deprecated old backends and forced the use of `torchcodec`, which failed to load the competition `.ogg` files due to FFmpeg issues.

**The Solution:**  
We abandoned the standard loading method and forced the use of the pure C `soundfile` library (`sf.read`), manually transposing the tensors from `[samples, channels]` to `[channels, samples]`.

---

### 2. CPU Thermal Bottleneck vs GPU Idleness

**The Challenge:**  
In Exp 001, `torchaudio.transforms.MelSpectrogram` and `Resample` were running on the CPU inside the DataLoader. The CPU reached critical temperature limits (98°C / Thermal Throttling), while the RTX 4050 operated at 3% usage with only 85 MB of VRAM allocated.

**The Solution:**  
We transferred the heavy mathematical transformation class into the GPU (`GPUAudioTransform()`). The DataLoader started delivering only the raw waveform. This woke up the graphics card, but created a new problem.

---

### 3. CUDA Out of Memory (OOM) and the Mathematical Cure

**The Challenge:**  
By moving spectrogram generation and training to the 6GB GPU, trying to use a `batch_size` of 64 or 128 caused a VRAM overflow.

**The Solution:**  
We implemented two industrial-grade techniques.

**Automatic Mixed Precision (AMP)**  
We used `torch.amp.autocast` and `GradScaler` to calculate the forward pass in 16-bits, halving memory consumption.

**Gradient Accumulation**  
We reduced the real `batch_size` to 16 (which fits comfortably in the GPU) and accumulated the gradients over 4 steps (`ACCUM_STEPS = 4`), resulting in an effective batch of 64 and ensuring mathematical stability without exceeding hardware limits.

---

### 4. Overfitting and the MixUp Paradox

**The Challenge:**  
In Exp 002 (20 Epochs), we noticed clear overfitting (Train Loss crossed and stayed below Val Loss). The model started memorizing noise from the training audio.

**The Solution:**  
In Exp 003, we implemented MixUp (mixing two audios and their labels with a 50% chance).

**The Side Effect:**  
MixUp destroyed the Macro-AUC of certain numeric classes (insect sound types like `67252`), as the continuous insect sound was mathematically erased when mixed with a strong bird call.

---

## Baseline Results (EfficientNet-B0)

Our base model is **EfficientNet-B0** (initialized via timm with `in_chans=1`). The model is lightweight (~4.3M parameters), ideal for Kaggle CPU inference restrictions.

| Metric | Exp 001 (10 Ep) | Exp 002 (20 Ep) | Exp 003 (30 Ep + Pro Setup) |
|------|------|------|------|
| Total Time | 38m 26s | 1h 20m 56s | 2h 48m 29s |
| Best Val Loss | 0.0061 | 0.0060 | 0.0057 |
| Best Macro-AUC | 0.9881 | 0.9849 | 0.9869 |
| Overfitting | Not measured | Yes | Cured (MixUp) |
| Worst Class (67252) | 0.601 | 0.601 | 0.411 |


---

## Next Steps (Next Experiments)

Since EfficientNet-B0 stagnated in its ability to separate background noise and insects from real calls under the effect of MixUp, the next experimentation phase will test models with a larger receptive field.

### convnext_nano
Imminent replacement. Its **7x7 convolutions** (compared to EfficientNet's 3x3) should allow the model to understand a broader temporal context, mitigating the insect class problem.

### swin_tiny
Future test of **attention mechanisms** for noisy audio environments.
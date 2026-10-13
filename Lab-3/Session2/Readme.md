# OWL-ViT Lab on the DGX Spark: Getting Started

You will upload your CVAT dataset to the DGX, prepare it, fine-tune OWL-ViT on the DGX's GPU
with three different settings, download the trained model, and plot the results for your lab report.

## Before you start

You need:

| What | Example |
| --- | --- |
| Your group account | `group02` |
| Your DGX number | `spark1`, `spark2` or `spark3` |
| The DGX address | shown in class (written as `DGX-IP` below) |
| Your photo folder, with one sub-folder per class | `MyDataset/` with 5 sub-folders of 125 photos each |
| The COCO JSON exported from CVAT | `instances_default.json` |

To get the JSON from CVAT, export your task or project with the format **COCO 1.0**. Unzip the
download. The JSON is `annotations/instances_default.json`.

In every command below, replace:

- `groupXX` with your group account, for example `group02`
- `DGX-IP` with the address shown in class
- `MyDataset` with the name of your photo folder

Each command block says where to run it:

- **Laptop** means a terminal on your own computer: Terminal on macOS/Linux, PowerShell on Windows.
- **DGX** means your SSH session on the DGX.
- **Container** means inside the training container, after step 3.

> Tip: avoid spaces in folder and file names. If a name has spaces, put the whole path in quotes.

---

## 1. Log in to the DGX

**Laptop:**

```bash
ssh groupXX@DGX-IP
```

Your password is **`spark#-group#`**:

- `spark#` is your DGX: `spark1`, `spark2` or `spark3`
- `group#` is your group, with two digits: `group01` to `group10`

> **Example:** group02 on DGX spark1 uses the password **`spark1-group02`**.

- **Typing the password:** nothing appears on screen while you type it. That's normal; press Enter when you're done.
- **"Are you sure you want to continue connecting?":** you'll see this the first time. Type `yes`.
- **This password is fixed** for your group. It can't be changed.

You're logged in when the prompt looks like `groupXX@spark-....:~$`.

---

## 2. Upload your photos and the JSON

Run these in a **new terminal window on your laptop**, not in the DGX session. They put everything
straight into the right folder on the DGX.

**Laptop (macOS / Linux):**

```bash
scp -r ~/Downloads/MyDataset groupXX@DGX-IP:~/data/
scp ~/Downloads/instances_default.json groupXX@DGX-IP:~/data/MyDataset/
```

**Laptop (Windows PowerShell):**

```powershell
scp -r C:\Users\YOU\Downloads\MyDataset groupXX@DGX-IP:~/data/
scp C:\Users\YOU\Downloads\instances_default.json groupXX@DGX-IP:~/data/MyDataset/
```

- **The first command** uploads the photo folder and its 5 class sub-folders. It can take a few minutes.
- **The second command** puts the CVAT JSON inside that same folder. Skip it if the JSON is already in the folder.
- **Finding the path:** on macOS you can drag the folder from Finder into the terminal to paste its path.

**DGX:** check the upload:

```bash
ls ~/data/MyDataset
```

You should see your 5 class folders and `instances_default.json`.

---

## 3. Start the training container and check everything

**DGX:**

```bash
tmux new -s owl
~/train.sh
```

`tmux` keeps your work running if your laptop sleeps or disconnects, so always start it first.
`~/train.sh` starts your container. The prompt changes to `root@...:/workspace#`. Inside the
container you are "root", but only within your own files.

**Container:** check that everything is there:

```bash
nvidia-smi                                       # the GPU "NVIDIA GB10" is listed
ls /workspace/data/MyDataset                     # your 5 class folders + the JSON
ls /workspace/code                               # preprocess.py  train_owlvit.py  plot_results.py
python -c "import torch; print('GPU ready:', torch.cuda.is_available())"   # GPU ready: True
```

Your three DGX folders appear inside the container under `/workspace`:

| On the DGX | Inside the container | Use it for |
| --- | --- | --- |
| `~/data` | `/workspace/data` | photos, JSON, the processed dataset |
| `~/code` | `/workspace/code` | the Python scripts |
| `~/weights` | `/workspace/weights` | trained models and results |

> Only these three folders are kept. Anything else in the container is deleted when you leave it.

**tmux basics:**

- **Leave training running:** press `Ctrl-b`, then `d`. You can then close your laptop.
- **Come back later:** `ssh` in and run `tmux attach -t owl`.
- **Leave the container:** type `exit`.

---

## 4. Prepare the dataset

**Container:**

```bash
python /workspace/code/preprocess.py /workspace/data/MyDataset
```

This takes about a minute. The script:

1. **Matches photos:** links every image in the CVAT JSON to the right photo in your folder.
2. **Converts to PNG:** handles jpg, jpeg, png, heic, webp and more. It applies the photo's
   rotation the same way CVAT showed it to you.
3. **Resizes:** makes every image 768 × 768, OWL-ViT's input size, and **rescales every box** to
   match. The file paths in the JSON are updated too.
4. **Splits the data:** each class folder is split into **70% train, 15% validation (val) and 15% test**.
   With 125 photos per class that's 87 / 19 / 19, so every class appears in every split.

The result is in `/workspace/data/processed/`:

```
processed/
  train.json  val.json  test.json      the new COCO files
  images/train/<class>/*.png           the resized images
  images/val/<class>/*.png
  images/test/<class>/*.png
  preview/                             a few images with their boxes drawn
```

**Read the summary at the end.** Every class should have boxes in train, val and test. Fix anything
marked `WARNING` before training.

**Check the boxes by eye.** **Laptop:** download the previews:

```bash
scp -r groupXX@DGX-IP:~/data/processed/preview ./preview
```

Open the images. Each box should sit tightly around its object. If the boxes are shifted or
rotated, tell the instructor before you train.

You can run the script again at any time. It replaces the old `processed` folder. Its settings
(image size, split fractions) are at the top of `preprocess.py`; normally you don't need to change them.

---

## 5. Train OWL-ViT

### The settings you can change

Open the training script.

**Container:**

```bash
nano /workspace/code/train_owlvit.py
```

At the top is the `HYPERPARAMETERS` block. **Change only the settings in this block.** Everything
below it is fixed for this lab, so leave it as it is. In nano, press `Ctrl-O` then `Enter` to save,
and `Ctrl-X` to quit.

| Setting | Default | What it does |
| --- | --- | --- |
| `RUN_NAME` | `""` (automatic) | Name of the results folder, e.g. `"combo1_baseline"` |
| `MODEL_NAME` | `"google/owlvit-base-patch32"` | `"google/owlvit-base-patch16"` is slower and often more accurate |
| `PROMPT_TEMPLATE` | `"a photo of a {}"` | Text given to the model for each class; `{}` becomes the class name |
| `EPOCHS` | 20 | Passes over the training images |
| `BATCH_SIZE` | 8 | Images per update; lower it if you get "out of memory" |
| `LEARNING_RATE` | 1e-4 | Step size of each update |
| `TRAIN_VISION_BACKBONE` | False | True also trains the image encoder: slower, more GPU memory |
| `HORIZONTAL_FLIP_PROB`, `COLOR_JITTER` | 0.5, 0.2 | Data augmentation; 0 turns it off |

### Run training

**Container:**

```bash
python /workspace/code/train_owlvit.py
```

The script first measures the original OWL-ViT on your validation images, then prints one line per epoch:

```
epoch train loss  val loss val mAP50  mAP50-95        lr   time
    1     0.4671    0.2702     0.995     0.877  1.00e-04    12s  <- best, saved
```

- **Loss:** should go down.
- **mAP50 / mAP50-95:** detection accuracy on the validation images, from 0 to 1. Higher is better.
  mAP50 counts a box as correct if it overlaps the true box by at least 50%. mAP50-95 averages
  stricter overlaps too.
- **`<- best, saved`:** the model from this epoch is the best so far, and it's the one kept.
- **Stopping:** training stops early after 8 epochs without improvement. Press `Ctrl-C` to stop it
  yourself; the best model so far is still tested and kept.

At the end, the best model is tested once on the **test** images it has never seen. The script prints
the accuracy per class next to the original OWL-ViT's.

Each run is saved in `~/weights/<RUN_NAME>/`:

```
combo1_baseline/
  best/                  the trained model: config, weights, prompts.json, categories.json
  metrics/               everything recorded during training (for the lab report)
  test_predictions/      test images: green = your boxes, red = the model's boxes
  training_curves.png    a quick look at loss and accuracy per epoch
```

An existing run is never overwritten. If you reuse a `RUN_NAME`, the new run gets `_2` added to its name.

---

## 6. The three experiments (required)

Train **three times**, once for each combination below. Each combination changes **one** setting
compared with the baseline, so you can see the effect of that one change. Keep every other setting
at its default.

| Setting | Combination 1: baseline | Combination 2: smaller learning rate | Combination 3: train the image encoder |
| --- | --- | --- | --- |
| `RUN_NAME` | `"combo1_baseline"` | `"combo2_low_lr"` | `"combo3_finetune_encoder"` |
| `LEARNING_RATE` | `1e-4` | **`1e-5`** | `1e-4` |
| `TRAIN_VISION_BACKBONE` | `False` | `False` | **`True`** |
| All other settings | default | default | default |

Use exactly these `RUN_NAME`s; the lab report commands below expect them.

For each combination:

1. **Container:** run `nano /workspace/code/train_owlvit.py` and set the values from the table.
   For combination 3, set `LEARNING_RATE` back to `1e-4`.
2. Save and quit, then run `python /workspace/code/train_owlvit.py`.
3. Wait for `Done.` before starting the next combination.

| Combination | What it tests | What to look for |
| --- | --- | --- |
| 1: baseline | Only the detection heads learn (about 1.6 million parameters). The image and text encoders stay frozen. | The reference point for the other two. Compare with the original OWL-ViT. |
| 2: smaller learning rate | The same model with 10× smaller update steps. | How fast the losses fall, and the validation mAP after 20 epochs, compared with combination 1. |
| 3: train the image encoder | The image encoder (about 88 million parameters) learns as well as the heads. | Accuracy, the gap between training and validation loss (overfitting), time per epoch, and GPU memory. |

On a free GPU, combinations 1 and 2 take about 4 minutes each and combination 3
about 8 minutes. They take longer when other groups are training, so run them inside
`tmux` (step 3).

---

## 7. Download the trained model to your computer

**DGX:** list your runs:

```bash
ls ~/weights
```

**Laptop:** download the model of the run you want, for example `combo1_baseline`:

```bash
scp -r groupXX@DGX-IP:~/weights/combo1_baseline/best ./owlvit_combo1
```

The model folder is about 600 MB.

### Use the model on your computer

**Laptop:** install the libraries once:

```bash
pip install torch transformers==4.46.3 pillow
```

Save this as `detect.py` next to the `owlvit_combo1` folder, then run `python detect.py my_photo.jpg`:

```python
import json, sys
import torch
from PIL import Image, ImageOps
from transformers import OwlViTForObjectDetection, OwlViTProcessor

MODEL_DIR = "owlvit_combo1"
THRESHOLD = 0.3                       # minimum score to show a box

processor = OwlViTProcessor.from_pretrained(MODEL_DIR)
model = OwlViTForObjectDetection.from_pretrained(MODEL_DIR).eval()
prompts = json.load(open(f"{MODEL_DIR}/prompts.json"))          # the exact training prompts
names = [c["name"] for c in json.load(open(f"{MODEL_DIR}/categories.json"))]

image = ImageOps.exif_transpose(Image.open(sys.argv[1])).convert("RGB")
inputs = processor(text=[prompts], images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
result = processor.post_process_object_detection(
    outputs, threshold=THRESHOLD, target_sizes=torch.tensor([[image.height, image.width]]))[0]
for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
    print(f"{names[label]:<20} score {score:.2f}  box {[round(v) for v in box.tolist()]}")
```

Always give the model the prompts from `prompts.json`. They are the same text it was trained with.

---

## Common problems

| Problem | Fix |
| --- | --- |
| `Permission denied` when logging in | Check the password (`spark#-group#`, e.g. `spark1-group02`) and that you use your own DGX's address. |
| `scp: No such file or directory` | The path on your laptop is wrong. Drag the folder into the terminal to get the exact path. |
| `The container name "/groupXX-train" is already in use` | Your container is already running, probably in tmux. Run `tmux attach -t owl`. Or open a second shell in it with `docker exec -it groupXX-train bash`. |
| `No COCO JSON found` | Upload the CVAT JSON into your photo folder (step 2, second command). |
| `image(s) in the JSON were not found` | Upload the same photos you annotated in CVAT, with the same file names. |
| `out of memory` | The GPU is shared by all groups. Lower `BATCH_SIZE` (e.g. 4) and run again. |
| Training is slow | Other groups are training too. Run it in tmux, detach, and come back later. |
| You broke a script | **DGX**, outside the container: `cp /home/share/owlvit_lab/*.py ~/code/` restores fresh copies. |
| You lost your SSH connection | Log in again and run `tmux attach -t owl`. |
| A run name got `_2` added | You reused a `RUN_NAME`. Use the new folder name in the download commands, or delete the old run first: `rm -r ~/weights/combo1_baseline` (on the DGX). |

---

## Lab report

### What each run records

Every run saves its training record in `~/weights/<RUN_NAME>/metrics/`. These files are small, so
they're quick to download.

| File | What it contains |
| --- | --- |
| `epoch_history.csv` | One row per epoch. Training and validation loss: total, classification, L1 and GIoU. Validation mAP50, mAP50-95 and mAP75. Learning rate, time and GPU memory. Epoch 0 is the original OWL-ViT before training. |
| `step_history.csv` | One row per training step: the loss and its three parts, and the learning rate. |
| `val_per_class_history.csv` | Validation AP50 and AP50-95 for every class after every epoch. |
| `test_per_class.csv` | Final test AP per class, for your fine-tuned model and for the original OWL-ViT. |
| `test_pr_curve.csv` | Precision-recall curve on the test images for every class (IoU 0.5), for both models. |
| `run_summary.json` | Your settings, the dataset size (images and boxes per class and split), trainable parameters, best epoch, training time, peak GPU memory, and the final test scores. |

### Download the data to your computer

When all three combinations have finished, run these on your **laptop**:

```bash
mkdir lab_report
scp -r groupXX@DGX-IP:~/weights/combo1_baseline/metrics lab_report/combo1_baseline
scp -r groupXX@DGX-IP:~/weights/combo2_low_lr/metrics lab_report/combo2_low_lr
scp -r groupXX@DGX-IP:~/weights/combo3_finetune_encoder/metrics lab_report/combo3_finetune_encoder
scp -r groupXX@DGX-IP:~/weights/combo1_baseline/test_predictions lab_report/predictions_combo1
scp -r groupXX@DGX-IP:~/weights/combo2_low_lr/test_predictions lab_report/predictions_combo2
scp -r groupXX@DGX-IP:~/weights/combo3_finetune_encoder/test_predictions lab_report/predictions_combo3
scp -r groupXX@DGX-IP:~/data/processed/preview lab_report/dataset_preview
scp groupXX@DGX-IP:~/code/plot_results.py .
```

You now have:

```
plot_results.py
lab_report/
  combo1_baseline/            metrics of combination 1 (the CSV and JSON files above)
  combo2_low_lr/              metrics of combination 2
  combo3_finetune_encoder/    metrics of combination 3
  predictions_combo1/ ...     test images with boxes, for each combination
  dataset_preview/            your images with the CVAT boxes
```

If you run the commands a second time, delete `lab_report` first. Otherwise `scp` puts the new copy
inside the old folder.

### Make the plots

**Laptop:**

```bash
pip install pandas matplotlib
python plot_results.py lab_report
```

This saves the figures and a results table in `lab_report/figures/`:

| File | Shows |
| --- | --- |
| `fig0_dataset_boxes.png` | Labeled boxes per class in train, val and test |
| `fig1_loss_per_epoch.png` | Training and validation loss per epoch, for all three combinations |
| `fig2_val_map_per_epoch.png` | Validation mAP50 and mAP50-95 per epoch (epoch 0 = original OWL-ViT) |
| `fig3_loss_per_step.png` | Training loss at every step, with a moving average |
| `fig4_loss_components.png` | Classification, L1 and GIoU loss per epoch |
| `fig5_learning_rate.png` | The learning rate schedule: warm-up, then cosine decay |
| `fig6_test_map.png` | Test mAP of each combination and of the original OWL-ViT |
| `fig7_test_ap_per_class.png` | Test AP50-95 per class |
| `fig8_pr_curves.png` | Test precision-recall curves per class |
| `fig9_val_ap_per_class.png` | Validation AP50-95 per class during training |
| `results_table.csv` | Settings, parameters, time, GPU memory and test mAP of each combination |

The script `plot_results.py` (also in `~/code` on the DGX):

```python
"""Make the lab-report figures from your OWL-ViT training runs.

Run on YOUR computer (not the DGX):
    pip install pandas matplotlib
    python plot_results.py lab_report

lab_report/ must contain one folder per run, each holding that run's metrics files
(epoch_history.csv, step_history.csv, val_per_class_history.csv, test_per_class.csv,
test_pr_curve.csv, run_summary.json). The figures and a results table are saved in
lab_report/figures/.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "lab_report")
out = root / "figures"
out.mkdir(parents=True, exist_ok=True)

# ---------- load every run ----------
runs = {}
for folder in sorted(p for p in root.iterdir() if (p / "epoch_history.csv").is_file()):
    run = {"epochs": pd.read_csv(folder / "epoch_history.csv"),
           "summary": json.loads((folder / "run_summary.json").read_text())}
    for name, file in (("steps", "step_history.csv"), ("val_class", "val_per_class_history.csv"),
                       ("test", "test_per_class.csv"), ("pr", "test_pr_curve.csv")):
        run[name] = pd.read_csv(folder / file) if (folder / file).is_file() else None
    runs[folder.name] = run
if not runs:
    sys.exit(f"No runs found in {root}. Each run folder needs epoch_history.csv (see the guide).")
print("Runs:", ", ".join(runs))
colors = dict(zip(runs, plt.rcParams["axes.prop_cycle"].by_key()["color"]))


def save(figure, name):
    figure.tight_layout()
    figure.savefig(out / name, dpi=150)
    plt.close(figure)
    print("saved", out / name)


# ---------- Fig 0: your dataset (boxes per class in each split) ----------
counts = next(iter(runs.values()))["summary"].get("boxes_per_class")
if counts:
    ax = pd.DataFrame(counts).plot.bar(figsize=(8, 4.5), rot=0, title="Labeled boxes per class and split")
    ax.set_ylabel("number of boxes")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    for container in ax.containers:
        ax.bar_label(container, fontsize=7)
    save(ax.figure, "fig0_dataset_boxes.png")


# ---------- Fig 1: train and validation loss per epoch ----------
figure, ax = plt.subplots(figsize=(7, 4.5))
for name, run in runs.items():
    e = run["epochs"][run["epochs"].epoch > 0]
    ax.plot(e.epoch, e.train_loss, color=colors[name], label=f"{name} train")
    ax.plot(e.epoch, e.val_loss, color=colors[name], linestyle="--", label=f"{name} val")
ax.set(xlabel="epoch", ylabel="loss", title="Training (solid) and validation (dashed) loss")
ax.legend(fontsize=8)
save(figure, "fig1_loss_per_epoch.png")

# ---------- Fig 2: validation mAP per epoch (epoch 0 = original OWL-ViT) ----------
figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, metric in zip(axes, ("val_mAP50", "val_mAP50-95")):
    for name, run in runs.items():
        e = run["epochs"]
        ax.plot(e.epoch, e[metric], marker="o", markersize=3, color=colors[name], label=name)
    ax.set(xlabel="epoch (0 = before training)", title=metric.replace("val_", "Validation "), ylim=(0, 1))
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mAP")
axes[0].legend(fontsize=8)
save(figure, "fig2_val_map_per_epoch.png")

# ---------- Fig 3: training loss at every step ----------
figure, ax = plt.subplots(figsize=(7, 4.5))
for name, run in runs.items():
    s = run["steps"]
    if s is None:
        continue
    ax.plot(s.global_step, s.loss, color=colors[name], alpha=0.2)
    ax.plot(s.global_step, s.loss.rolling(20, min_periods=1).mean(), color=colors[name], label=f"{name} (moving avg.)")
ax.set(xlabel="training step", ylabel="loss", title="Training loss per step")
ax.legend(fontsize=8)
save(figure, "fig3_loss_per_step.png")

# ---------- Fig 4: the three parts of the loss ----------
figure, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, part, title in zip(axes, ("class", "l1", "giou"), ("Classification (focal)", "Box L1", "Box GIoU")):
    for name, run in runs.items():
        e = run["epochs"][run["epochs"].epoch > 0]
        ax.plot(e.epoch, e[f"train_{part}_loss"], color=colors[name], label=f"{name} train")
        ax.plot(e.epoch, e[f"val_{part}_loss"], color=colors[name], linestyle="--", label=f"{name} val")
    ax.set(xlabel="epoch", title=title)
axes[0].legend(fontsize=7)
save(figure, "fig4_loss_components.png")

# ---------- Fig 5: learning rate schedule ----------
figure, ax = plt.subplots(figsize=(7, 4))
for name, run in runs.items():
    if run["steps"] is not None:
        ax.plot(run["steps"].global_step, run["steps"].learning_rate, color=colors[name], label=name)
ax.set(xlabel="training step", ylabel="learning rate", title="Learning rate schedule (warm-up + cosine)")
ax.legend(fontsize=8)
save(figure, "fig5_learning_rate.png")

# ---------- Fig 6: test mAP per run vs the original model ----------
rows = []
for name, run in runs.items():
    t = run["summary"].get("test")
    if t:
        rows.append({"model": name, "mAP50": t["mAP50"], "mAP50-95": t["mAP50-95"]})
first = next(iter(runs.values()))["summary"].get("original_model_test")
if first:
    rows.insert(0, {"model": "original OWL-ViT", "mAP50": first["mAP50"], "mAP50-95": first["mAP50-95"]})
if rows:
    table = pd.DataFrame(rows).set_index("model")
    ax = table.plot.bar(figsize=(9, 4.5), rot=15, ylim=(0, 1.05), title="Test mAP (untouched test images)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7)
    save(ax.figure, "fig6_test_map.png")

# ---------- Fig 7: test AP per class ----------
per_class = []
for name, run in runs.items():
    t = run["test"]
    if t is None:
        continue
    for _, r in t[t["class"] != "ALL"].iterrows():
        label = name if r["model"] == "fine-tuned" else "original OWL-ViT"
        per_class.append({"class": r["class"], "model": label, "AP50-95": r["AP50-95"]})
if per_class:
    pivot = (pd.DataFrame(per_class).drop_duplicates(["class", "model"])
             .pivot(index="class", columns="model", values="AP50-95"))
    pivot = pivot[["original OWL-ViT"] + [c for c in pivot.columns if c != "original OWL-ViT"]]
    ax = pivot.plot.bar(figsize=(12, 4.5), rot=0, ylim=(0, 1), title="Test AP50-95 per class")
    ax.set_ylabel("AP50-95")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    save(ax.figure, "fig7_test_ap_per_class.png")

# ---------- Fig 8: precision-recall curves on the test images (IoU 0.5) ----------
with_pr = {n: r for n, r in runs.items() if r["pr"] is not None}
if with_pr:
    figure, axes = plt.subplots(1, len(with_pr), figsize=(5 * len(with_pr), 4.5), squeeze=False)
    for ax, (name, run) in zip(axes[0], with_pr.items()):
        pr = run["pr"][run["pr"]["model"] == "fine-tuned"]
        for cls, curve in pr.groupby("class"):
            ax.plot(curve.recall, curve.precision, label=cls)
        ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02), title=f"{name}\nPR curves, IoU 0.5")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    save(figure, "fig8_pr_curves.png")

# ---------- Fig 9: validation AP per class during training ----------
with_class = {n: r for n, r in runs.items() if r["val_class"] is not None}
if with_class:
    figure, axes = plt.subplots(1, len(with_class), figsize=(5 * len(with_class), 4.5), squeeze=False, sharey=True)
    for ax, (name, run) in zip(axes[0], with_class.items()):
        for cls, curve in run["val_class"].groupby("class"):
            ax.plot(curve.epoch, curve["AP50-95"], label=cls)
        ax.set(xlabel="epoch", title=f"{name}\nValidation AP50-95 per class", ylim=(0, 1))
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    save(figure, "fig9_val_ap_per_class.png")

# ---------- Results table ----------
table = []
for name, run in runs.items():
    s, h = run["summary"], run["summary"]["hyperparameters"]
    table.append({"run": name, "model": h["MODEL_NAME"].split("/")[-1], "learning_rate": h["LEARNING_RATE"],
                  "batch_size": h["BATCH_SIZE"], "train_vision_backbone": h["TRAIN_VISION_BACKBONE"],
                  "flip": h["HORIZONTAL_FLIP_PROB"], "color_jitter": h["COLOR_JITTER"],
                  "trainable_params": s["trainable_parameters"], "epochs_run": s.get("epochs_completed"),
                  "best_epoch": s.get("best_epoch"),
                  "train_minutes": round(s.get("training_seconds", float("nan")) / 60, 1),
                  "peak_gpu_gb": s.get("peak_gpu_memory_gb"),
                  "test_mAP50": round(s.get("test", {}).get("mAP50", float("nan")), 3),
                  "test_mAP50-95": round(s.get("test", {}).get("mAP50-95", float("nan")), 3)})
table = pd.DataFrame(table)
table.to_csv(out / "results_table.csv", index=False)
print("\n" + table.to_string(index=False))
print(f"\nsaved {out / 'results_table.csv'}")
```

### What to put in your lab report

1. **Title:** group number, members, and which DGX you used.
2. **Introduction:** what OWL-ViT is (an open-vocabulary detector that finds objects described by
   text), and the goal of this lab.
3. **Dataset**
   - **Classes and photos:** your 5 classes, and how you took the photos (camera, backgrounds,
     lighting, distances).
   - **Annotation:** how you labeled the photos in CVAT, and how many boxes there are per class (`fig0`).
   - **Preprocessing:** conversion to PNG, resizing to 768 × 768 with box rescaling, and the
     70 / 15 / 15 split. Give the images and boxes per split.
   - **Examples:** 2–3 images from `dataset_preview` showing your boxes.
4. **Method**
   - **The model:** OWL-ViT in a few sentences, covering the image encoder, the text encoder and
     prompts, and the class and box heads.
   - **What was trained:** what each combination trains, and how many trainable parameters it has
     (`results_table.csv`).
   - **The loss:** Hungarian matching, then focal classification loss plus L1 and GIoU box losses.
     One sentence each.
   - **Settings:** a table of the three combinations.
5. **Results**
   - **Results table:** the original OWL-ViT and the three combinations. Include test mAP50,
     mAP50-95, best epoch, training time and peak GPU memory (`results_table.csv`, `fig6`).
   - **Figures:** `fig1` (loss), `fig2` (validation mAP), `fig7` (AP per class) and `fig8`
     (precision-recall), each with a caption.
   - **Example detections:** 3–4 images from the `predictions_*` folders, including at least one
     correct detection and one mistake.
6. **Discussion.** Answer these questions:
   - **Fine-tuning:** how much did fine-tuning improve on the original OWL-ViT? Which classes
     improved most and least, and why?
   - **Combination 2 vs 1:** what did the 10× smaller learning rate change? Look at how fast the
     loss fell and the final accuracy (`fig1`, `fig2`, `fig5`).
   - **Combination 3 vs 1:** what did training the image encoder change? Consider accuracy, the
     training/validation loss gap (overfitting?), time and memory.
   - **Deployment:** which combination would you deploy, and why?
   - **Mistakes:** what kinds of mistakes does the model make (missed objects, wrong class, loose
     boxes), and what might cause them?
   - **Next steps:** what would you try next to improve the model?
7. **Conclusion:** 3–5 sentences.
8. **Appendix:** the `run_summary.json` settings of each run, and any extra runs you tried.

# OWL-ViT Lab on the DGX Spark: Getting Started

You will upload your CVAT dataset to the DGX, prepare it, fine-tune OWL-ViT on the DGX's GPU,
and download the trained model to your own computer.

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
ls /workspace/code                               # preprocess.py  train_owlvit.py
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

### Choose the hyperparameters

All the settings you can experiment with are at the top of the training script.

**Container:**

```bash
nano /workspace/code/train_owlvit.py
```

Edit the values in the `HYPERPARAMETERS` block. Press `Ctrl-O` then `Enter` to save, and `Ctrl-X` to quit.

| Setting | Default | What to try |
| --- | --- | --- |
| `EPOCHS` | 20 | more if accuracy is still rising at the end |
| `LEARNING_RATE` | 1e-4 | 3e-5 to 3e-4 |
| `BATCH_SIZE` | 8 | lower it if you get "out of memory" |
| `TRAIN_VISION_BACKBONE` | False | True also trains the image encoder: slower, sometimes better |
| `HORIZONTAL_FLIP_PROB`, `COLOR_JITTER` | 0.5, 0.2 | data augmentation; 0 turns it off |
| `PROMPT_TEMPLATE` | `"a photo of a {}"` | e.g. `"{}"` or `"a {} on a table"` |
| `MODEL_NAME` | owlvit-base-patch32 | `"google/owlvit-base-patch16"`: slower, often more accurate |
| `RUN_NAME` | automatic | name the experiment, e.g. `"lr3e-4"` |

Change **one thing at a time** and compare the test results. Every run is saved in its own folder,
so nothing is overwritten.

### Run training

**Container:**

```bash
python /workspace/code/train_owlvit.py
```

The script first measures the original OWL-ViT on your validation images, then prints one line per epoch:

```
epoch train loss  val loss val mAP50  mAP50-95        lr   time
    1     0.4671    0.2702     0.995     0.877  1.00e-04    11s  <- best, saved
```

- **Loss:** should go down.
- **mAP50 / mAP50-95:** detection accuracy on the validation images, from 0 to 1. Higher is better.
  mAP50 counts a box as correct if it overlaps the true box by at least 50%. mAP50-95 averages
  stricter overlaps too.
- **`<- best, saved`:** the model from this epoch is the best so far, and it's the one kept.
- **Stopping:** training stops early if accuracy stops improving. Press `Ctrl-C` to stop it yourself;
  the best model so far is still tested and kept.

An epoch takes about 10 seconds when the GPU is free, and longer while other groups are training.
At the end, the best model is tested once on the **test** images it has never seen. The script prints
the accuracy per class next to the original OWL-ViT's.

Each run is saved in `~/weights/run1`, `~/weights/run2`, and so on:

```
run1/
  best/                  the trained model: config, weights, prompts.json, categories.json
  test_predictions/      test images: green = your boxes, red = the model's boxes
  training_curves.png    loss and accuracy per epoch
  history.csv            the numbers behind the plot
  test_metrics.json      final test accuracy, per class
  hyperparameters.json   the settings used for this run
```

---

## 6. Download the trained model to your computer

**DGX:** list your runs:

```bash
ls ~/weights
```

**Laptop:** download the model of the run you want, here `run1`:

```bash
scp -r groupXX@DGX-IP:~/weights/run1/best ./owlvit_run1
```

Also download the plots and prediction images:

```bash
scp -r groupXX@DGX-IP:~/weights/run1/test_predictions ./run1_test_predictions
scp groupXX@DGX-IP:~/weights/run1/training_curves.png .
```

### Use the model on your computer

**Laptop:** install the libraries once:

```bash
pip install torch transformers==4.46.3 pillow
```

Save this as `detect.py` next to the `owlvit_run1` folder, then run `python detect.py my_photo.jpg`:

```python
import json, sys
import torch
from PIL import Image, ImageOps
from transformers import OwlViTForObjectDetection, OwlViTProcessor

MODEL_DIR = "owlvit_run1"
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

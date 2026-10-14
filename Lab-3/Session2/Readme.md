# OWL-ViT Lab: Train on the DGX Spark, Run on the Jetson Orin Nano

This lab has two parts:

- **Part 1, training (steps 1–7):** upload your CVAT dataset to the DGX, prepare it, and fine-tune
  OWL-ViT on the DGX's GPU with three different settings.
- **Part 2, inference (steps 8–13):** export your best model, copy it to a Jetson Orin Nano, and
  run it live on a camera, with a web page where you type what to detect.

The **lab report** section at the end explains how to download your training data, make the
plots and write the report.

## Before you start

You need:

| What | Example |
| --- | --- |
| Your group account | `group02` |
| Your DGX number | `spark1`, `spark2` or `spark3` |
| The DGX address | shown in class (written as `DGX-IP` below) |
| Your photo folder, with one sub-folder per class | `MyDataset/` with 5 sub-folders of 125 photos each |
| The COCO JSON exported from CVAT | `instances_default.json` |
| **For Part 2:** the Jetson address and login | given in class (written as `JETSON-IP` and `JETSON-USER`) |
| **For Part 2:** a USB webcam plugged into the Jetson | optional: you can also test with photos |

To get the JSON from CVAT, export your task or project with the format **COCO 1.0**. Unzip the
download. The JSON is `annotations/instances_default.json`.

In every command below, replace:

- `groupXX` with your group account, for example `group02`
- `DGX-IP` with the address shown in class
- `MyDataset` with the name of your photo folder

Each command block says where to run it:

- **Laptop** means a terminal on your own computer: Terminal on macOS/Linux, PowerShell on Windows.
- **DGX** means your SSH session on the DGX.
- **Container** means inside the DGX training container, after step 3.
- **Jetson** means an SSH session on the Jetson (`ssh JETSON-USER@JETSON-IP`) or a terminal on its desktop.
- **Jetson container** means inside the Jetson container, after step 11.

All the scripts are already in your `~/code` folder on the DGX. You only edit one block of
settings in `train_owlvit.py`.

| Script in `~/code` | Runs on | What it does |
| --- | --- | --- |
| `preprocess.py` | DGX container | Prepares your dataset (step 4) |
| `train_owlvit.py` | DGX container | Fine-tunes OWL-ViT (steps 5 and 6) |
| `detect.py` | your laptop | Runs your model on a photo (step 7) |
| `export_jetson.py` | DGX container | Packs your model for the Jetson (step 8) |
| `jetson/` | Jetson container | The live web page and a one-photo test (steps 12 and 13) |
| `plot_results.py` | your laptop | Makes the lab report figures |

> Tip: avoid spaces in folder and file names. If a name has spaces, put the whole path in quotes.

---

# Part 1: Training on the DGX

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
ls /workspace/code                               # detect.py  export_jetson.py  jetson  plot_results.py  preprocess.py  train_owlvit.py
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

You can run the script again at any time. It replaces the old `processed` folder.

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
compared with the baseline (marked `# <- changed`), so you can see the effect of that one change.

For each combination:

1. **Container:** run `nano /workspace/code/train_owlvit.py`.
2. Make the `HYPERPARAMETERS` block look exactly like the block for that combination.
3. Save and quit, then run `python /workspace/code/train_owlvit.py`.
4. Wait for `Done.` before starting the next combination.

Use exactly these `RUN_NAME`s; the lab report commands expect them.

### Combination 1: baseline

Only the detection heads learn (about 1.6 million parameters). The image and text encoders stay
frozen. This is the reference point for the other two.

```python
RUN_NAME = "combo1_baseline"                # <- changed
MODEL_NAME = "google/owlvit-base-patch32"
PROMPT_TEMPLATE = "a photo of a {}"

EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

TRAIN_VISION_BACKBONE = False

HORIZONTAL_FLIP_PROB = 0.5
COLOR_JITTER = 0.2
```

### Combination 2: smaller learning rate

The same model as combination 1, with 10× smaller update steps.

```python
RUN_NAME = "combo2_low_lr"                  # <- changed
MODEL_NAME = "google/owlvit-base-patch32"
PROMPT_TEMPLATE = "a photo of a {}"

EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-5                        # <- changed

TRAIN_VISION_BACKBONE = False

HORIZONTAL_FLIP_PROB = 0.5
COLOR_JITTER = 0.2
```

### Combination 3: train the image encoder

The image encoder (about 88 million parameters) learns as well as the heads. Set the learning
rate **back to `1e-4`**.

```python
RUN_NAME = "combo3_finetune_encoder"        # <- changed
MODEL_NAME = "google/owlvit-base-patch32"
PROMPT_TEMPLATE = "a photo of a {}"

EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4                        # <- back to 1e-4

TRAIN_VISION_BACKBONE = True                # <- changed

HORIZONTAL_FLIP_PROB = 0.5
COLOR_JITTER = 0.2
```

### What to compare

| Combination | What to look for |
| --- | --- |
| 1: baseline | Accuracy compared with the original OWL-ViT. |
| 2: smaller learning rate | How fast the losses fall, and the validation mAP after 20 epochs, compared with combination 1. |
| 3: train the image encoder | Accuracy, the gap between training and validation loss (overfitting), time per epoch, and GPU memory. |

On a free GPU, combinations 1 and 2 take about 4 minutes each and combination 3
about 8 minutes. They take longer when other groups are training, so run them inside
`tmux` (step 3).

---

## 7. Download the trained model to your computer

**DGX:** list your runs:

```bash
ls ~/weights
```

**Laptop:** download the model of the run you want, for example `combo1_baseline`, and the
`detect.py` script:

```bash
scp -r groupXX@DGX-IP:~/weights/combo1_baseline/best ./owlvit_combo1
scp groupXX@DGX-IP:~/code/detect.py .
```

The model folder is about 600 MB.

**Laptop:** install the libraries once, then run the model on any photo:

```bash
pip install torch transformers==4.46.3 pillow
python detect.py owlvit_combo1 my_photo.jpg
```

It prints each object found with its score and box, and saves `my_photo_result.jpg` with the boxes
drawn. It always gives the model the prompts from `prompts.json`, the same text it was trained with.

---

## Common problems in Part 1

| Problem | Fix |
| --- | --- |
| `Permission denied` when logging in | Check the password (`spark#-group#`, e.g. `spark1-group02`) and that you use your own DGX's address. |
| `scp: No such file or directory` | The path on your laptop is wrong. Drag the folder into the terminal to get the exact path. |
| `The container name "/groupXX-train" is already in use` | Your container is already running, probably in tmux. Run `tmux attach -t owl`. Or open a second shell in it with `docker exec -it groupXX-train bash`. |
| `No COCO JSON found` | Upload the CVAT JSON into your photo folder (step 2, second command). |
| `image(s) in the JSON were not found` | Upload the same photos you annotated in CVAT, with the same file names. |
| `out of memory` | The GPU is shared by all groups. Lower `BATCH_SIZE` (e.g. 4) and run again. |
| Training is slow | Other groups are training too. Run it in tmux, detach, and come back later. |
| You broke a script | **DGX**, outside the container: `cp -r /home/share/owlvit_lab/*.py /home/share/owlvit_lab/jetson ~/code/` restores fresh copies. |
| You lost your SSH connection | Log in again and run `tmux attach -t owl`. |
| A run name got `_2` added | You reused a `RUN_NAME`. Use the new folder name in the download commands, or delete the old run first: `rm -r ~/weights/combo1_baseline` (on the DGX). |

---

# Part 2: Inference on the Jetson Orin Nano

You will run your best model on a Jetson Orin Nano in the `dustynv/pytorch:2.1-r36.2.0` container.
A web page, like the NanoOWL demo, shows the camera with boxes drawn on it. You type what to look
for in a box at the bottom:

```text
[screwdriver, multimeter, breadboard]
```

## 8. Export your model on the DGX

Pick the combination with the best test mAP50-95 (the end of its training output, or
`results_table.csv` from the lab report).

**DGX:** start the container if it isn't running: `~/train.sh`

**Container:**

```bash
python /workspace/code/export_jetson.py combo1_baseline
exit
```

Replace `combo1_baseline` with your run. Without a run name, the script lists the runs you can
export. It makes one file, `~/weights/owlvit_jetson_combo1_baseline.tar` (about 620 MB), with:

| Inside the file | What it is |
| --- | --- |
| `model/` | your fine-tuned model, with its class names and training prompts |
| `web_demo.py`, `index.html` | the live web page |
| `detect_image.py` | tests the model on one photo |
| `Dockerfile` | adds the needed libraries to `dustynv/pytorch:2.1-r36.2.0` |
| `test_images/` | 3 of your test photos per class, to try the model without a camera |

---

## 9. Copy the model to the Jetson

**Laptop:** download it from the DGX, then send it to the Jetson:

```bash
scp groupXX@DGX-IP:~/weights/owlvit_jetson_combo1_baseline.tar .
scp owlvit_jetson_combo1_baseline.tar JETSON-USER@JETSON-IP:~/
```

**Jetson:** unpack it:

```bash
cd ~
tar xf owlvit_jetson_combo1_baseline.tar
ls ~/owlvit_jetson          # Dockerfile  detect_image.py  index.html  model  test_images  web_demo.py
```

> Unpacking another export later replaces the model in `~/owlvit_jetson`. To keep both, rename
> the old folder first: `mv ~/owlvit_jetson ~/owlvit_jetson_old`.

---

## 10. Prepare the Jetson (once per Jetson)

Skip this step if `sudo docker images` on the Jetson already lists `owlvit-jetson`. If `sudo docker`
says `command not found`, Docker must be installed first (see "For the instructor" at the end).

**Jetson:** check the software version. The container needs JetPack 6 (L4T `R36`):

```bash
cat /etc/nv_tegra_release        # starts with: # R36 (release), ...
```

Download the PyTorch container and add the libraries. This needs internet and about 15 GB of
free disk space (check with `df -h ~`):

```bash
sudo docker pull dustynv/pytorch:2.1-r36.2.0
cd ~/owlvit_jetson
sudo docker build -t owlvit-jetson:1.0 .
```

The `Dockerfile` adds Pillow, OpenCV and the same `transformers` version (4.46.3) used for
training on the DGX. When the build finishes it prints `naming to docker.io/library/owlvit-jetson:1.0`.

Check that the container can use the Jetson GPU:

```bash
sudo docker run --rm --runtime nvidia owlvit-jetson:1.0 \
    python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see `True` and the GPU name, for example `True Orin`.

---

## 11. Start the Jetson container

**Jetson:**

```bash
cd ~/owlvit_jetson
sudo docker run --runtime nvidia -it --rm --network host \
    --device /dev/video0 \
    -v ~/owlvit_jetson:/workspace \
    owlvit-jetson:1.0
```

- **`--runtime nvidia`** gives the container the Jetson GPU.
- **`--network host`** makes the web page reachable from your laptop.
- **`--device /dev/video0`** gives it the USB camera. **No camera?** Remove that line.
- **`-v ~/owlvit_jetson:/workspace`** shares your folder: files the container writes (like
  `result.jpg`) appear in `~/owlvit_jetson` on the Jetson.

The prompt changes to `root@...:/workspace#`.

**Jetson container:** check:

```bash
ls                                    # Dockerfile  detect_image.py  index.html  model  test_images  web_demo.py
ls /dev/video*                        # /dev/video0  (only with a camera)
python3 -c "import torch; print('GPU ready:', torch.cuda.is_available())"     # GPU ready: True
```

---

## 12. Test the model on one photo

**Jetson container:**

```bash
ls test_images                                         # pick one, e.g. screwdriver_IMG_0042.jpg
python3 detect_image.py test_images/screwdriver_IMG_0042.jpg
```

It prints each object found, with its score and box, and saves `result.jpg` with the boxes drawn.
To look at it, open `~/owlvit_jetson/result.jpg` on the Jetson desktop, or copy it to your laptop:

```bash
scp JETSON-USER@JETSON-IP:~/owlvit_jetson/result.jpg .
```

`detect_image.py` uses the Hugging Face `pipeline` with your model folder and the exact training
prompts from `model/prompts.json`. It is a good first check that the model works on the Jetson.

---

## 13. Run the live web page

**Jetson container:**

```bash
python3 web_demo.py
```

Loading the model takes a little while. It's ready when it prints:

```text
Open http://<this-device-ip>:7860 in a browser (Ctrl-C to stop).
```

**Laptop:** open **`http://JETSON-IP:7860`** in a browser. On the Jetson's own desktop, use
`http://localhost:7860`. To find `JETSON-IP`, run `hostname -I` on the Jetson.

### Using the page

- **Camera view:** your camera, with a colored box and score for every object found.
- **Prompt box at the bottom:** type the objects to find, in brackets, separated by commas, then
  press Enter or **Detect**. It starts with your trained classes.
- **Trained classes:** click a class name to add it to the prompt.
- **threshold slider:** only boxes with a score above it are shown. Lower it to see more boxes,
  raise it to remove wrong ones.
- **Upload photo:** run the model on a JPG or PNG from your laptop or phone.
  **Back to camera** returns to the live view.
- **Top-right status line:** frames per second, time per frame in the model, the GPU, and the
  number of objects found.

### How prompts work

| You type | The model looks for |
| --- | --- |
| `[screwdriver, multimeter]` | your trained classes, with the exact training sentences: `a photo of a screwdriver`, `a photo of a multimeter` |
| `[Screwdriver]`, `[a screwdriver]`, `[the_screwdriver]` | the same as `screwdriver` (capitals, `a`/`an`/`the` and `_` don't matter) |
| `[pliers, coffee mug]` | words you did not train on, tried "zero-shot" as `a photo of a pliers` and `a photo of a coffee mug` |

Your model was fine-tuned only on your classes, so new words usually work worse than with the
original OWL-ViT. NanoOWL's nested prompts (`[a face [an eye]]`) and `(...)` classification
aren't supported here; use a flat list.

**To stop the page:** press `Ctrl-C`. Type `exit` to leave the container.

### No camera? Other inputs

Add `--source` to choose what the page shows:

```bash
python3 web_demo.py --source test_images      # loops over your test photos, 3 seconds each
python3 web_demo.py --source my_video.mp4     # a video file in ~/owlvit_jetson (loops)
python3 web_demo.py --source 1                # the second camera (/dev/video1)
```

Other options:

| Option | What it does |
| --- | --- |
| `--threshold 0.2` | starting value of the threshold slider (default 0.3) |
| `--resolution 1280x720` | camera resolution (default 640x480) |
| `--nms 0` | keep overlapping boxes of the same label (default 0.5 removes duplicates) |
| `--fp32` | run in float32 instead of float16: slower, for comparison |
| `--port 7861` | use another port, then open `http://JETSON-IP:7861` |

### Things to try

- **Trained vs. new words:** compare the scores for a trained class and an object you never labeled.
- **Threshold:** find the value where the wrong boxes disappear but the right ones stay.
- **Speed:** note the FPS and model time in the status line, then restart with `--fp32` and compare.
- **Real conditions:** try other lighting, distances and backgrounds than in your training photos.

---

## Common problems in Part 2

| Problem | Fix |
| --- | --- |
| `error gathering device information ... "/dev/video0"` | No camera is connected. Plug it in, or remove the `--device /dev/video0` line. |
| The page shows "Can't open camera" | Another program is using the camera, or it's a different number: try `--source 1`. |
| `GPU ready: False`, or `CUDA-enabled PyTorch is required` | You left out `--runtime nvidia` in step 11. |
| The page doesn't open on the laptop | Check `JETSON-IP` with `hostname -I` and that the laptop is on the same network. Try `http://localhost:7860` on the Jetson itself. |
| `is not an exported model folder` | Run the scripts from `/workspace` (step 11), and check that `ls model` shows `prompts.json`. |
| `Killed`, or the Jetson freezes | It ran out of memory. Close the browser and other programs on the Jetson and use the page from your laptop. |
| Very slow (1 to 2 FPS) | Check that the status line shows `Orin` and `float16`, not `CPU`. Run `sudo jetson_clocks` on the Jetson for full speed. |
| `docker build` fails while installing packages | The Jetson has no internet. Ask your instructor for the prepared image (see "For the instructor"). |
| Upload says "Can't read that file" | Upload a JPG or PNG. iPhone HEIC photos aren't supported. |

---

# Lab report

## What each run records

Every training run saves its record in `~/weights/<RUN_NAME>/metrics/`. These files are small, so
they're quick to download.

| File | What it contains |
| --- | --- |
| `epoch_history.csv` | One row per epoch. Training and validation loss: total, classification, L1 and GIoU. Validation mAP50, mAP50-95 and mAP75. Learning rate, time and GPU memory. Epoch 0 is the original OWL-ViT before training. |
| `step_history.csv` | One row per training step: the loss and its three parts, and the learning rate. |
| `val_per_class_history.csv` | Validation AP50 and AP50-95 for every class after every epoch. |
| `test_per_class.csv` | Final test AP per class, for your fine-tuned model and for the original OWL-ViT. |
| `test_pr_curve.csv` | Precision-recall curve on the test images for every class (IoU 0.5), for both models. |
| `run_summary.json` | Your settings, the dataset size (images and boxes per class and split), trainable parameters, best epoch, training time, peak GPU memory, and the final test scores. |

## Download the data to your computer

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

## Make the plots

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

## What to put in your lab report

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

---

# For the instructor

- **Installing Docker on the Jetson** (once, needs internet). JetPack 6 doesn't always include
  Docker. These are the steps from NVIDIA's Jetson AI Lab setup guide:

  ```bash
  sudo apt update && sudo apt install -y nvidia-container curl
  curl https://get.docker.com | sh && sudo systemctl --now enable docker
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

- **Jetson without internet:** build the image on any arm64 machine with Docker, for example a
  DGX Spark, and copy it over:

  ```bash
  sudo docker save owlvit-jetson:1.0 | gzip > owlvit-jetson.tar.gz      # on the build machine
  gunzip -c owlvit-jetson.tar.gz | sudo docker load                       # on the Jetson
  ```

- **Testing on a DGX Spark:** the Jetson container runs there too, but only on the CPU, because
  its PyTorch build only includes GPU code for Orin (sm_87). Use the same commands without
  `--runtime nvidia` and `--device`. `web_demo.py` switches to the CPU by itself; add `--cpu` to
  `detect_image.py`.
- **Where the files are:** every group's `~/code` holds all the scripts. Fresh copies are in
  `/home/share/owlvit_lab` on each DGX.

# OWL-ViT Lab: Train on the DGX Spark, Run on the Jetson Orin Nano

This lab has two parts:

- **Part 1, training on the DGX (steps 1–6):** upload your CVAT dataset from your laptop to the
  DGX, prepare it, and fine-tune OWL-ViT on the DGX's GPU with three different settings.
- **Part 2, on the Jetson Orin Nano (steps 7–13):** set up the Jetson, copy your trained models
  and training data from the DGX to it, make the lab report plots, and run your models live on a
  camera, with a web page where you type what to detect.

Your laptop is only used to upload your dataset and to log in to the DGX. Nothing is downloaded
to it: after training, everything goes to the Jetson. The **lab report** section at the end lists
what to put in your report.

## Before you start

You need:

| What | Example |
| --- | --- |
| Your group account | `group02` |
| Your DGX number | `spark1`, `spark2` or `spark3` |
| The DGX address | shown in class (written as `DGX-IP` below) |
| Your photo folder, with one sub-folder per class | `MyDataset/` with 5 sub-folders of 125 photos each |
| The COCO JSON file(s) exported from CVAT | `instances_default.json` (one, or one per CVAT task) |
| **For Part 2:** a Jetson Orin Nano with a screen, keyboard and mouse | one per group, shown in class |
| **For Part 2:** the Jetson login and password | given in class |
| **For Part 2:** internet on the Jetson and about 15 GB of free disk space | only for the one-time setup in step 8 |
| **For Part 2:** a USB webcam plugged into the Jetson | optional: you can also test with photos |

To get the JSON from CVAT, export your task or project with the format **COCO 1.0**. Unzip the
download. The JSON is `annotations/instances_default.json`.

In every command below, replace:

- `groupXX` with your group account, for example `group02`
- `DGX-IP` with the address shown in class
- `MyDataset` with the name of your photo folder

Each command block says where to run it:

- **Laptop** means a terminal on your own computer: Terminal on macOS/Linux, PowerShell on Windows.
  You use it only to upload your dataset (step 2) and to log in to the DGX (step 1).
- **DGX** means your SSH session on the DGX.
- **Container** means inside the DGX training container, after step 3.
- **Jetson terminal** means a terminal on the Jetson's desktop. You use it only to set up the
  Jetson (step 8) and to start the Jetson container (step 9).
- **Jetson container** means inside the PyTorch container on the Jetson, after step 9. Everything
  else on the Jetson runs here.

All the scripts are already in your `~/code` folder on the DGX. You only edit one block of
settings in `train_owlvit.py`.

| Script in `~/code` | Runs on | What it does |
| --- | --- | --- |
| `preprocess.py` | DGX container | Prepares your dataset (step 4) |
| `train_owlvit.py` | DGX container | Fine-tunes OWL-ViT (steps 5 and 6) |
| `export_jetson.py` | DGX container | Packs your models and training data for the Jetson (step 7) |
| `plot_results.py` | Jetson container | Makes the lab report figures (step 11) |
| `jetson/` | Jetson container | A one-photo test and the live web page (steps 12 and 13) |

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
- **Several JSON files** (for example one per CVAT task) are fine. Put each one next to the photos of
  that task, inside your photo folder, before you upload. Photos can be in sub-folders at any depth.
- **Finding the path:** on macOS you can drag the folder from Finder into the terminal to paste its path.

**DGX:** check the upload:

```bash
ls ~/data/MyDataset
```

You should see your class folders and your JSON file(s).

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
ls /workspace/data/MyDataset                     # your class folders + the JSON
ls /workspace/code                               # export_jetson.py  jetson  plot_results.py  preprocess.py  train_owlvit.py
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

1. **Finds everything:** every photo and every CVAT JSON file in your folder and all its
   sub-folders. Several JSON files are merged, and labels with the same name (ignoring capitals,
   spaces, `-` and `_`) become one class.
2. **Matches photos:** links every image in the JSON files to the right photo in your folder.
3. **Converts to PNG:** handles jpg, jpeg, png, heic, webp and more. It applies the photo's
   rotation the same way CVAT showed it to you.
4. **Resizes:** makes every image 768 × 768, OWL-ViT's input size, and **rescales every box** to
   match. The file paths in the JSON are updated too.
5. **Splits the data:** each class is split into **70% train, 15% validation (val) and 15% test**.
   With 125 photos per class that's 87 / 19 / 19, so every class appears in every split. A photo's
   class is the class its folder is named after, or else the label with the most boxes in it.

The result is in `/workspace/data/processed/`:

```
processed/
  train.json  val.json  test.json      the new COCO files
  images/train/<class>/*.png           the resized images
  images/val/<class>/*.png
  images/test/<class>/*.png
  preview/                             a few images with their boxes drawn
```

**Read the summary at the end.** Check the class list, and that every class has boxes in train,
val and test. Fix anything marked `WARNING` before training. Files it can't use (other JSON files,
broken photos, photos not in any JSON) are listed and skipped, not fatal.

The `preview` images (your photos with their boxes drawn) are copied to the Jetson with everything
else, where you can look at them (step 10).

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

Use exactly these `RUN_NAME`s, so your models and plots are named clearly.

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

## Common problems in Part 1

| Problem | Fix |
| --- | --- |
| `Permission denied` when logging in | Check the password (`spark#-group#`, e.g. `spark1-group02`) and that you use your own DGX's address. |
| `scp: No such file or directory` | The path on your laptop is wrong. Drag the folder into the terminal to get the exact path. |
| `The container name "/groupXX-train" is already in use` | Your container is already running, probably in tmux. Run `tmux attach -t owl`. Or open a second shell in it with `docker exec -it groupXX-train bash`. |
| `No COCO JSON found` | Upload the CVAT JSON into your photo folder (step 2, second command). Export it from CVAT as **COCO 1.0**. |
| `image(s) in the JSON were not found` | Upload the same photos you annotated in CVAT, with the same file names. |
| `match several photos and were skipped` | Two photos have the same name in different folders. Put each JSON next to its own photos (step 2), then run `preprocess.py` again. |
| Too many classes, e.g. `screwdriver` and `screw driver` | Labels are merged only when they differ in capitals, spaces, `-` or `_`. Fix the label names in CVAT and export again. |
| `out of memory` | The GPU is shared by all groups. Lower `BATCH_SIZE` (e.g. 4) and run again. |
| Training is slow | Other groups are training too. Run it in tmux, detach, and come back later. |
| You broke a script | **DGX**, outside the container: `cp -r /home/share/owlvit_lab/*.py /home/share/owlvit_lab/jetson ~/code/` restores fresh copies. |
| You lost your SSH connection | Log in again and run `tmux attach -t owl`. |
| A run name got `_2` added | You reused a `RUN_NAME`, so both runs are kept and exported. To avoid it, delete the old run first: `rm -r ~/weights/combo1_baseline` (on the DGX). |

---

# Part 2: On the Jetson Orin Nano

After training, everything goes from the DGX to the Jetson Orin Nano: your trained models,
everything recorded during training, the test predictions and your dataset previews. On the Jetson
you make the lab report plots and run your models live on a camera.

On the Jetson, everything runs **inside the PyTorch container** `owlvit-jetson:1.0`, which you
build from `dustynv/pytorch:2.1-r36.2.0` in step 8. The Jetson's own terminal is used only to
set up the Jetson (step 8) and to start the container (step 9).

## 7. Export everything on the DGX

Do this when all three combinations have finished.

**DGX:** start the container if it isn't running: `~/train.sh`

**Container:**

```bash
python /workspace/code/export_jetson.py
```

It packs every run in `~/weights` into one file, `~/weights/owlvit_jetson.tar` (about 2 GB for
three runs). To pack only some runs, name them, e.g.
`python /workspace/code/export_jetson.py combo1_baseline combo3_finetune_encoder`.
If you train more runs later, export again; the new file replaces the old one.

On the Jetson, the file unpacks to:

| Folder or file | What it is |
| --- | --- |
| `models/<run>/` | the trained model of each run |
| `lab_report/<run>/` | everything recorded during that run's training (see "Lab report") |
| `lab_report/predictions_<run>/` | test images: green = your boxes, red = the model's boxes |
| `lab_report/dataset_preview/` | a few of your images with the CVAT boxes |
| `test_images/` | 3 of your test photos per class |
| `web_demo.py`, `index.html` | the live camera web page |
| `detect_image.py` | tests a model on one photo |
| `plot_results.py` | makes the lab report figures |

---

## 8. Set up the Jetson (once per Jetson)

The Jetson container `owlvit-jetson:1.0` is the PyTorch container `dustynv/pytorch:2.1-r36.2.0`
with scp, Pillow, OpenCV, pandas, matplotlib and `transformers` 4.46.3 (the DGX version) added.
You build it once on each Jetson. It needs internet on the Jetson and about 15 GB of free disk
space. You can do this step while your models are training.

Log in on the Jetson's desktop and open a terminal.

### Check the Jetson

**Jetson terminal:**

```bash
cat /etc/nv_tegra_release                       # must start with "# R36" (JetPack 6)
sudo docker images owlvit-jetson                # is the image already built?
```

- **Password:** `sudo` asks for the Jetson password.
- **Already built:** if the second command lists `owlvit-jetson` with the tag `1.0` (for example
  because another group set up this Jetson), skip the rest of this step and go to step 9.
- **`docker: command not found`:** Docker isn't installed yet. Install it first (next section).

### Install Docker (only if it's missing)

These are the steps from NVIDIA's Jetson AI Lab setup guide.

**Jetson terminal:**

```bash
sudo apt update && sudo apt install -y nvidia-container curl
curl https://get.docker.com | sh && sudo systemctl --now enable docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Build the image

**Jetson terminal:**

```bash
sudo docker pull dustynv/pytorch:2.1-r36.2.0
mkdir -p ~/owlvit-image && cd ~/owlvit-image
scp groupXX@DGX-IP:/home/share/owlvit_lab/jetson/Dockerfile .
sudo docker build -t owlvit-jetson:1.0 .
```

- **`scp`** copies the image's recipe (the `Dockerfile`) from the DGX. Use your DGX password,
  `spark#-group#`, and type `yes` if you're asked "Are you sure you want to continue connecting?".
  The Jetson must be able to reach the DGX over SSH (port 22).
- **The download and build take a while.** Wait until the prompt comes back.

**Jetson terminal:** check that the image can use the GPU:

```bash
sudo docker run --rm --runtime nvidia owlvit-jetson:1.0 \
    python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It should print `True` and the GPU name, for example `True Orin`.

### Set the Jetson to full speed

**Jetson terminal:**

```bash
sudo jetson_clocks
```

This gives the best FPS in step 13. Run it again after the Jetson restarts.

### No internet on the Jetson?

Build the image on any arm64 machine with Docker, for example a DGX Spark, with the same
`docker pull` and `docker build` commands. Then save it to a file, copy the file to the Jetson,
and load it there:

```bash
sudo docker save owlvit-jetson:1.0 | gzip > owlvit-jetson.tar.gz      # on the build machine
gunzip -c owlvit-jetson.tar.gz | sudo docker load                       # on the Jetson
```

### Testing on a DGX Spark

The Jetson container also runs on a DGX Spark, but only on the CPU, because its PyTorch build only
includes GPU code for Orin (sm_87). Use the same commands without `--runtime nvidia` and
`--device`. `web_demo.py` switches to the CPU by itself; add `--cpu` to `detect_image.py`.

---

## 9. Start the container on the Jetson

**Jetson terminal:** this is the last command you run outside the container:

```bash
sudo docker run --runtime nvidia -it --rm --network host \
    --device /dev/video0 \
    -v ~/owlvit_groupXX:/workspace \
    owlvit-jetson:1.0
```

`sudo` asks for the Jetson password. The prompt then changes to `root@...:/workspace#`.

- **`--runtime nvidia`** gives the container the Jetson GPU.
- **`--network host`** lets the container reach the DGX and serve the web page.
- **`--device /dev/video0`** gives it the USB camera. **No camera?** Remove that line.
- **`-v ~/owlvit_groupXX:/workspace`** is your group's folder on the Jetson. It is created the
  first time. Everything in `/workspace` is kept there; anything else in the container is deleted
  when you leave it.
- **Several groups on one Jetson:** each group's files are in its own `~/owlvit_groupXX`. Two web
  pages at the same time need different ports (`--port 7861`, step 13).

**Jetson container:** check:

```bash
python3 -c "import torch; print('GPU ready:', torch.cuda.is_available())"     # GPU ready: True
ls /dev/video*                                                                # /dev/video0  (only with a camera)
```

To leave the container, type `exit`. To come back later, run the same `sudo docker run` command;
your files are still in `~/owlvit_groupXX`.

---

## 10. Copy your models and data from the DGX

**Jetson container:**

```bash
scp groupXX@DGX-IP:~/weights/owlvit_jetson.tar .
tar xf owlvit_jetson.tar
rm owlvit_jetson.tar
ls                    # detect_image.py  index.html  lab_report  models  plot_results.py  test_images  web_demo.py
ls models             # combo1_baseline  combo2_low_lr  combo3_finetune_encoder
```

- **"Are you sure you want to continue connecting?":** you'll see this once per container session. Type `yes`.
- **Password:** your DGX password, `spark#-group#`.
- **A newer export** replaces the files with the same names; the figures you made stay.

**Looking at images:** `/workspace` in the container is the folder `~/owlvit_groupXX` on the
Jetson. Open it in the Jetson's **Files** app to view any image. Start with
`lab_report/dataset_preview`: each box should sit tightly around its object. If boxes are shifted or
rotated, tell the instructor. Then look at the `lab_report/predictions_*` folders.

---

## 11. Make the lab report plots

**Jetson container:**

```bash
python3 plot_results.py lab_report
```

It prints the results table and saves the figures in `lab_report/figures/`. Open them in the
Files app.

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

---

## 12. Test a model on one photo

**Jetson container:**

```bash
ls test_images                                         # pick one, e.g. screwdriver_IMG_0042.jpg
python3 detect_image.py test_images/screwdriver_IMG_0042.jpg
```

- **Which model:** it uses your run with the best test mAP50-95 and prints its name. To test
  another one, add `--model`, e.g. `--model models/combo2_low_lr`.
- **Output:** it prints each object found, with its score and box, and saves `result.jpg` with
  the boxes drawn. Open `~/owlvit_groupXX/result.jpg` in the Files app.

`detect_image.py` uses the Hugging Face `pipeline` with your model folder and the exact training
prompts from `prompts.json`. It is a good first check that the model works on the Jetson.

---

## 13. Run the live web page

**Jetson container:**

```bash
python3 web_demo.py
```

It uses your best run. For another model: `python3 web_demo.py --model models/combo2_low_lr`.
Loading the model takes a little while. It's ready when it prints:

```text
Open http://<this-device-ip>:7860 in a browser (Ctrl-C to stop).
```

**Jetson desktop:** open **`http://localhost:7860`** in the web browser.

### Using the page

- **Camera view:** your camera, with a colored box and score for every object found.
- **Prompt box at the bottom:** type the objects to find, in brackets, separated by commas, then
  press Enter or **Detect**. It starts with your trained classes.
- **Trained classes:** click a class name to add it to the prompt.
- **threshold slider:** only boxes with a score above it are shown. Lower it to see more boxes,
  raise it to remove wrong ones.
- **Upload photo:** run the model on a JPG or PNG from the Jetson.
  **Back to camera** returns to the live view.
- **Top-right status line:** the model, frames per second, time per frame in the model, the GPU,
  and the number of objects found.

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

### No camera? Other inputs and options

```bash
python3 web_demo.py --source test_images      # loops over your test photos, 3 seconds each
python3 web_demo.py --source my_video.mp4     # a video file in ~/owlvit_groupXX (loops)
python3 web_demo.py --source 1                # the second camera (/dev/video1)
```

| Option | What it does |
| --- | --- |
| `--model models/<run>` | which of your models to use (default: the best test mAP50-95) |
| `--threshold 0.2` | starting value of the threshold slider (default 0.3) |
| `--resolution 1280x720` | camera resolution (default 640x480) |
| `--nms 0` | keep overlapping boxes of the same label (default 0.5 removes duplicates) |
| `--fp32` | run in float32 instead of float16: slower, for comparison |
| `--port 7861` | use another port, then open `http://localhost:7861` |

### Things to try

- **Your three models:** restart the page with each `--model` and compare the scores on the same objects.
- **Trained vs. new words:** compare the scores for a trained class and an object you never labeled.
- **Threshold:** find the value where the wrong boxes disappear but the right ones stay.
- **Speed:** note the FPS and time per frame in the status line, then restart with `--fp32` and compare.
- **Real conditions:** try other lighting, distances and backgrounds than in your training photos.

---

## Common problems in Part 2

| Problem | Fix |
| --- | --- |
| `Unable to find image 'owlvit-jetson:1.0'` | The image isn't built on this Jetson yet. Do step 8. |
| `docker: command not found` | Docker isn't installed. Install it in step 8. |
| `unknown or invalid runtime name: nvidia` | Docker isn't set up for the GPU. Run the last two commands of "Install Docker" in step 8. |
| `error gathering device information ... "/dev/video0"` | No camera is connected. Plug it in, or remove the `--device /dev/video0` line. |
| `GPU ready: False`, or `CUDA-enabled PyTorch is required` | You left out `--runtime nvidia` in step 9. |
| `scp`: `Connection timed out` or `No route to host` | The Jetson can't reach the DGX. Check `DGX-IP`, then ask your instructor. |
| `scp`: `Permission denied, please try again` | Use your DGX password, `spark#-group#`. |
| `scp`: `No such file or directory` | Run the export first (step 7). |
| `No space left on device` | Delete the copied file: `rm owlvit_jetson.tar` (in the container, in `/workspace`). |
| The page shows "Can't open camera" | Another program is using the camera, or it's a different number: try `--source 1`. |
| `is not an exported model folder` | Run the scripts from `/workspace`, and check `ls models`. |
| `Killed`, or the Jetson freezes | It ran out of memory. Close other programs and browser tabs, and run only one `web_demo.py` at a time. |
| Very slow (1 to 2 FPS) | Check that the status line shows `Orin` and `float16`, not `CPU`. Set the Jetson to full speed with `sudo jetson_clocks` (step 8). |
| Upload says "Can't read that file" | Upload a JPG or PNG. iPhone HEIC photos aren't supported. |

---

# Lab report

## What each run records

Every training run saves its record in `~/weights/<RUN_NAME>/metrics/` on the DGX. The export
(step 7) copies it to `lab_report/<RUN_NAME>/` on the Jetson, where `plot_results.py` (step 11)
turns it into figures.

| File | What it contains |
| --- | --- |
| `epoch_history.csv` | One row per epoch. Training and validation loss: total, classification, L1 and GIoU. Validation mAP50, mAP50-95 and mAP75. Learning rate, time and GPU memory. Epoch 0 is the original OWL-ViT before training. |
| `step_history.csv` | One row per training step: the loss and its three parts, and the learning rate. |
| `val_per_class_history.csv` | Validation AP50 and AP50-95 for every class after every epoch. |
| `test_per_class.csv` | Final test AP per class, for your fine-tuned model and for the original OWL-ViT. |
| `test_pr_curve.csv` | Precision-recall curve on the test images for every class (IoU 0.5), for both models. |
| `run_summary.json` | Your settings, the dataset size (images and boxes per class and split), trainable parameters, best epoch, training time, peak GPU memory, and the final test scores. |

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
   - **Examples:** 2–3 images from `lab_report/dataset_preview` showing your boxes.
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
   - **Example detections:** 3–4 images from the `lab_report/predictions_*` folders, including at
     least one correct detection and one mistake.
   - **On the Jetson:** the FPS and time per frame from the web page's status line for your best
     model, and 1–2 screenshots of the live page (press `PrtSc` on the Jetson).
6. **Discussion.** Answer these questions:
   - **Fine-tuning:** how much did fine-tuning improve on the original OWL-ViT? Which classes
     improved most and least, and why?
   - **Combination 2 vs 1:** what did the 10× smaller learning rate change? Look at how fast the
     loss fell and the final accuracy (`fig1`, `fig2`, `fig5`).
   - **Combination 3 vs 1:** what did training the image encoder change? Consider accuracy, the
     training/validation loss gap (overfitting?), time and memory.
   - **Deployment:** which combination would you deploy on the Jetson, and why? How did it do on
     the live camera compared with the test images?
   - **Mistakes:** what kinds of mistakes does the model make (missed objects, wrong class, loose
     boxes), and what might cause them?
   - **Next steps:** what would you try next to improve the model?
7. **Conclusion:** 3–5 sentences.
8. **Appendix:** the `run_summary.json` settings of each run, and any extra runs you tried.

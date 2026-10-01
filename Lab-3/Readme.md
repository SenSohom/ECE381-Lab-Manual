# Lab3-Part1 : Ultralytics YOLO on NVIDIA Jetson with Docker

This guide shows how to set up a **persistent Docker environment** on an NVIDIA Jetson for running Ultralytics YOLO with a USB camera.

Supported demo tasks include:

- Object detection
- Object tracking
- Pose estimation
- Instance segmentation

The setup uses:

- NVIDIA Jetson
- `dustynv/l4t-pytorch:r36.4.0`
- USB camera (`/dev/video0`)
- Ultralytics YOLO
- OpenCV GUI through X11
- A persistent host workspace

---

## 1. Create a Persistent Workspace

Create a folder on the Jetson host:

```bash
mkdir -p ~/yolo-workspace && cd ~/yolo-workspace
```

Create the Python demo file:

```bash
vim realtime_yolo.py
```

Press `i` to enter insert mode, then paste the following complete code into `realtime_yolo.py`:

```python
"""
Real-Time YOLO Demo

Tasks:
    1. Object Detection
    2. Pose Estimation
    3. Object Tracking
    4. Instance Segmentation

Students can also change:
    CONF -> confidence threshold
    IOU  -> NMS IoU threshold

Run:
    python3 realtime_yolo.py
"""

import cv2
from ultralytics import YOLO


# ============================================================
# Module 1: Configuration
# ============================================================

# Choose one task:
# "detect"  -> object detection
# "pose"    -> human pose estimation
# "track"   -> object tracking
# "segment" -> instance segmentation
#
# Example values:
# TASK = "detect"
# TASK = "pose"
# TASK = "track"
# TASK = "segment"
TASK = "pose"


# YOLO confidence threshold:
# Controls how confident YOLO must be before showing a detection.
#
# Lower value:
#   -> more detections
#   -> may include more false positives
#
# Higher value:
#   -> fewer detections
#   -> keeps only more confident detections
#
# Example values to try:
# CONF = 0.10   # very permissive, many detections
# CONF = 0.25   # common/default-style setting
# CONF = 0.50   # stricter
# CONF = 0.80   # very strict
CONF = 0.25


# YOLO IoU threshold for Non-Maximum Suppression (NMS):
# Controls how much overlap between bounding boxes is allowed.
#
# Lower value:
#   -> suppress overlapping boxes more aggressively
#
# Higher value:
#   -> allow more overlapping boxes to remain
#
# Example values to try:
# IOU = 0.20   # aggressive suppression
# IOU = 0.50   # medium suppression
# IOU = 0.70   # common/default-style setting
# IOU = 0.90   # keeps many overlapping boxes
IOU = 0.70


# Camera index:
# 0 usually means the first USB camera.
#
# Example:
# CAMERA_ID = 0   # /dev/video0
# CAMERA_ID = 1   # /dev/video1
CAMERA_ID = 0


# ============================================================
# Module 2: Load the YOLO Model
# ============================================================

if TASK == "detect":
    # General object detection model.
    model = YOLO("yolo11n.pt")

elif TASK == "pose":
    # Detect people and estimate body keypoints.
    model = YOLO("yolo11n-pose.pt")

elif TASK == "track":
    # Tracking uses a normal detection model.
    # YOLO adds a tracker on top of the detections.
    model = YOLO("yolo11n.pt")

elif TASK == "segment":
    # Detect objects and predict a pixel-level mask for each object.
    model = YOLO("yolo11n-seg.pt")

else:
    raise ValueError("TASK must be: detect, pose, track, or segment")


# ============================================================
# Module 3: Open the Camera
# ============================================================

camera = cv2.VideoCapture(CAMERA_ID)

if not camera.isOpened():
    raise RuntimeError("Cannot open camera")

print(f"Running YOLO task: {TASK}")
print(f"Confidence threshold: {CONF}")
print(f"IoU threshold: {IOU}")
print("Press 'q' to quit.")


# ============================================================
# Module 4: Real-Time Processing Loop
# ============================================================

while True:

    # --------------------------------------------------------
    # Step 1: Capture one frame from the camera
    # --------------------------------------------------------
    success, frame = camera.read()

    if not success:
        print("Failed to read camera frame")
        break

    # --------------------------------------------------------
    # Step 2: Run YOLO
    # --------------------------------------------------------

    if TASK == "track":
        # persist=True tells YOLO to remember objects between frames.
        # This allows an object to keep the same tracking ID.
        results = model.track(
            frame,
            persist=True,
            conf=CONF,
            iou=IOU,
            verbose=False
        )
    else:
        # Detection, pose, and segmentation all use normal inference.
        results = model(
            frame,
            conf=CONF,
            iou=IOU,
            verbose=False
        )

    # --------------------------------------------------------
    # Step 3: Draw the YOLO results
    # --------------------------------------------------------

    # results[0] corresponds to the current frame.
    #
    # plot() automatically draws:
    #   detection     -> bounding boxes + class names
    #   pose          -> skeleton + keypoints
    #   tracking      -> boxes + tracking IDs
    #   segmentation  -> masks + bounding boxes
    annotated_frame = results[0].plot()

    # --------------------------------------------------------
    # Step 4: Show the result
    # --------------------------------------------------------

    cv2.imshow(
        f"YOLO - {TASK}",
        annotated_frame
    )

    # --------------------------------------------------------
    # Step 5: Check keyboard input
    # --------------------------------------------------------

    # waitKey(1) waits approximately 1 ms.
    # Press "q" to stop the program.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# Module 5: Cleanup
# ============================================================

camera.release()
cv2.destroyAllWindows()
```

After pasting:

1. Press `Esc`.
2. Type `:wq`.
3. Press `Enter` to save and exit.

Your Python file is now stored permanently on the host at:

```text
~/yolo-workspace/realtime_yolo.py
```

Students can later edit `TASK`, `CONF`, and `IOU` near the top of the file to observe how YOLO behavior changes.

---

## 2. Allow Docker to Use the Desktop Display

The YOLO demo uses `cv2.imshow()`, so the Docker container needs access to the Jetson's X11 display.

Run this on the Jetson host:

```bash
xhost +local:docker
```

---

## 3. Create the Docker Container

Create the container **once**:

```bash
sudo docker run -it \
  --runtime nvidia \
  --device=/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/yolo-workspace:/workspace \
  --name yolo-jetson \
  dustynv/l4t-pytorch:r36.4.0 \
  bash
```

### What these options do

| Option | Purpose |
|---|---|
| `--runtime nvidia` | Gives the container access to the Jetson GPU |
| `--device=/dev/video0` | Gives the container access to the USB camera |
| `-e DISPLAY=$DISPLAY` | Passes the host display into Docker |
| `-v /tmp/.X11-unix:/tmp/.X11-unix` | Allows OpenCV/Qt windows to appear on the host desktop |
| `-v ~/yolo-workspace:/workspace` | Makes the Python code persistent |
| `--name yolo-jetson` | Gives the container a reusable name |

> This setup is for a **USB/V4L2 camera**. You do not need `/tmp/argus_socket`, which is mainly used for Jetson CSI/Argus cameras.

---

## 4. Install Python Dependencies

Run the following commands **inside the container**.

### Install a Jetson-compatible NumPy version

The Jetson PyTorch build may be incompatible with NumPy 2.x, so use NumPy 1.26.4:

```bash
python3 -m pip uninstall -y numpy

python3 -m pip install numpy==1.26.4 \
  -i https://pypi.org/simple
```

### Install Ultralytics

```bash
python3 -m pip install ultralytics \
  -i https://pypi.org/simple
```

### Install the YOLO tracking dependency

```bash
python3 -m pip install "lap>=0.5.12" \
  -i https://pypi.org/simple
```

Some packages may upgrade NumPy again during installation, so reinstall the correct version afterward:

```bash
python3 -m pip install --force-reinstall numpy==1.26.4 \
  -i https://pypi.org/simple
```

---

## 5. Verify the Environment

Run:

```bash
python3 - <<'PY'
import numpy
import torch
import lap
from ultralytics import YOLO

print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Ultralytics environment ready.")
PY
```

You should see:

- NumPy `1.26.4`
- PyTorch information
- `CUDA available: True`
- Your Jetson GPU name
- `Ultralytics environment ready.`

---

## 6. Run the YOLO Demo

Inside the container:

```bash
cd /workspace
python3 realtime_yolo.py
```

Press:

```text
Ctrl+C
```

to stop the program from the terminal.

---

## 7. Change the YOLO Task

Edit the program:

```bash
vim /workspace/realtime_yolo.py
```

Find:

```python
TASK = "detect"
```

Change it to one of:

```python
TASK = "detect"
TASK = "track"
TASK = "pose"
TASK = "segment"
```

In Vim:

1. Press `i` to edit.
2. Change `TASK`.
3. Press `Esc`.
4. Type `:wq`.
5. Press `Enter`.

Then rerun:

```bash
python3 /workspace/realtime_yolo.py
```

---

## 8. Exit the Container

When finished:

```bash
exit
```

This stops the shell session, but your container and installed Python packages remain available.

Your source code also remains on the host because `/workspace` is mapped to:

```text
~/yolo-workspace
```

---

## 9. Re-enter the Container Later

Do **not** run `docker run` again.

Start the existing container:

```bash
sudo docker start yolo-jetson
```

Enter it:

```bash
sudo docker exec -it yolo-jetson bash
```

Or use one command:

```bash
sudo docker start yolo-jetson && sudo docker exec -it yolo-jetson bash
```

Then run:

```bash
cd /workspace
python3 realtime_yolo.py
```

---

## 10. Docker Architecture

```text
Jetson Host
│
├── /dev/video0
│      └── USB Camera
│
├── X11 Display
│      └── /tmp/.X11-unix
│
├── ~/yolo-workspace/
│      └── realtime_yolo.py
│
└── Docker container: yolo-jetson
       │
       ├── dustynv/l4t-pytorch:r36.4.0
       ├── CUDA / PyTorch
       ├── NumPy 1.26.4
       ├── Ultralytics
       ├── lap
       ├── /dev/video0
       ├── DISPLAY
       └── /workspace
```

---

## Common Problems

### `No matching distribution found`

If pip shows errors such as:

```text
Failed to establish a new connection
Name or service not known
```

the problem may be the configured Jetson package index or DNS rather than the Python package itself.

Use standard PyPI explicitly:

```bash
python3 -m pip install PACKAGE_NAME \
  -i https://pypi.org/simple
```

### NumPy 2.x Error

If you see:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

fix it with:

```bash
python3 -m pip uninstall -y numpy
python3 -m pip install numpy==1.26.4 \
  -i https://pypi.org/simple
```

### Tracking Fails with `No module named 'lap'`

```bash
python3 -m pip install "lap>=0.5.12" \
  -i https://pypi.org/simple
```

### OpenCV Cannot Connect to Display

If you see:

```text
qt.qpa.xcb: could not connect to display
```

run on the host:

```bash
xhost +local:docker
```

and make sure the container was created with:

```bash
-e DISPLAY=$DISPLAY
-v /tmp/.X11-unix:/tmp/.X11-unix
```

### Container Name Already Exists

If Docker reports that `/yolo-jetson` already exists, reuse it:

```bash
sudo docker start yolo-jetson
sudo docker exec -it yolo-jetson bash
```

### Delete the Container

```bash
sudo docker rm -f yolo-jetson
```

This removes packages installed inside the container, but not files stored in `~/yolo-workspace`.

---

## Key Rule

Create the Docker container **once** with GPU, USB camera, X11, and workspace access.

After that, always reuse it with:

```bash
sudo docker start yolo-jetson
sudo docker exec -it yolo-jetson bash
```

Do not repeatedly use `docker run`, because `docker run` creates a new container.

---

# Lab3-Part2: NanoOWL Vision Transformer

## ⚠️ Important Setup Instructions

**Do not use Headless Mode.** It can cause compatibility problems with display forwarding and GUI applications.

### Before Starting the Lab

Connect all required peripherals to the Jetson Orin Nano:

- Power cable
- DisplayPort (DP) cable
- Ethernet cable
- Keyboard and mouse
- USB webcam

Set the Jetson to maximum power mode:

1. Click the **power icon** in the top-right corner of the desktop.
2. Select **MAXN SUPER**.
3. This ensures maximum performance for inference and training.

> These setup steps are important for OpenCV GUI windows, webcam access, and overall performance.

---

---

## Step 1: Run the NanoOWL Docker Container

First create a host output directory:

```bash
mkdir -p /home/$USER/nanoowl_outputs
```

Then start the container:

```bash
sudo docker run -it --rm \
  --runtime nvidia \
  --device /dev/video0 \
  --network host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -v /home/$USER/nanoowl_outputs:/outputs \
  --workdir /opt/nanoowl/examples/tree_demo \
  dustynv/nanoowl:r36.4.0 \
  /bin/bash
```

When you are inside the container, the terminal prompt should look similar to:

```text
root@ubuntu:/opt/nanoowl/examples/tree_demo#
```

The `/outputs` directory is shared with the Jetson host:

```text
Container: /outputs
Host:      /home/$USER/nanoowl_outputs
```

Files written to `/outputs` inside the container can therefore be accessed from the host.

---

## Step 2: Install the Required Module

Inside the NanoOWL container, install `aiohttp`:

```bash
pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  aiohttp
```

`aiohttp` is used by the web server that streams NanoOWL detection output to the browser.

> This NanoOWL container uses `--rm`, so it is deleted when you exit it. You therefore need to reinstall `aiohttp` each time you create a new NanoOWL container.

---

## Step 3: Set Up the Model Cache Directory

Run:

```bash
rm /root/.cache/clip
mkdir -p /root/.cache/clip
```

The first command removes an incorrectly created cache file, and the second creates the proper cache directory required for CLIP model files.

---

## Step 4: Run the NanoOWL Tree Demo

Start the detection server:

```bash
python3 tree_demo.py ../../data/owl_image_encoder_patch32.engine
```

When successful, the terminal should show something similar to:

```text
======== Running on http://0.0.0.0:7860 ========
```

Hold `Ctrl` and click the link, or open the address in the browser.

NanoOWL is optimized to run **OWL-ViT** in real time on NVIDIA Jetson Orin platforms using NVIDIA TensorRT.

It also provides a **tree detection** pipeline that combines OWL-ViT and CLIP, allowing nested detection and classification through text prompts.

Example prompts:

```text
[a face [a nose, an eye, a mouth]]
[a face (interested, yawning / bored)]
(indoors, outdoors)
```

Experiment with different prompts to see how the model responds.

> If the webcam feed does not appear, reload the browser.

---

## Step 5: Guided NanoOWL Experiments

Before writing the report, use the NanoOWL browser interface to perform the following short experiments. Keep the camera scene as unchanged as possible when comparing prompts.

### A. Establish a Baseline

Point the webcam toward a clear object or person and begin with a simple prompt such as:

```text
[a face]
```

Record:

- the detection score shown by NanoOWL,
- where the bounding box appears, and
- whether the detected region matches the concept described by the prompt.

This baseline will be used for comparison in the next experiments.

### B. Prompt Specificity Ladder

Keep the same scene and make the text prompt progressively more specific. For example:

```text
(an object)
(a person)
(a face)
(a face with glasses)
```

You may adapt the prompts to match your own scene. Observe whether the detection score, selected region, or detection result changes as more semantic information is added.

### C. Wrong-Prompt Test

Keep the same scene, but enter **two prompts describing objects that are not present**. For example, while only a person is visible:

```text
(a dog)
(a car)
```

Observe whether NanoOWL still returns a candidate region and how confident it is. This experiment is intended to show that an open-vocabulary model still produces similarity-based predictions and that a text prompt does not guarantee that the requested object is actually present.

### D. Tree-Prompt Refinement

Now use the tree detection interface. Start simple and refine the description of the scene over **three iterations**. For example:

```text
[a person]
[a person [a face]]
[a person [a face [an eye, a nose, a mouth]]]
```

Or, for a desk scene:

```text
[a person]
[a person [a desk, a chair]]
[a person [a desk [a laptop], a chair]]
```

For each iteration, observe what new information is detected and what is still missed. The purpose is to understand that a tree prompt can express **relationships and levels of semantic detail**, rather than only a single flat class label.

### E. Small Robustness / Failure Test

Choose **two** of the following conditions while keeping the text prompt fixed:

- partial occlusion,
- dim or very bright lighting,
- increased distance from the camera,
- approximately 45-degree object/person rotation.

Record how the detection score and bounding-box behavior change. You do not need to exhaustively test every condition.

> The purpose of these experiments is not to prove that NanoOWL is always better than YOLO. The purpose is to observe how **text-conditioned open-vocabulary detection behaves**, including its flexibility and its failure modes.

---

# Lab Report Requirements

The goal of the report is to demonstrate that you understand **both vision approaches and the motivation for moving from a fixed-class YOLO workflow to a text-conditioned Vision Transformer workflow**. Do not simply show that the programs run. Your report should use your own observations to explain what changed, why it changed, and when each approach is useful.

Keep the report concise. A well-organized **5–7 page report including tables and screenshots** is sufficient.

## Part 1 — Ultralytics YOLO Experiments

Use approximately the same camera scene whenever possible so that comparisons are meaningful.

### 1. Compare the Four YOLO Tasks

Run:

- `detect`
- `track`
- `pose`
- `segment`

For each task, include **one representative screenshot** and complete the following table.

| Task | Main Output | Additional Information Beyond Detection | One Possible Application |
|---|---|---|---|
| Detect | | | |
| Track | | | |
| Pose | | | |
| Segment | | | |

In a short paragraph, explain how the four tasks differ even though they all begin from visual input.

### 2. Confidence Threshold Experiment

Use:

```text
TASK = "detect"
IOU = 0.70
```

Test:

```text
CONF = 0.10
CONF = 0.50
CONF = 0.80
```

Use a scene containing multiple visible objects.

| CONF | Number of Detections | Weak / False Detections | Main Observation |
|---|---:|---|---|
| 0.10 | | | |
| 0.50 | | | |
| 0.80 | | | |

Explain the trade-off between a low and high confidence threshold in **2–3 sentences**.

### 3. IoU / NMS Experiment

Keep:

```text
CONF = 0.25
```

Test:

```text
IOU = 0.20
IOU = 0.70
IOU = 0.90
```

Try to create a scene containing partially overlapping objects or people. Include **one comparison figure or three screenshots**.

Briefly explain:

- what changed as IoU changed, and
- why changing IoU changes suppression of overlapping detections but does **not** teach YOLO a new object category.

---

## Part 2 — NanoOWL / OWL-ViT Experiments

This section should show that NanoOWL is not simply “another detector.” It uses **text prompts to specify the visual concept being searched for**, and its tree interface can express more detailed semantic structure. NanoOWL combines OWL-ViT and CLIP for nested detection/classification through text prompts.

### 1. Baseline + Prompt Specificity

Choose one object/person in the scene and record the baseline result. Then run a **four-level specificity ladder** similar to:

```text
(an object)
(a person)
(a face)
(a face with glasses)
```

Use prompts appropriate for your actual scene.

| Prompt | Detection Score | Box / Region Selected | Correct for the Requested Concept? |
|---|---:|---|---|
| General | | | |
| More specific | | | |
| Specific | | | |
| Most specific | | | |

Include **one representative screenshot** and answer:

> As the prompt became more specific, did the score or selected region change? What does this suggest about the relationship between the text description and the image representation?

Answer in **3–4 sentences**.

### 2. Wrong-Prompt / Uncertainty Experiment

Keep the same scene and test **two prompts describing objects that are not present**.

| Wrong Prompt | Returned a Box? | Score | Where Did the Model Focus? |
|---|---|---:|---|
| 1 | | | |
| 2 | | | |

In **2–3 sentences**, explain what this reveals about interpreting model confidence. A text prompt is a request to search for a concept; it is not proof that the concept exists in the scene.

### 3. Tree-Prompt Refinement

Create a hierarchical prompt and refine it through **three iterations**.

| Iteration | Tree Prompt | What Was Detected | What Was Missed |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

Include a screenshot of the final tree result. In **3–4 sentences**, explain why a nested tree description provides a different kind of interaction from giving YOLO a fixed class label.

### 4. Small Robustness / Failure Experiment

Choose **two** conditions from the guided experiment:

- occlusion,
- lighting change,
- increased distance,
- rotation.

Keep the NanoOWL prompt fixed.

| Condition | Baseline Score | Changed Score | Detection / Box Change |
|---|---:|---:|---|
| Test 1 | | | |
| Test 2 | | | |

For each test, give **one sentence** describing why you think the model's behavior changed.

---

## Part 3 — Transition Experiment: Why Move from YOLO to a ViT-Based Open-Vocabulary Model?

This is the most important comparison in the report. Use the **same physical scene** for YOLO and NanoOWL.

### Step 1 — Run YOLO

Use:

```text
TASK = "detect"
CONF = 0.25
IOU = 0.70
```

Record the labels produced by YOLO.

### Step 2 — Ask for a Concept YOLO Does Not Directly Provide

Choose **one visible object, subtype, attribute, or description** that YOLO does not return with the label you want. Examples could include a more specific object type, an attribute, or another visual concept present in the scene.

Do **not** retrain YOLO. Ask yourself:

- Can changing `CONF` or `IOU` make YOLO learn this new semantic category?
- Can you simply type the new category name into the pretrained YOLO script and make it search for that concept?

### Step 3 — Query the Same Concept with NanoOWL

Keep approximately the same scene and use a text prompt for the concept you selected. For example:

```text
(a blue notebook)
```

```text
(a power adapter)
```

```text
(a person wearing glasses)
```

Use a concept that is actually visible in your own scene.

Include **one YOLO screenshot and one NanoOWL screenshot**.

Complete this table:

| Comparison | YOLO | NanoOWL / OWL-ViT |
|---|---|---|
| How is the requested visual concept specified? | | |
| Can the requested concept be changed during this lab without retraining? | | |
| What happened for your selected concept? | | |
| What is one limitation you observed? | | |

### Required Interpretation

In **5–7 sentences**, explain the transition using your experiment. Your explanation must include all of the following ideas:

1. YOLO is effective for **task-specific detection over the classes represented by its trained detector**.
2. `CONF` and `IOU` modify decision/suppression behavior; they do **not create a new semantic class**.
3. NanoOWL/OWL-ViT allows the requested concept to be changed through **natural-language text prompts**.
4. This provides **open-vocabulary flexibility**, especially when the required visual concepts are not conveniently fixed in advance.
5. Open-vocabulary flexibility does **not** mean NanoOWL is automatically more accurate or more reliable; your wrong-prompt and robustness tests should make that clear.

Your report should demonstrate the following conceptual progression:

```text
YOLO
Fast, task-specific detection using learned classes
        ↓
Need to request a new or more specific visual concept
without collecting data and retraining a detector
        ↓
NanoOWL / OWL-ViT
Text-conditioned, open-vocabulary visual search
        ↓
Greater semantic flexibility, but still subject to
confidence errors and visual failure modes
```

---

## Part 4 — Short Conceptual Questions

Answer each question in **2–4 sentences** using evidence from your experiments.

1. What determines which object labels a pretrained YOLO detector can output?
2. Why did changing `CONF` and `IOU` change YOLO behavior without expanding what concepts it understands?
3. What role did the text prompt play in the NanoOWL experiments?
4. What did the wrong-prompt experiment teach you about treating a model score as certainty?
5. What did the tree prompt allow you to express that a single YOLO class label does not?
6. Give one application where you would prefer a fixed YOLO detector and one where an open-vocabulary approach would be useful. Explain why.

---

## Submission Checklist

Your report should contain:

- Brief objective/introduction (**one short paragraph**)
- YOLO four-task comparison
- YOLO confidence experiment
- YOLO IoU/NMS experiment
- NanoOWL specificity experiment
- NanoOWL wrong-prompt experiment
- NanoOWL three-stage tree-prompt refinement
- Two-condition NanoOWL robustness test
- **YOLO → NanoOWL transition experiment** using the same scene
- Short conceptual answers
- Clearly labeled screenshots/tables
- Brief conclusion (**one short paragraph**) stating what you learned about fixed-class detection versus text-conditioned open-vocabulary detection

> **Do not submit a step-by-step copy of the lab manual.** The report should focus on experimental results, observations, comparisons, and interpretation.

### Suggested Time Allocation

The required experiments are intentionally limited so that the practical work can be completed in approximately **2 hours**:

- YOLO four-task comparison: ~15 minutes
- YOLO confidence + IoU experiments: ~20 minutes
- NanoOWL baseline + specificity + wrong-prompt tests: ~20 minutes
- NanoOWL tree + two robustness tests: ~20 minutes
- YOLO → NanoOWL transition experiment: ~20 minutes
- Organize screenshots/tables and answer conceptual questions: ~25 minutes

---

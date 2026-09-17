# Lab 2: Real-Time Image Regression and Point Tracking on the Jetson Orin Nano

**Course:** ECE 381 — Applied Machine Learning

## Lab objectives

In this lab, you will:

- Run the same Docker-based lab environment used in Lab 1 on the Jetson Orin Nano.
- Use the prepared regression notebook to collect images and label target locations with `(x, y)` coordinates.
- Complete two three-point tracking tasks:
  1. Facial-feature tracking: left eye, right eye, and nose.
  2. DIY tracking: three distinct target points of your choice from the lab.
- Train and compare two transfer-learning models: ResNet-18 and ResNet-34.
- Compare three dataset sizes and three epoch settings.
- Use training loss and live coordinate predictions to recognize underfitting, overfitting, and reliable tracking.

You will complete **9 experiments for each task-model combination** and **36 experiments in total**.

```text
2 tracking tasks × 2 models × 3 dataset sizes × 3 epoch settings
= 36 total experiments
```

---

## 1. Set up the Jetson and open JupyterLab

### Step 1 — Connect and power the Jetson

1. Find your **kit number** on the box.
2. Connect the following devices to the Jetson:
   - Mouse
   - Keyboard
   - Webcam
3. Power on the Jetson and finish its initial setup.

> [!IMPORTANT]
> Connect the webcam **before** starting the Docker container. The notebook may produce errors if the webcam is not connected.

### Jetson Orin Nano password

The Jetson login password follows this format:

```text
machinelearning<Kit#>
```

Replace `<Kit#>` with the number written on your kit's box. Do not type the angle brackets or the `#` symbol.

For example, if your kit number is **#23**, the password is:

```text
machinelearning23
```

### Step 2 — Start the Docker container

Open a terminal on the Jetson and run:

```bash
./docker_dli_run.sh
```

This is the same Docker environment used in Lab 1. The first launch may download and install many files, so allow several minutes for it to finish.

You will work in a prepared JupyterLab notebook. Your responsibilities are to:

- Execute the notebook cells in the correct order.
- Configure each tracking task.
- Collect and annotate balanced coordinate data.
- Select the required model architecture.
- Change the dataset size and epoch settings for each experiment.
- Record and interpret the results.

### Step 3 — Open JupyterLab

When the container is ready, the terminal will display a JupyterLab URL. The URL shown in the original lab slides is:

```text
http://192.168.0.149:8888
```

Hold <kbd>Ctrl</kbd> and click the URL in the terminal, or copy and paste it into a web browser.

> [!NOTE]
> The Jetson's IP address may be different. Always use the complete URL printed in **your terminal**.

When JupyterLab asks for a password, enter:

```text
dlinano
```

This is the **JupyterLab password**. It is different from the Jetson login password.

---

## 2. Open the image-regression notebook

In the JupyterLab file browser:

1. Open the `regression` folder.
2. Open `regression_interactive.ipynb`.
3. Execute the code cells in order from top to bottom after making the required task and model changes described below.

The notebook cells perform the following tasks:

- **Camera setup:** Sets the image size to 224 × 224 and starts the USB camera. Shut down any other notebook kernel that is using the camera.
- **Task setup:** Defines the task, the three tracking categories, data augmentation, and dataset directory.
- **Data collection:** Displays a live image that saves an image and its target coordinate when you click a point.
- **Model definition:** Loads a pretrained ResNet model and replaces its final layer with six outputs—an `(x, y)` pair for each of the three target categories.
- **Live execution:** Predicts a target coordinate and draws the predicted point on the camera image.
- **Training and evaluation:** Trains the model using mean squared error (MSE) loss.
- **Interactive tool:** Combines the collection, training, testing, and model-saving controls.

> [!IMPORTANT]
> This is a **regression** lab. The notebook reports coordinate loss, not classification accuracy or class probability. A useful model places its predicted point close to the selected target on previously unseen camera views.

---

## 3. Configure the two tracking tasks

Each task must contain exactly three categories. The notebook creates two outputs per category, so three categories produce six model outputs:

```text
(x1, y1, x2, y2, x3, y3)
```

### 3.1 Task 1 — Facial-feature tracking

Use the following task settings:

```python
TASK = 'face'
CATEGORIES = ['left_eye', 'right_eye', 'nose']
DATASETS = ['A']
```

You must track:

1. Center of the subject's left eye
2. Center of the subject's right eye
3. Tip or center of the nose

Use the subject's anatomical left and right, not the left and right sides of the displayed image. If the webcam preview is mirrored, confirm the convention before collecting data and use it consistently throughout the lab.

### 3.2 Task 2 — DIY three-point tracking

Design a coordinate-regression task using **three distinct target points** found on an object or setup in the lab. The three targets should normally be visible together in the same camera frame.

Possible examples include:

- Three colored stickers placed on an object
- The tip, clip, and cap end of a pen
- Three marked corners or locations on a lab component
- Three distinct points on a handheld tool or printed shape

Do not reuse the facial landmarks from Task 1. Choose short, descriptive category names and document what each point represents. For example:

```python
TASK = 'pen_tracking'
CATEGORIES = ['tip', 'clip', 'cap_end']
DATASETS = ['A']
```

Your actual DIY task and category names may be different.

> [!IMPORTANT]
> After changing `TASK` or `CATEGORIES`, restart the notebook kernel and run the cells again from the beginning. This prevents a previous task's dataset, model, or callbacks from remaining active.

---

## 4. Collect coordinate data

Each saved entry consists of an image, a category, and the `(x, y)` location you clicked for that category.

For each task:

1. Select the first category from the **category** menu.
2. Place the subject or object in front of the camera.
3. Click the exact target location in the live camera image.
4. Confirm that the saved-image preview shows a marker at the intended location.
5. Repeat while changing position, distance, angle, lighting, and background.
6. Select the other categories and repeat the process.
7. Keep the number of labeled images balanced across all three categories.

For the facial-feature task, collect examples with reasonable changes in:

- Head position and rotation
- Distance from the camera
- Facial expression
- Lighting direction and brightness
- Background
- Subject, when permitted by the instructor

For the DIY task, vary the object's position and orientation while keeping all three selected targets visible whenever possible.

### Annotation-quality rules

- Click the same physical point for a category every time.
- Do not alternate between the center and edge of a feature.
- Do not save a sample if the selected target is hidden or outside the frame.
- Avoid collecting many nearly identical images.
- Keep all three categories balanced at each required dataset size.
- Review the saved-image preview after every click and recollect obvious annotation mistakes.

> [!NOTE]
> “Dataset size” in this lab means the number of labeled images **per target category**. With three categories, 10 images per category gives 30 labeled samples, 50 gives 150 samples, and 200 gives 600 samples.

---

## 5. Configure ResNet-18 and ResNet-34

You will train each task using both ResNet-18 and ResNet-34. In the notebook's **Model** cell, use one architecture at a time while keeping the output dimension unchanged.

The following model-selection pattern may be used:

```python
import torch
import torchvision

device = torch.device('cuda')
output_dim = 2 * len(dataset.categories)

MODEL_NAME = 'resnet18'  # change to 'resnet34' when required

if MODEL_NAME == 'resnet18':
    model = torchvision.models.resnet18(pretrained=True)
elif MODEL_NAME == 'resnet34':
    model = torchvision.models.resnet34(pretrained=True)
else:
    raise ValueError('MODEL_NAME must be resnet18 or resnet34')

model.fc = torch.nn.Linear(model.fc.in_features, output_dim)
model = model.to(device)
```

The final layer produces:

- Outputs 0–1: first category `(x, y)`
- Outputs 2–3: second category `(x, y)`
- Outputs 4–5: third category `(x, y)`

Use the same batch size, optimizer, preprocessing, augmentation, dataset, and other training settings for both architectures. Only the model architecture, dataset size, and required epoch count should differ.

> [!IMPORTANT]
> Begin every experiment from a fresh pretrained model initialization. Do not train one model for 5 epochs and then continue that same model to 25 or 50 epochs. Restart or reinitialize the model and optimizer before every experiment.

---

## 6. Train one experiment

For every required experiment:

1. Stop **live** mode before changing or training the model.
2. Confirm the correct task, categories, dataset, and model architecture.
3. Reinitialize the pretrained model and optimizer.
4. Enter the required number of epochs in the notebook widget.
5. Click **train**.
6. Wait for training to finish.
7. Record the final displayed MSE loss.
8. Test all three categories using live execution.
9. Save the required inference evidence.

The epoch widget counts down during training. Enter the required epoch value again before starting a new experiment.

Use a descriptive model filename if you save a checkpoint. A recommended format is:

```text
<Task>-<Model>-D<DatasetSize>-E<Epochs>.pth
```

Examples:

```text
Face-ResNet18-D50-E25.pth
DIY-ResNet34-D200-E50.pth
```

---

## 7. Test coordinate tracking in real time

After training:

1. Enable **live** mode.
2. Select the first category.
3. Confirm whether the predicted point follows the correct target.
4. Repeat for the other two categories.
5. Test positions, angles, distances, lighting conditions, and backgrounds that were not copied exactly from the training data.
6. Capture a three-panel figure or three clearly labeled screenshots showing the prediction for all three target categories.

When testing, consider both:

- **Precision:** How close is the predicted point to the correct target?
- **Stability:** Does the point remain on the target, or does it jump when the image changes slightly?

> [!WARNING]
> If the camera freezes, shut down the notebook kernel from the JupyterLab menu. Restart the kernel and run all cells again. Collected images stored in the mounted data directory remain available, but an unsaved trained model must be trained again.

---

## 8. Lab report deliverables

### 8.1 Tracking tasks

Complete both tasks:

1. **Facial-feature tracking**
   - `left_eye`
   - `right_eye`
   - `nose`
2. **DIY three-point tracking**
   - Three student-selected target points from an object or setup in the lab

Clearly list and define the three category names used for each task.

### 8.2 Required models

Train every required experiment with both:

1. **ResNet-18**
2. **ResNet-34**

Use the same collected images for both models so that the comparison is fair.

### 8.3 Required dataset sizes and epochs

For each task and model, train all combinations of:

- **Dataset sizes:** 10, 50, and 200 labeled images per target category
- **Epoch settings:** 5, 25, and 50 epochs

Build each task's dataset cumulatively:

1. Begin with 10 images per category.
2. Add images until you have 50 images per category.
3. Add images until you have 200 images per category.

At a given dataset size, use the same images for every epoch setting and for both model architectures.

| Experiment | Labeled images per category | Total labeled samples | Epochs |
|---:|---:|---:|---:|
| 1 | 10 | 30 | 5 |
| 2 | 10 | 30 | 25 |
| 3 | 10 | 30 | 50 |
| 4 | 50 | 150 | 5 |
| 5 | 50 | 150 | 25 |
| 6 | 50 | 150 | 50 |
| 7 | 200 | 600 | 5 |
| 8 | 200 | 600 | 25 |
| 9 | 200 | 600 | 50 |

Repeat this nine-experiment matrix for each task-model combination:

```text
Facial features + ResNet-18:  9 experiments
Facial features + ResNet-34:  9 experiments
DIY tracking + ResNet-18:     9 experiments
DIY tracking + ResNet-34:     9 experiments
------------------------------------------------
Total:                       36 experiments
```

### 8.4 Required evidence for every experiment

For **each of the 36 experiments**, include:

1. Tracking task and three target-category names.
2. Model architecture: ResNet-18 or ResNet-34.
3. Number of labeled images per category.
4. Total number of labeled samples.
5. Number of epochs.
6. Final displayed MSE loss.
7. One composite inference figure containing three labeled panels, or three labeled screenshots, showing:
   - The input visible in the camera feed.
   - The selected target category.
   - The predicted point overlaid on the image.
8. A short observation about prediction precision and stability.

Use a consistent experiment name so your results, checkpoints, and figures are easy to identify:

```text
<Task>-<Model>-D<DatasetSize>-E<Epochs>
```

Example:

```text
Face-ResNet34-D50-E25
```

### 8.5 Recommended results table

Create one copy of this table for each task-model combination, for a total of four tables:

| Experiment | Images per category | Epochs | Final MSE loss | Three-target inference figure | Precision/stability notes |
|---:|---:|---:|---:|---|---|
| 1 | 10 | 5 |  |  |  |
| 2 | 10 | 25 |  |  |  |
| 3 | 10 | 50 |  |  |  |
| 4 | 50 | 5 |  |  |  |
| 5 | 50 | 25 |  |  |  |
| 6 | 50 | 50 |  |  |  |
| 7 | 200 | 5 |  |  |  |
| 8 | 200 | 25 |  |  |  |
| 9 | 200 | 50 |  |  |  |

Place or reference the corresponding inference evidence for every row.

### 8.6 Results discussion

For each tracking task, discuss:

- How tracking changed when the dataset increased from 10 to 50 to 200 labeled images per category.
- How tracking changed when the epoch setting increased from 5 to 25 to 50.
- Whether lower training loss consistently produced better live tracking.
- Which combinations appeared to underfit.
- Which combinations appeared to overfit or become less reliable on new views.
- Which facial feature or DIY target was easiest and hardest to track, and why.
- How robust the predictions were to position, angle, distance, lighting, background, and partial occlusion.
- How ResNet-18 and ResNet-34 compared in loss, training time, prediction precision, and stability.
- Whether the additional depth of ResNet-34 produced a meaningful improvement for the collected dataset.
- Which dataset-size, epoch, and model combination produced the most reliable result.

Do not rely only on training loss. Use the live overlay behavior on new camera views when explaining model quality.

---

## Final submission checklist

- [ ] I completed facial-feature tracking for the left eye, right eye, and nose.
- [ ] I completed a DIY tracking task with three distinct target points from the lab.
- [ ] I clearly defined all target categories and used them consistently.
- [ ] I trained both ResNet-18 and ResNet-34 for both tasks.
- [ ] I completed all nine dataset-size and epoch combinations for every task-model pair.
- [ ] My report contains results for all 36 experiments.
- [ ] Every experiment includes the final MSE loss.
- [ ] Every experiment includes inference evidence for all three target categories.
- [ ] I compared the effects of dataset size and epoch count.
- [ ] I compared ResNet-18 with ResNet-34 using the same datasets.
- [ ] I discussed underfitting, overfitting, prediction precision, stability, and the best-performing configuration.
- [ ] I shut down the camera or notebook kernel when finished.

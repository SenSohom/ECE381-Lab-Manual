# Lab 1: Image Classification on the Jetson Orin Nano

**Course:** ECE 381 — Applied Machine Learning

## Lab objectives

In this lab, you will:

- Run the lab environment inside a Docker container on the Jetson Orin Nano.
- Use a prepared Jupyter notebook to collect image data, train a classifier, and test it in real time.
- Complete four image-classification tasks.
- Compare three dataset sizes and three epoch settings.
- Use the results to recognize underfitting, overfitting, and appropriate training.

You will complete **9 experiments for each classification task** and **36 experiments in total**.

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

This command starts a Docker container containing the Python modules required for the lab. The first launch may download and install many files, so allow several minutes for it to finish.

You will work in a JupyterLab environment. Most of the code has already been written for you. Your responsibilities are to:

- Execute the notebook cells in the correct order.
- Collect and organize image data.
- Change the required settings for each experiment.
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

You should now see the JupyterLab launcher.

---

## 2. Open the image-classification notebook

In the JupyterLab file browser:

1. Open the `classification` folder.
2. Open `classification_interactive.ipynb`.
3. Execute the code cells in order from top to bottom.

The notebook cells perform the following tasks:

- **Camera setup:** Sets the image size and starts the camera. Make sure the correct USB webcam is selected. Shut down any other active notebook kernels that are using the camera.
- **Task setup:** Defines the task and its categories. The category subdirectory names become the training labels.
- **Data collection:** Displays an interactive widget that captures and organizes images for each category.
- **Model definition:** Loads a pretrained ResNet-18 model, changes its final layer for the required categories, and moves the model to the Jetson GPU using CUDA.
- **Live execution:** Runs the model in the background so the camera feed and predictions can update in real time.
- **Training and evaluation:** Sets hyperparameters such as epochs, batch size, learning rate, and momentum, and loads images for training or evaluation.
- **Interactive tool:** Displays the controls used to collect data, train the model, and test predictions.

## 3. Collect the initial dataset

For each classification task:

1. Select the first category in the interactive tool.
2. Place an example of that category in front of the webcam.
3. Click **add** to capture and save the image.
4. Repeat while changing the object's position, angle, distance, and background.
5. Select the next category and repeat the process.
6. Keep the number of images balanced across all classes.

For example, when collecting the `thumbs_up` and `thumbs_down` dataset:

1. Select `thumbs_up`, vary the position and angle of your thumb, and click **add** for each image.
2. Select `thumbs_down` and repeat the same process.

Avoid capturing many nearly identical images. A diverse dataset usually generalizes better to new camera views.

## 4. Train the initial model

Complete an initial practice run before starting the report experiments:

1. Confirm that the correct dataset and categories are selected.
2. Set **epochs** to `10`.
3. Click **train**.
4. Wait for the data to load and the training process to finish.
5. Observe and record the final **loss** and **accuracy**.

The progress bar, loss, and accuracy update during training. Remember that the displayed accuracy is based on the current dataset and may not represent performance on unseen images.

## 5. Test the model in real time

After training is complete:

1. Enable **live** mode if it is not already active.
2. Present an object or gesture from one of the trained classes to the webcam.
3. Observe the predicted class.
4. Observe the probability associated with the prediction.
5. Test different positions, angles, distances, lighting conditions, and backgrounds.
6. Capture the required inference screenshot while the prediction and probability are visible.

> [!WARNING]
> If the camera freezes, shut down the notebook kernel from the JupyterLab menu. Restart the kernel and run all cells again. Your collected images are saved, but you must train the model again.

---

## 6. Lab report deliverables

### 6.1 Classification tasks

Complete all four classification tasks:

1. **Thumbs Up and Down**
2. **Emotions**
3. **Fingers**
4. **DIY multiclass classification**

For the DIY task, design your own image-classification dataset using objects available in the lab. Your DIY dataset must contain **at least three different classes**.

Clearly list the class names used for every task in your report.

### 6.2 Required experiments

Train each classification task using all combinations of the following settings:

- **Dataset sizes:** 10, 50, and 200 data points
- **Epoch settings:** 5, 25, and 50 epochs

To keep the classes balanced and the experiments comparable, collect the specified number of images **for each class**. You may build the datasets cumulatively:

1. Begin with 10 images per class.
2. Add images until you have 50 images per class.
3. Add images until you have 200 images per class.

Use the same images for all three epoch settings at a given dataset size. Only the epoch value should change.

> [!IMPORTANT]
> For a fair comparison, begin every experiment from the same fresh model initialization. Do not train for 5 epochs and then continue the same model for another 25 and 50 epochs. Restart or reinitialize the model before each experiment, then train it using the required epoch value.

| Experiment | Data points per class | Epochs |
|---:|---:|---:|
| 1 | 10 | 5 |
| 2 | 10 | 25 |
| 3 | 10 | 50 |
| 4 | 50 | 5 |
| 5 | 50 | 25 |
| 6 | 50 | 50 |
| 7 | 200 | 5 |
| 8 | 200 | 25 |
| 9 | 200 | 50 |

This produces:

```text
4 classification tasks × 9 experiments per task = 36 total experiments
```

### 6.3 Required evidence for every experiment

For **each of the 36 experiments**, include:

1. The classification task and class names.
2. The dataset size.
3. The number of epochs.
4. The trained model's final loss.
5. The trained model's final accuracy.
6. An inference screenshot showing:
   - The input visible in the camera feed.
   - The class predicted by the model.
   - The probability associated with the prediction.

Use a consistent experiment name so your results and screenshots are easy to identify. A recommended format is:

```text
<Task>-D<DatasetSize>-E<Epochs>
```

Example:

```text
Thumbs-D50-E25
```

### 6.4 Recommended results table

Create one copy of this table for each classification task:

| Experiment | Data points per class | Epochs | Final loss | Final accuracy | Inference figure |
|---:|---:|---:|---:|---:|---|
| 1 | 10 | 5 |  |  |  |
| 2 | 10 | 25 |  |  |  |
| 3 | 10 | 50 |  |  |  |
| 4 | 50 | 5 |  |  |  |
| 5 | 50 | 25 |  |  |  |
| 6 | 50 | 50 |  |  |  |
| 7 | 200 | 5 |  |  |  |
| 8 | 200 | 25 |  |  |  |
| 9 | 200 | 50 |  |  |  |

Place or reference the corresponding inference screenshot for every row.

### 6.5 Results discussion

For each classification task, discuss:

- How performance changed when the dataset increased from 10 to 50 to 200 data points per class.
- How performance changed when the epoch setting increased from 5 to 25 to 50.
- Which combinations appeared to underfit.
- Which combinations appeared to overfit.
- Which dataset-size and epoch combination produced the most reliable predictions.
- Any failure cases caused by object position, angle, distance, lighting, or background.

Do not rely only on training accuracy. Use your real-time inference tests and prediction probabilities when explaining model quality.

---

## Final submission checklist

- [ ] I completed all four classification tasks.
- [ ] My DIY task contains at least three classes.
- [ ] I completed all nine dataset-size and epoch combinations for every task.
- [ ] My report contains results for all 36 experiments.
- [ ] Every experiment includes final loss and final accuracy.
- [ ] Every experiment includes an inference screenshot showing the prediction and its probability.
- [ ] I compared the effects of dataset size and epoch count.
- [ ] I discussed underfitting, overfitting, and the best-performing training configuration.

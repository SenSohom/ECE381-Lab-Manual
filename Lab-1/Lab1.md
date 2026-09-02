# Lab 1(a): Image Classification on the Jetson

**Course:** ECE 381 — Applied Machine Learning

## Objective

In this lab, you will use a USB camera and a Jupyter notebook to train and test a two-class image classifier for:

- `thumbs_up`
- `thumbs_down`

The notebook uses a pretrained ResNet-18 model. Its final layer is configured for the two categories, and the model runs on the Jetson GPU using CUDA.

## Prerequisites

Before starting, make sure that:

- The Jetson has been set up in headless mode or connected directly to a monitor.
- A USB camera is connected to the Jetson.
- You can open a terminal in the Jetson's Ubuntu environment.
- You have an internet connection. The first container launch may download and install many files, so it can take several minutes.

## 1. Start the Docker container

Open a terminal and run:

```bash
./docker_dli_run.sh
```

Wait for the downloads and installation steps to finish. When JupyterLab is ready, the terminal will display a URL.

The URL shown in the lab slides is:

```text
http://192.168.0.149:8888
```

Hold <kbd>Ctrl</kbd> and click the URL in the terminal, or copy and paste it into a web browser. If your terminal displays a different IP address, use the URL printed in your terminal.

When prompted, enter this password:

```text
dlinano
```

You should now see the JupyterLab launcher.

## 2. Open the image-classification notebook

In the JupyterLab file browser:

1. Open the `classification` folder.
2. Open `classification_interactive.ipynb`.
3. Execute the code cells in order from top to bottom.

The notebook cells perform the following tasks:

- **Camera setup:** Sets the image size and starts the camera. Make sure the correct USB camera is selected. Shut down any other active notebook kernels that are using the camera.
- **Task setup:** Defines the task and its categories. The category subdirectory names become the training labels.
- **Data collection:** Displays an interactive widget that captures and organizes images for each category.
- **Model definition:** Loads a pretrained ResNet-18 model, modifies its final layer for two categories, and moves the model to the GPU.
- **Live execution:** Runs the model in the background so that the camera feed and predictions can update in real time.
- **Training and evaluation:** Sets training hyperparameters such as epochs, batch size, learning rate, and momentum, and loads images for training or evaluation.
- **Interactive tool:** Displays the controls used to collect data, train the model, and test predictions.

## 3. Collect the initial dataset

Use the interactive tool to collect examples of both gestures.

1. Select the `thumbs_up` category.
2. Hold a thumbs-up gesture in front of the camera.
3. Move your thumb through different positions and angles.
4. Click **add** to save each image.
5. Select the `thumbs_down` category.
6. Repeat the process for the thumbs-down gesture.

Keep the two categories approximately balanced. The slides do not specify an exact initial image count; their example shows roughly 30 images per category.

For better data, vary the gesture's angle and position rather than collecting many nearly identical images.

## 4. Train the initial model

1. Set **epochs** to `10`.
2. Click **train**.
3. Allow approximately 30 seconds for the data to load.
4. Watch the progress bar and observe the loss and accuracy after each epoch.

You may experiment with a different number of epochs. Remember that the displayed accuracy is based on the current dataset, not on unseen data.

## 5. Test the model in real time

After training is complete:

1. Enable **live** mode if it is not already active.
2. Hold a thumbs-up or thumbs-down gesture in front of the camera.
3. Observe the predicted category.
4. Observe the two sliders. They show the probabilities assigned to `thumbs_up` and `thumbs_down`.
5. Test the gesture at different positions, angles, and distances from the camera.

> **Camera troubleshooting:** If the camera freezes, shut down the notebook kernel from the JupyterLab menu. Restart the kernel and run all cells again. The collected images are saved, but you must train the model again.

## 6. Improve the model

Improve the dataset and retrain the model:

1. Change the background.
2. Collect more `thumbs_up` and `thumbs_down` images while varying the angle.
3. Test what happens when the gesture is near the corners and edges of the camera view.
4. Test what happens when the gesture is very close to or very far from the camera.
5. Using a variety of distances, collect **30 additional images for each category**.
6. Click **train** again.
7. Continue collecting, training, and testing until the model performs reliably.

## Completion checklist

- [ ] The Docker container starts successfully.
- [ ] JupyterLab opens in the browser.
- [ ] `classification/classification_interactive.ipynb` runs successfully.
- [ ] Images are collected for both `thumbs_up` and `thumbs_down`.
- [ ] The initial model is trained for 10 epochs.
- [ ] Real-time predictions are tested at multiple positions, angles, and distances.
- [ ] An additional 30 images are collected for each category.
- [ ] The model is retrained and tested again.

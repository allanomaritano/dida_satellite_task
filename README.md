# DIDA Satellite Task: Roof Detection with U-Net

**Allano Maritano** | allano@hotmail.it

## Project Overview

This project presents the implementation of a deep learning model using the U-Net architecture presented 
by Ronneberger et al. (2015) to perform pixel-wise segmentation on satellite images. Image segmenta-
tion is crucial for analyzing satellite imagery, as it enables the classification of each pixel, making
it suitable for identifying specific regions (like roofs).

## Environment Setup and libraries

```
conda create -n satellite_unet python=3.8
conda activate satellite_unet
```

```
conda install pytorch torchvision pillow numpy -c pytorch
```

## `unet.py`: U-Net Model Implementation

The U-Net architecture is optimized for pixel-wise classification, utilizing a contracting (encoder) and expansive (decoder) path to balance context and precise localization.

- **Skip connections** between encoder and decoder layers preserve spatial details, improving segmentation accuracy.
- **Double convolutional layers** with ReLU activation capture both high-level features and fine-grained details for accurate satellite image segmentation.

## Dataset Management: `satellite_dataset.py`

- Loads images from a directory, resizes them to a standard resolution, and normalizes pixel values.
- These preprocessing steps ensure consistency and efficiency during model training.

## Training Process: `main.py`

The steps for training the U-Net model are:

1. GPU Utilization: 
   - If available, leverage GPU resources for faster computation.
   
2. Data Preparation: 
   - Load training data using the **DataLoader**, which handles batching and shuffling of the dataset.
   
3. Loss Function: 
   - Use Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss) for binary classification tasks, 
     such as roof detection in satellite imagery.
   
4. Optimizer: 
   - Adam optimizer is employed to update the model parameters based on the computed gradients.
   
5. Training Loop: 
   - For each epoch (e.g., 50 epochs):
       a) Perform the forward pass: Input the images to the model and obtain predictions.
       b) Calculate the loss between predictions and true labels using the loss function.
       c) Perform backpropagation: Compute gradients and update model parameters using the optimizer.
       d) Track the training progress, monitor loss values, and adjust hyperparameters if needed.
   
6. Model Saving: 
   - After training is completed, save the model to a **.pth** file for future use.


## Test Dataset Management: `test_satellite_dataset.py`
Similar to the satellite_dataset.py but adapted for testing purposes:
- Loads and processes images for testing without labels.
- Returns both the image and its filename, allowing the model to save predictions with consistent naming.
- Designed for efficient access to individual images during the prediction phase.

## Prediction Process: `predictions.py`
The steps for making predictions with the trained U-Net model:

1. Model Loading: 
   - Load the pre-trained U-Net model from the **.pth** file and set it to evaluation mode 
     (disabling any training-specific components such as dropout).
   
2. Data Preparation: 
   - Use **TestSatelliteDataset** to load the test images, resizing them to 256x256 pixels and 
     transforming them into PyTorch tensors.
   
3. Prediction Loop: 
   - Process the test images one at a time (batch size = 1).
   - For each test image:
       a) Input the image to the trained model to generate a predicted segmentation mask.
       b) Apply a binary threshold to the output to convert the model's soft predictions into a 
          binary mask (roof vs. non-roof).
       c) Save the predicted mask in the output directory, ensuring it uses the same filename as the 
          original test image.
   
4. Output: 
   - The predicted masks are saved in the specified output directory, ensuring consistency between 
     input images and predicted segmentation results.
'''

## Future Improvements:
Future iterations of this project could include:
- Refining the U-Net architecture or exploring other segmentation models such as DeepLab or SegNet 
  for potentially improved accuracy.
- Incorporating a larger and more diverse dataset of satellite imagery to improve generalization 
  and robustness of the model.
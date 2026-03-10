# Optic Disc Detection in Retinal Fundus Images

<!-- Badges -->
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![TensorFlow/PyTorch](https://img.shields.io/badge/Deep_Learning-Supported-orange)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> A comprehensive comparative study of traditional computer vision techniques and deep learning models for the automated localization and segmentation of the optic disc.

<!-- Note: It's highly recommended to add a sample output image here -->
<!-- <p align="center"><img src="link_to_your_example_image.png" alt="Optic Disc Detection Example" width="600"/></p> -->

## 📖 About The Project

Accurately locating and segmenting the optic disc in retinal fundus images is a critical preliminary step in the development of automated diagnostic systems for ophthalmic conditions like Glaucoma, Diabetic Retinopathy, and Macular Degeneration. 

This repository serves as an experimental pipeline, implementing and comparing four distinct methodologies ranging from unsupervised clustering to advanced convolutional neural networks.

### Built With
* [Python](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [Scikit-Learn](https://scikit-learn.org/)
* [TensorFlow](https://www.tensorflow.org/) / [PyTorch](https://pytorch.org/) 


## 🗂️ Project Architecture & Methodologies

The project is modularized based on the specific detection algorithm employed:

*   **`CNN/`**: Deep learning approach utilizing Convolutional Neural Networks for high-accuracy feature extraction and pixel-wise segmentation.
*   **`DBSCAN/`**: Density-Based Spatial Clustering of Applications with Noise, used to cluster high-intensity pixel regions while filtering out noise and artifacts (like blood vessels).
*   **`K-Mean Clustering/`**: An unsupervised machine learning approach that segments the retinal image into distinct color/intensity clusters to isolate the optic disc.
*   **`Morphological + Thresholding/`**: Traditional computer vision pipeline applying adaptive thresholding, followed by morphological operations (erosion, dilation, opening/closing) to refine the optic disc boundaries.
*   **`original/`**: The raw, unprocessed dataset of retinal fundus images used for training, testing, and validation.
*   **`summary.docx`**: Complete project documentation, including mathematical background, detailed methodology, and analytical comparisons.

## 📊 Results & Evaluation

The performance of each methodology was evaluated using standard medical image segmentation metrics: Mean Dice Coefficient, Mean Intersection over Union (IoU / Jaccard), and Mean Pixel Accuracy.

| Methodology | Mean Dice | Mean IoU | Pixel Accuracy | Key Observations |
| :--- | :---: | :---: | :---: | :--- |
| **CNN (VGG / U-Net)** | 92.92% | 87.97% | 99.80% | Exceptional score for a supervised deep learning method. The U-Net encoder-decoder architecture captures localized context precisely, ignoring the vast retinal background. |
| **K-Means Clustering** | 85.41% | 77.98% | 99.34% | Exceptional unsupervised score. Forcing the final cluster into a mathematically perfect shape eliminated jagged boundaries. |
| **DBSCAN Clustering** | 84.34% | 76.40% | 99.25% | Strong baseline for an unsupervised density-based algorithm. Successfully grouped dense core pixels while discarding scattered retinal noise and exudates. |
| **Morphological + Thresholding**| 83.71% | 76.83% | 98.76% | Highly successful and robust for a "No-training" algorithm. The IoU score confirms masks are well-localized and consistent. |

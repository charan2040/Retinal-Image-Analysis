# Retinal Image Analysis for Diabetic Retinopathy Detection 

 A deep learning-based solution for early-stage detection of diabetic retinopathy using CNN and transfer learning techniques.
 Automates detection of diabetic retinopathy from fundus images  
 Leverages a Convolutional Neural Network (CNN) for end-to-end classification  
 Helps flag early signs of DR to assist ophthalmologists  


##  Overview

This project focuses on analyzing large-scale retinal datasets and detecting the severity of diabetic retinopathy through deep learning models. The pipeline is built using PyTorch Lightning and integrates preprocessing, model training, evaluation, and visualization.

## 🛠 Features

- Efficient data preprocessing (cropping, resizing, cleaning)
- CNN architecture fine-tuned with transfer learning
- Training metrics tracked via TensorBoard
- Inference via a simple Gradio web interface
- Supports real-time image prediction

## 📂 Dataset

- Kaggle Diabetic Retinopathy Detection: https://www.kaggle.com/c/diabetic-retinopathy-detection
- Publicly available fundus image datasets (e.g., Kaggle’s “APTOS 2019”)  
- Contains graded images across 5 severity levels (No DR → Proliferative DR)  
- Balanced / augmented to reduce class imbalance  


## 📦 Installation

```bash
git clone https://github.com/charan2040/Retinal-Image-Analysis
cd Retinal-Image-Analysis
pip install -r requirements.txt

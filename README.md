---
title: Detect_Brain_Tumours_From_MRI_scans_by_Ark_Abhyudaya
app_file: app.py
sdk: gradio
sdk_version: 5.23.3
---
%%writefile README.md
# Brain Tumor Detection using Xception

This project uses deep learning (Xception architecture) to detect brain tumors from MRI images.  
Datasets used: Sartaj, Figshare, BraTS.

## 📁 Dataset
- Sourced from Kaggle
- Preprocessed and augmented
- Binary classification (Tumor / No Tumor)

## 📊 Model
- Base Model: `Xception`
- Fine-tuned on tumor datasets
- Evaluation metrics: Accuracy, Precision, Recall

## 🚀 How to Run
Train the model in Google Colab and save it using:
```python
model.save('xception_brain_tumor.keras')

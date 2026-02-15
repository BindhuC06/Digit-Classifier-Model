# Handwritten Digit Classifier (MNIST)

## Overview
This project implements a **machine learning–based handwritten digit classifier** capable of recognizing digits from 0 to 9. The model is trained on the widely used **MNIST dataset** and demonstrates the ML workflow—from data preprocessing and model training to evaluation and prediction.

The goal of this project is to build a reliable baseline digit recognition system while following clean coding practices and proper version control.

---

## Problem Statement
Handwritten digit recognition is a classic computer vision and pattern recognition problem. Variations in handwriting styles, stroke thickness, and digit alignment make classification difficult. This project approaches the problem using supervised machine learning techniques.

---

## About the Dataset
- **Dataset**: MNIST
- **Training samples**: 60,000
- **Test samples**: 10,000
- **Image size**: 28 × 28 pixels (greyscale)
- **Classes**: Digits from 0 to 9

---

## Approach -Step - by - Step-
1. Loaded and explored the MNIST dataset
2. Normalization and preprocessing of data (Images)
3. Built a machine learning model for multiclass classification 
4. Trained the model using the preprocessed and labeled dataset
5. Evaluated performance on unseen test data set

---

## Tech Stack
- Python  
- NumPy  
- TensorFlow and keras

---

## How to work this out 
1. Clone the repository  
   ```bash
   git clone https://github.com/BindhuC06/Digit-Classifier-Model.git

2. Install dependencies
   ```bash
   pip install -r requirements.txt

3. Run the model
   ```bash
   python main.py

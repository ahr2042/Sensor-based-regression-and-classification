# Sensor-Based Regression & Fault Detection (Embedded ML Fundamentals)

## Overview

This project is an introductory machine learning exercise focused on **embedded-system-style sensor data**.  
The goal is to understand how classical machine learning models behave when applied to **physically meaningful signals**, rather than abstract datasets.

The project demonstrates:
- Data generation based on physical intuition
- Supervised learning for **regression** and **classification**
- Model evaluation and failure analysis
- Key pitfalls such as feature leakage and overfitting

The work is intentionally kept simple and interpretable to build a solid foundation before moving to deep learning or embedded inference.

---

## Problem Statement

Given simulated sensor measurements from an embedded system:
- Ambient temperature
- Supply voltage
- Load current
- System temperature

We aim to:
1. **Predict power consumption** (regression)
2. **Detect fault conditions** (classification)

---

## Dataset

The dataset is **synthetically generated** using known physical relationships and noise models.

### Features

| Name | Description | Units |
|-----|------------|-------|
| `ambient_temp` | Ambient temperature | °C |
| `voltage` | Supply voltage | V |
| `current` | Load current | A |
| `temperature` | System temperature | °C |

### Targets

| Name | Type | Description |
|-----|------|------------|
| `power` | Regression target | Power consumption (W) |
| `fault` | Classification target | Normal (0) / Fault (1) |

### Fault Conditions

A fault is injected when any of the following are true:
- Overcurrent
- Overtemperature
- Undervoltage

This allows complete control over **ground truth**, which is critical for learning and debugging ML behavior.

---

## Project Structure
```text
.
├── data_generation.py     # Synthetic sensor data generation
├── sensor_data.csv        # Generated dataset
├── train_regression.py    # Regression model training and evaluation
├── train_classification.py # Fault classification (next step)
└── README.md              # Project documentation
```

---

## Regression Models

The regression task predicts **power consumption** using the following inputs:
- `ambient_temp`
- `voltage`
- `current`

### Models Used
- **Linear Regression**
- **Random Forest Regressor**

### Evaluation Metric
- **Mean Squared Error (MSE)**  
  RMSE is used for intuitive interpretation.

### Key Observations
- Linear regression performs surprisingly well due to the near-linear nature of the underlying system.

---

## Key ML Concepts Demonstrated

- Train/test split
- Feature vs target separation
- Model interpretability
- Overfitting vs underfitting
- Feature leakage (intentional experiment)
- Bias–variance tradeoff

---

## Embedded Engineering Perspective

This project emphasizes **engineering judgment**:
- When ML is appropriate vs when physics-based models are better
- How correlated sensor signals affect ML behavior
- Why simpler models are often preferable in embedded systems
- How ML failure modes differ from traditional firmware bugs

---

## Requirements

- Python 3.8+
- NumPy
- Pandas
- scikit-learn
- Matplotlib (optional)

Install dependencies:
```bash
pip3 install numpy pandas scikit-learn matplotlib
```

---

## How to Run
Generate Dataset
```bash
python3 data_generation.py
```
Train Regression Models
```bash
python3 train_regression.py
```

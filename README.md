# Human_Activity_Recognition_research
# 🧠 Human Activity Recognition (HAR) Using Custom SVM Kernel & Feature Selection

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)]()
[![Machine Learning](https://img.shields.io/badge/ML-SVM%20%7C%20RFE%20%7C%20Ensembles-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20HAR-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Research%20Project-brightgreen.svg)]()

This repository contains the implementation and research work for  
**“Optimizing Human Activity Recognition Using a Support Vector Machine With a Custom Kernel and Feature Selection Techniques.”**

The project uses the **UCI HAR dataset** to recognize human activities using smartphone sensor signals.  
It integrates **Recursive Feature Elimination (RFE)**, a **custom Gaussian-like SVM kernel**, and ensemble learning methods to significantly improve HAR accuracy over baseline models.

---

## 🚀 Features

- Custom Gaussian-like **SVM kernel** for non-linear classification  
- **RFE** to reduce 561 features → **100 optimal features**  
- **SMOTE** for handling class imbalance  
- Ensemble models: Random Forest, XGBoost, Voting, Stacking  
- **SHAP interpretability** for feature analysis  
- Achieves **96% accuracy**, outperforming several state-of-the-art models  
(Results from the uploaded paper 📄) :contentReference[oaicite:0]{index=0}

---

## 📂 Dataset

- **Dataset:** UCI Human Activity Recognition (HAR)
- **Samples:** 10,299  
- **Features:** 561  
- **Activities:** Walking, Sitting, Standing, Laying, etc.  
- Sensor type: Accelerometer, Gyroscope  
(Described in page 3 of the paper) :contentReference[oaicite:1]{index=1}

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Handle missing values  
- Standardize features  
- Convert labels & remove subject bias  
- Apply **SMOTE** to balance activity classes  

### 2️⃣ Feature Selection (RFE)
- Logistic Regression as estimator  
- CV-based selection → **Best = 100 features**  
- Reduces dimensionality by **82%**
  ### 3️⃣ Custom SVM Kernel
Formula: K(x, y) = exp(-γ ||x - y||² )
- γ = 0.001, C = 1000  
- Captures complex non-linear motion patterns  

### 4️⃣ Ensemble Models
- Random Forest, XGBoost, LightGBM  
- Voting Classifier  
- Stacking Classifier (RF + SVM + ANN)

### 5️⃣ Explainability
- Global interpretability with **SHAP**  
- Highlights key HAR indicators like body jerk signals

---

## 📊 Results

**Test Accuracy (Table II):**

| Model | Accuracy |
|-------|----------|
| ⭐ Custom SVM Kernel | **95.9%** |
| Stacking | 95.8% |
| Voting | 95.6% |
| Logistic Regression | 95.5% |
| SVM (RBF) | 95.3% |
| XGBoost | 92.3% |
| Random Forest | 89.7% |

Visualizations (from the paper):  
- Model comparison bar chart (Fig. 3)  
- ROC curves (Fig. 4)  
- Confusion matrix (Fig. 5)  
- Precision-Recall curves (Fig. 6)  
- SHAP summary plot (Fig. 7)  
(All seen in pages 5–6) :contentReference[oaicite:2]{index=2}

---

## 📁 Project Structure

HAR-Custom-SVM/
│── src/
│ ├── preprocessing.py
│ ├── rfe_selection.py
│ ├── custom_kernel_svm.py
│ ├── ensembles.py
│ ├── shap_analysis.py
│── results/
│── dataset/
│── README.md
│── requirements.txt
│── research-paper.pdf

---

## 🛠 Installation

```bash
git clone https://github.com/your-username/HAR-Custom-SVM.git
cd HAR-Custom-SVM
pip install -r requirements.txt
python src/custom_kernel_svm.py
python src/rfe_selection.py
python src/shap_analysis.py




### 3️⃣ Custom SVM Kernel
Formula:

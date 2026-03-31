# Week 06 — Day 30: Logistic Regression on SUV Purchase Dataset

**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**
Git Link: https://github.com/shubhamrangdal2001-hash/Assignment_Day30.git

## Assignment Overview
End-to-end ML pipeline using Logistic Regression on the SUV Purchase Dataset from Kaggle.

## Dataset
[SUV Purchase Dataset — Kaggle](https://www.kaggle.com/datasets/bittupanchal/logistics-regression-on-suv-dataset)

Download `Social_Network_Ads.csv` and place it in the project root before running.

## Project Structure
```
├── part_a_pipeline.py       # Data loading, preprocessing, scaling, model training
├── part_b_evaluation.py     # Evaluation, decision boundary, test-size comparison
├── part_c_interview_code.py # Interview Q2: train-test split & scaling snippet
├── part_d_ai_task.py        # AI-augmented task with evaluation notes
├── Social_Network_Ads.csv   # Dataset (download from Kaggle)
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install pandas scikit-learn matplotlib
```

### 2. Run each part
```bash
python part_a_pipeline.py
python part_b_evaluation.py
python part_c_interview_code.py
python part_d_ai_task.py
```

## Key Concepts Covered
- Logistic Regression (binary classification)
- Label Encoding for categorical variables
- Train-Test Split (80/20, 75/25, 70/30)
- Standard Scaling
- Accuracy & Confusion Matrix
- Decision Boundary Visualization

## Results Summary

| Split | Accuracy |
|-------|----------|
| 80/20 | ~91%     |
| 75/25 | ~90%     |
| 70/30 | ~89%     |

## Interview Questions (Part C)

**Q1 — What is Logistic Regression?**  
Logistic Regression is a supervised learning algorithm used for **binary classification**. Despite the name "regression", it predicts a class (0 or 1) using the sigmoid function to output probabilities.

**Q3 — What is a Confusion Matrix?**  
A confusion matrix is a 2×2 table that shows the count of:
- **True Positives (TP)**: correctly predicted positives
- **True Negatives (TN)**: correctly predicted negatives
- **False Positives (FP)**: negatives wrongly predicted as positive
- **False Negatives (FN)**: positives wrongly predicted as negative

It helps evaluate not just accuracy, but also precision and recall.

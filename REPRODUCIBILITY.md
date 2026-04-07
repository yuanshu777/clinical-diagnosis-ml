# Reproducibility Guide

## Environment
Python 3.10+ recommended.

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Run Exploratory Analysis
```bash
python src/run_eda.py
```

## Run Baseline Models
```bash
python src/train_logistic_regression.py
python src/train_svm_classifier.py
```

## Run Random Forest Pipeline
```bash
python src/train_random_forest_submission.py
```

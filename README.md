# Clinical Diagnosis Machine Learning

## Project Summary
This project compares multiple ML approaches (Logistic Regression, SVM, Random Forest) to predict clinical diagnosis from structured patient features.

## Modeling Question
How accurately can diagnosis be predicted from demographic, physiological, and behavioral variables?

## Dataset
- `train.csv`: 1,504 labeled samples
- `test.csv`: 645 unlabeled samples
- Target: `Diagnosis`
- Excluded metadata columns during training: `PatientID`, `DoctorInCharge`

## Methodology
- Baseline models: Logistic Regression and SVM
- Ensemble model: Random Forest with feature selection and scaling
- Model tuning and stability checks across splits
- Metrics: classification report, confusion matrix, ROC-AUC, feature importance

## Repository Contents
- `314logistic.py`, `314svm.py`: baseline pipelines
- `Random_Forest_for_submittion.py`, `RF model after finetune.py`: RF pipelines
- `data processing.py`, `new model.py`: EDA/modeling utilities
- `test_predictions.csv`: generated predictions

## How to Reproduce
```bash
python 314logistic.py
python 314svm.py
python "Random_Forest_for_submittion.py"
```

Recommended environment: Python + `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

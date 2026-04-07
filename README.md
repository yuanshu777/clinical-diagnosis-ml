# Alzheimer's Disease Prediction with Machine Learning

## Project Summary
This project compares multiple machine learning approaches (Logistic Regression, SVM, Random Forest) to predict Alzheimer's diagnosis from structured clinical and behavioral features.

## Modeling Question
How accurately can Alzheimer's diagnosis be predicted from demographic, physiological, and cognitive indicators?

## Dataset
- `data/alzheimers_train.csv`: 1,504 labeled samples
- `data/alzheimers_test.csv`: 645 unlabeled samples
- Target: `Diagnosis`
- Excluded metadata columns during training: `PatientID`, `DoctorInCharge`

## Methodology
- Baseline models: Logistic Regression and SVM
- Ensemble model: Random Forest with feature selection and scaling
- Model tuning and stability checks across random splits
- Metrics: classification report, confusion matrix, ROC-AUC, feature importance

## Repository Contents
- `src/train_logistic_regression.py`, `src/train_svm_classifier.py`: baseline pipelines
- `src/train_random_forest_submission.py`, `src/train_random_forest_stability.py`: Random Forest pipelines
- `src/run_eda.py`: exploratory analysis and distribution diagnostics
- `outputs/alzheimers_test_predictions_legacy.csv`: archived legacy prediction output
- `archive/`: earlier scripts and course artifacts retained for traceability

## How to Reproduce
```bash
pip install -r requirements.txt
python src/run_eda.py
python src/train_logistic_regression.py
python src/train_svm_classifier.py
python src/train_random_forest_submission.py
```

Recommended environment: Python 3.10+ with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

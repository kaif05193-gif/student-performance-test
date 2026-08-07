# Student Performance Prediction

A machine learning project predicting students' final grades (G3) using
academic and behavioral data, built to practice the end-to-end ML workflow:
data loading, cleaning, modeling, evaluation, and interpretation.

## Dataset
[Student Performance Dataset](https://www.kaggle.com/datasets/devansodariya/student-performance-data)
via Kaggle — includes study time, absences, past failures, and grades (G1, G2, G3).

## Approach
- Cleaned and selected relevant features (study time, failures, absences, prior grades)
- Split data into training and test sets (80/20)
- Trained two models: Linear Regression and Random Forest Regressor
- Evaluated using MAE, RMSE, and R²

## Results

| Model             | MAE  | RMSE | R²   |
|-------------------|------|------|------|
| Linear Regression | 1.34 | 2.11 | 0.78 |
| Random Forest      | 1.05 | 1.62 | 0.87 |

## Key finding: data leakage
Feature importance analysis showed `G2` (previous term's grade) accounted for
~81% of the Random Forest's predictive power. This makes sense but is a form
of data leakage — G1/G2 are prior grades of the same students, so the model is
largely just learning that grades are consistent over time, rather than
learning what *causes* strong or weak performance.

**Next step:** re-run the model without G1/G2 to see how much predictive power
comes from behavioral factors (study time, absences, failures) alone — this
would give a more honest picture of what actually drives performance.

## Tools used
Python, pandas, NumPy, scikit-learn, Matplotlib

## How to run
Open the notebook in Google Colab (badge at the top of the file) or run
locally with the packages listed above installed.

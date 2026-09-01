# Heart Disease Classification

A course-guided Python machine learning project completed while following Daniel Bourke’s Zero to Mastery material. This notebook explores a heart disease dataset, compares three classification models, and evaluates a tuned Logistic Regression model.

## Course context and credit

This project follows [Daniel Bourke’s heart disease classification notebook](https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-3-structured-data-projects/end-to-end-heart-disease-classification.ipynb) from Zero to Mastery. The project idea and core workflow come from the course. This repository contains my implementation and notes from working through that material.

## Why this project?

The goal is to practice a complete machine learning workflow on tabular data: define a binary prediction task, inspect the data, compare models, tune parameters, and explain the results. Comparing several models provides a baseline before tuning. Evaluation goes beyond accuracy to examine different prediction errors. The course’s 95% accuracy target is a learning goal, not an achieved result or a clinical standard.

## Project overview

The notebook covers exploratory data analysis, visualization, model comparison, hyperparameter tuning, and evaluation. It uses pandas, NumPy, Matplotlib, seaborn, and scikit-learn.

Models compared:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest

## Dataset

`heart-disease.csv` contains 303 rows and 14 columns: 13 input features and the binary `target` label. The supplied data has no missing values. The notebook interprets target 1 as disease present and 0 as disease absent.

The original notebook references the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). The included CSV is the project-specific version supplied with this project; its preprocessing history and categorical encodings have not been independently verified. Use the included CSV to reproduce this analysis rather than substituting a raw UCI download.

## Approach

1. Explore class balance, feature distributions, and correlations.
2. Split the data into 80% training and 20% testing sets.
3. Compare baseline classification models.
4. Tune models using randomized search and grid search with five-fold cross-validation.
5. Evaluate the tuned Logistic Regression model using a confusion matrix, classification report, and ROC curve.
6. Explore Logistic Regression coefficients.

## Results

Results from the project verification run:

| Baseline model | Test accuracy |
| --- | ---: |
| Logistic Regression | 88.52% |
| KNN | 68.85% |
| Random Forest | 83.61% |

The grid-search Logistic Regression model achieved **88.52% test accuracy** on 61 test records. The original 95% accuracy target was not reached.

| Tuned Logistic Regression metric | Five-fold mean |
| --- | ---: |
| Accuracy | 84.47% |
| Precision | 82.08% |
| Recall | 92.12% |
| F1 | 86.73% |

These cross-validation metrics use the full dataset after parameter selection; they are not nested cross-validation estimates. The notebook also uses the test set repeatedly during model comparison and KNN tuning. A future evaluation should reserve an untouched test set and tune only within the training data.

This is an educational classification project, not a clinically validated diagnostic tool. Coefficient magnitudes depend on feature scales and should not be interpreted as a definitive feature-importance ranking.

## Run locally

1. Download and extract this repository, or clone it.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment (macOS/Linux):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, activate it with `.venv\Scripts\activate`.

4. Install dependencies and start JupyterLab:

   ```bash
   python -m pip install -r requirements.txt
   python -m jupyterlab
   ```

5. Open `end-to-end-heart-disease-classification.ipynb` and run the cells from top to bottom. Keep `heart-disease.csv` in the same folder. The hyperparameter searches can take several minutes.

The numerical libraries in `requirements.txt` are pinned to the verification environment. The baseline Logistic Regression model may emit a convergence warning; scaling features and increasing its iteration limit are future improvements.

## Files

| File | Purpose |
| --- | --- |
| `end-to-end-heart-disease-classification.ipynb` | Analysis, models, and saved outputs |
| `heart-disease.csv` | Project dataset |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes local environments and temporary files |

## Preparation notes

The project copy corrects two averaging errors in cross-validation recall and F1 and fixes the placement of scatter-plot labels. All Python analysis cells were executed sequentially with inline plotting adapted for a headless environment, and saved outputs were regenerated. The original model-selection approach is preserved.

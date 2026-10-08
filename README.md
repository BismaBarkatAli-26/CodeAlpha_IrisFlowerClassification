# 🌸 Iris Flower Classification — CodeAlpha Data Science Internship (Task 1)

## Project description
A supervised machine-learning project that classifies Iris flowers into three species — **setosa, versicolor and virginica** — using four measurements: sepal length, sepal width, petal length and petal width (cm).

## Objectives
- Clean and explore the Iris dataset with Pandas, Matplotlib and Seaborn
- Train and compare five classification algorithms with cross-validation
- Evaluate the best model on unseen test data (accuracy, precision, recall, F1, confusion matrix)
- Interpret which measurements matter most and save the trained model for reuse

## Dataset
- **Source:** the classic Iris dataset (R. A. Fisher, 1936), available built-in via `sklearn.datasets.load_iris()` and as `Iris.csv` in the CodeAlpha-linked / Kaggle "Iris Species" download.
- **Size:** 150 rows × 4 numeric features + 1 target (50 flowers per species).
- **Target:** `species` (3 classes).
- The notebook uses `Iris.csv` if present (in the project root or `data/`), otherwise the built-in copy — no download is required.

## Technologies
Python 3 · NumPy · Pandas · Matplotlib · Seaborn · Scikit-learn · Joblib · Jupyter / Google Colab

## Project structure
```
CodeAlpha_IrisFlowerClassification/
├── Iris_Flower_Classification.ipynb   # full, commented workflow
├── requirements.txt
├── README.md
├── data/        # optional Iris.csv
├── images/      # charts saved by the notebook
└── models/      # iris_best_model.joblib (created when you run the notebook)
```

## How to run
**Google Colab:** open colab.research.google.com → File → Upload notebook → select `Iris_Flower_Classification.ipynb` → Runtime → Run all.

**Locally:**
```bash
git clone https://github.com/<your-username>/CodeAlpha_IrisFlowerClassification.git
cd CodeAlpha_IrisFlowerClassification
pip install -r requirements.txt
jupyter notebook Iris_Flower_Classification.ipynb
```

## Methodology
1. Load data → standardise column names and labels
2. Check missing values; remove duplicate rows (avoids near-identical rows in both train and test)
3. EDA: class balance, pair plot, box plots, correlation heatmap
4. Stratified 80/20 train-test split (`random_state=42`)
5. Compare Logistic Regression, KNN, Decision Tree, SVM and Random Forest with 5-fold stratified cross-validation (scaler inside a `Pipeline` to prevent leakage)
6. Train the best CV model on the full training set; evaluate once on the test set
7. Confusion matrix, feature importance, decision-region plot; save model with Joblib

## Results
Results below were produced by running this notebook with the **scikit-learn built-in dataset** (1 duplicate removed → 149 rows; 119 train / 30 test). If you use `Iris.csv`, re-run and replace these numbers with yours.

| Model | 5-fold CV accuracy (mean ± std) |
|---|---|
| Logistic Regression | 0.9656 ± 0.0506 |
| SVM (RBF) | 0.9652 ± 0.0696 |
| KNN (k=5) | 0.9572 ± 0.0476 |
| Decision Tree | 0.9319 ± 0.0591 |
| Random Forest | 0.9319 ± 0.0591 |

**Selected model:** Logistic Regression — **Test accuracy 0.9333** (28/30 correct); macro precision, recall and F1 = 0.9333. Setosa: 10/10 correct; one versicolor and one virginica were confused with each other.

Screenshots: `images/05_model_comparison.png`, `images/06_confusion_matrix.png`, `images/07_feature_importance.png`

## Key findings
- Petal length and petal width are by far the most informative features (Random Forest importances ≈ 0.44 and 0.42).
- Setosa is perfectly separable; all errors occur between versicolor and virginica, which overlap in measurement space.
- The top models differ by less than one standard deviation in CV — on this dataset several simple models perform similarly well.

## Limitations
- Very small dataset: one test error changes accuracy by ~3.3 percentage points.
- Clean, textbook data — real-world classification (photos, noisy measurements) is harder.

## Future improvements
Repeated cross-validation, `GridSearchCV` hyper-parameter tuning, a Streamlit web app using the saved model.

## Author
<Your Name> — Data Science Intern, CodeAlpha · [LinkedIn](<your-link>)

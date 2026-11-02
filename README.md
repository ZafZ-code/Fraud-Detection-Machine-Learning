# Fraud Detection Using Machine Learning

A machine learning project focused on detecting fraudulent financial transactions using different classification algorithms and resampling techniques to address class imbalance.

## 📌 Overview

Fraud detection is a challenging classification problem due to the highly imbalanced nature of transaction datasets, where fraudulent transactions represent only a small proportion of the total observations.

The objective of this project is to try and develop a machine learning model capable of identifying fraudulent transactions while maintaining a balance between fraud detection and false positive rates.

Several resampling strategies and classification algorithms are explored to investigate their impact on model performance and identify the most effective approach.

## 🎯 Objectives

- Develop a reliable machine learning solution for identifying fraudulent transactions.
- Address class imbalance using different resampling strategies:
  - SMOTE (Synthetic Minority Over-sampling Technique).
  - Random Under-Sampling.
  - Combined SMOTE and Under-Sampling.
- Train and compare multiple classification algorithms.
- Evaluate model performance using metrics suitable for imbalanced classification.
- Optimize the best-performing model through hyperparameter tuning.
- Investigate ways to improve fraud detection while controlling false positive predictions.

## 📊 Dataset

The project uses the [Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset) available on Kaggle.

| Attribute              | Description                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------- |
| Number of observations | Approximately 6.3 million                                                               |
| Target variable        | `isFraud`                                                                               |
| Problem type           | Binary classification                                                                   |
| Main features          | `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` |

The target variable indicates whether a transaction is fraudulent or legitimate.

- `0`: Legitimate transaction(99.87%)
- `1`: Fraudulent transaction(0.13%)

## 🛠️ Technologies & Libraries

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| Python           | Main programming language              |
| Pandas           | Data manipulation and analysis         |
| NumPy            | Numerical computations                 |
| Matplotlib       | Data visualization                     |
| Seaborn          | Statistical visualization              |
| Scikit-learn     | Preprocessing, modeling and evaluation |
| Imbalanced-learn | Resampling techniques                  |
| XGBoost          | Gradient boosting classification       |

## ⚙️ Project Workflow

The project follows these main steps:

1. **Data Exploration:** Understand the dataset, its features and class distribution.
2. **Data Cleaning & Preprocessing:** Data inconsistencies.
3. **Exploratory Data Analysis (EDA):** Investigate feature distributions, relationships and fraud patterns.
4. **Feature Engineering:** Select and transform relevant features to improve model performance.
5. **Handling Class Imbalance:** Apply and compare SMOTE, Random Under-Sampling and their combination.
6. **Model Training:** Train different machine learning classification algorithms(Logistic Regression, Random Forest, XGBoost)
7. **Model Evaluation:** Compare models using appropriate classification metrics(Precision and Recall).
8. **Hyperparameter Tuning:** Optimize the best-performing model using GridSearchCV.
9. **Final Evaluation:** Assess the optimized model on the original, untouched test set.

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/ZafZ-code/Fraud-Detection-Machine-Learning.git
```

### 2. Navigate to the project directory

```bash
cd Fraud-Detection-Machine-Learning
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


## ▶️ Usage

Open the notebooks and execute them in the appropriate order, following the project workflow.

Make sure the dataset is downloaded from Kaggle and placed in the appropriate data directory before running the notebooks.

## 📈 Results & Model Evaluation

Several machine learning models were trained and evaluated using different resampling strategies.

### Best-performing model before hyperparameter tuning

The best-performing configuration was **XGBoost combined with SMOTE and Random Under-Sampling**.

| Evaluation Metric |  Score |
| ----------------- | -----: |
| Training Accuracy | 99.91% |
| Testing Accuracy  | 99.85% |
| Precision         | 46.60% |
| Recall            | 99.15% |
| F1-Score          | 63.41% |
| ROC-AUC           | 99.88% |
| PR-AUC            | 94.28% |

### Best model after hyperparameter tuning

GridSearchCV was used to optimize the XGBoost classifier.

**Best hyperparameters:**

```python
{
    'learning_rate': 0.1,
    'max_depth': 7,
    'min_child_weight': 1,
    'n_estimators': 200,
    'subsample': 0.8
}
```

**Evaluation results:**

| Evaluation Metric |  Score |
| ----------------- | -----: |
| Training Accuracy | 99.81% |
| Testing Accuracy  | 99.78% |
| Precision         | 36.69% |
| Recall            | 99.31% |
| F1-Score          | 53.59% |
| ROC-AUC           | 99.92% |
| PR-AUC            | 94.45% |

### Key Observations

- The initial XGBoost model achieved a strong balance between precision and recall compared with the tuned model at the default classification threshold.
- Hyperparameter tuning improved recall from 99.15% to 99.31%.
- ROC-AUC and PR-AUC also improved slightly after tuning.
- However, precision and F1-score decreased, indicating an increase in false positive predictions.

These results highlight the importance of considering multiple evaluation metrics rather than relying on accuracy alone when dealing with imbalanced classification problems.

## 🔮 Future Improvements

- Experiment with additional machine learning algorithms and ensemble methods.
- Explore more advanced hyperparameter optimization strategies.
- Optimize the classification threshold to achieve a better balance between fraud detection and false positive rates.
- Investigate cost-sensitive learning approaches.
- Explore deep learning architectures for fraud detection.

## 👨‍💻 Author

**Abderahmen Abdelmoula**

- GitHub: [ZafZ-code](https://github.com/ZafZ-code)
- LinkedIn: [Abderahmen Abdelmoula](https://www.linkedin.com/in/abderahmen-abdelmoula-826256201/)

---

⭐ If you find this project interesting, feel free to star the repository!

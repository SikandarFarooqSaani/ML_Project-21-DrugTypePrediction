# 💊 Drug Type Classification Project  

![Drug Logo](https://img.icons8.com/emoji/96/pill.png)  

---

## 📌 Project Overview  
This project focuses on **predicting drug types (A, B, C, X, Y)** based on patient data using **Decision Tree, Logistic Regression, and K-Nearest Neighbors (KNN)**.  
We also applied **GridSearchCV** and **RandomizedSearchCV** to find the best hyperparameters and improve accuracy.  

---

## 🛠️ Libraries Used  

| Logo | Library | Purpose |
|------|---------|---------|
| 🐍 | **OS** | File handling |
| ![Kaggle](https://upload.wikimedia.org/wikipedia/commons/7/7c/Kaggle_logo.png) | **Kaggle** | Dataset import |
| ![NumPy](https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg) | **NumPy** | Numerical operations |
| ![Pandas](https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg) | **Pandas** | Data cleaning & transformation |
| 🌳 | **DecisionTreeClassifier** | Model for classification |
| 🔎 | **GridSearchCV / RandomizedSearchCV** | Hyperparameter tuning |
| 📉 | **LogisticRegression** | Linear model for classification |
| 👥 | **KNeighborsClassifier** | Instance-based learning |
| 📊 | **Matplotlib / Seaborn** | Visualization |

---

## 📂 Dataset  

📌 **Source:** [Drug Type Dataset](https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees)  

- Shape: **(200, 6)**  
- Target variable: **Drug Type (A, B, C, X, Y)**  
- Converted object columns → integers using mapping.  
- No null values or duplicates found.  

---

## ⚙️ Workflow  

1. Imported dataset from Kaggle.  
2. Converted categorical features into **integer mappings**.  
3. Verified dataset integrity (no nulls or duplicates).  
4. Train-test split.  
5. Applied and compared different models.  

---

## 🔍 Model Experiments & Results  

### 📌 Decision Tree (Default)  
- Accuracy: **0.95**  
- Confusion Matrix:  
 <img width="640" height="547" alt="21-1" src="https://github.com/user-attachments/assets/d7308b48-7738-4bf6-a0fb-438766545e51" />

- Feature Importance: Found **Sex** has *zero* importance in decision making.  

---

### 📌 Decision Tree + GridSearchCV  
- Parameter Grid:
- python
param_grid = {
    'criterion':['gini','entropy'],
    'splitter':['best','random'],
    'max_depth':[2,4,8,10,13],
    'max_features':['sqrt','log2',None],
    'min_samples_leaf':[4,5],
}

---
Best Score: 1.0 (Overfitted)

Best Parameters:

{'criterion': 'gini',
 'max_depth': 4,
 'max_features': None,
 'min_samples_leaf': 4,
 'splitter': 'best'}
 
### 📌 Decision Tree + RandomizedSearchCV

Parameters tested with n_iter=10

Best Score: 1.0

Best Parameters:

{'criterion': 'entropy',
 'max_depth': 4,
 'max_features': None,
 'min_samples_leaf': 5,
 'splitter': 'best'}

### 📌 Logistic Regression

Accuracy: 0.875

<img width="640" height="547" alt="21-2" src="https://github.com/user-attachments/assets/9cb9cb72-8a2f-4a9b-addb-b8f52cff716f" />



GridSearchCV on Logistic Regression

Best Score: 0.99375

Best Parameters:

{'C': 50, 'max_iter': 100, 'penalty': 'l2', 'solver': 'newton-cg'}
### 📌 K-Nearest Neighbors (KNN)

Accuracy: 0.70

<img width="640" height="547" alt="21-3" src="https://github.com/user-attachments/assets/98f5184e-0e34-482d-a58e-d2281b47d1d0" />



### 📊 Final Comparison
Model	Accuracy
Decision Tree	0.95
Decision Tree + GridSearch	1.0 (overfit)
Decision Tree + Randomized	1.0 (overfit)
Logistic Regression	0.875
Logistic Regression + Grid	0.9937
K-Nearest Neighbors (KNN)	0.70
### 🚀 Conclusion

Decision Tree gave 95% accuracy, but with tuning it overfit to 100%.

Logistic Regression was robust, especially after tuning (99.3% accuracy).

KNN performed poorly with only 70% accuracy.

Learned the power of GridSearchCV & RandomizedSearchCV for tuning models.

### 💡 Future Improvements

Use Random Forest and XGBoost for better generalization.

Apply cross-validation to avoid overfitting.

Perform scaling/normalization for Logistic Regression and KNN.

Visualize decision boundaries for better interpretation.

### 👨‍💻 Author

Made with ❤️ by Sikandar Farooq Saani

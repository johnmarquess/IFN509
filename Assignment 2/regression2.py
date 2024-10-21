import pandas as pd
import numpy as np
import data_preprocess as dp
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import pickle

# Load and preprocess data
data = pd.read_csv('./data/D2.csv')
df = dp.data_prep(data)
df = df[df.discharge_disposition != 'Deceased']  # Remove 'Deceased' rows

# Features and Target
X = df.drop('readmitted', axis=1)  # Features
y = df['readmitted']  # Target

# Loading a previously saved Decision Tree model
with open('./data/DT.pkl', 'rb') as file:
    dt_model = pickle.load(file)

# Categorical and numerical columns
categorical_columns = ['race', 'medical_specialty', 'admission_type', 'discharge_disposition', 'admission_source']
numerical_columns = ['gender', 'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                     'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
                     'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                     'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'insulin', 'change', 'diabetesMed']

# Encode and scale data
label_encoder = LabelEncoder()
X['age'] = label_encoder.fit_transform(X['age'])
X = pd.get_dummies(X, columns=categorical_columns, dtype=int)
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Feature selection from a decision tree model
important_features_array = np.where(dt_model.feature_importances_ > 0)[0]  # Selecting important features
X = X.iloc[:, important_features_array]  # Keep only important features

# Splitting data
rs = 1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rs)

# Original Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=rs)
log_reg.fit(X_train, y_train)

# GridSearchCV for Logistic Regression
from sklearn.model_selection import GridSearchCV
params = {'C': [pow(10, x) for x in range(-6, 4)]}
grid_search = GridSearchCV(param_grid=params, estimator=log_reg, return_train_score=True, cv=10, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# RFE + Logistic Regression
from sklearn.feature_selection import RFECV
rfe = RFECV(estimator=log_reg, cv=10, n_jobs=-1, verbose=1)
rfe.fit(X_train, y_train)
X_train_sel = rfe.transform(X_train)
X_test_sel = rfe.transform(X_test)

# GridSearchCV on RFE-selected Logistic Regression
rfe_grid_search = GridSearchCV(param_grid=params, estimator=rfe.estimator_, return_train_score=True,
                               cv=10, n_jobs=-1, verbose=1)
rfe_grid_search.fit(X_train_sel, y_train)

# Calculating AUC and ROC Curves for each model
from sklearn.metrics import roc_curve, roc_auc_score

# Original Logistic Regression (Grid Search)
y_pred_proba_gs = grid_search.predict_proba(X_test)
roc_index_gs = roc_auc_score(y_test, y_pred_proba_gs[:, 1])
fpr_gs, tpr_gs, _ = roc_curve(y_test, y_pred_proba_gs[:, 1])

# RFE + Logistic Regression
y_pred_proba_rfe = rfe.estimator_.predict_proba(X_test_sel)
roc_index_rfe = roc_auc_score(y_test, y_pred_proba_rfe[:, 1])
fpr_rfe, tpr_rfe, _ = roc_curve(y_test, y_pred_proba_rfe[:, 1])

# RFE + GridSearchCV
y_pred_proba_rfe_grid = rfe_grid_search.best_estimator_.predict_proba(X_test_sel)
roc_index_rfe_grid = roc_auc_score(y_test, y_pred_proba_rfe_grid[:, 1])
fpr_rfe_grid, tpr_rfe_grid, _ = roc_curve(y_test, y_pred_proba_rfe_grid[:, 1])

# Plotting ROC curves
plt.plot(fpr_gs, tpr_gs, label=f"Grid Search (AUC = {roc_index_gs:.3f})")
plt.plot(fpr_rfe, tpr_rfe, label=f"RFE (AUC = {roc_index_rfe:.3f})")
plt.plot(fpr_rfe_grid, tpr_rfe_grid, label=f"RFE + GridSearch (AUC = {roc_index_rfe_grid:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Save the plot
plt.savefig('./data/roc_curve.png')
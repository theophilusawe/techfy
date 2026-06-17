import os, json, pickle, logging, yaml
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import (cross_val_score,
                                     StratifiedKFold, 
                                     train_test_split,
                                     GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (StandardScaler, 
                                   OneHotEncoder, 
                                   LabelEncoder)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix,ConfusionMatrixDisplay,
    classification_report
)

 

DATA_PATH = "census/adult.data"






# 1. LOAD DATA


column_names = [
    ['age', 'workclass', 'fnlwgt', 'education', 
     'education-num', 'marital_status', 'occupation', 
     'relationship', 'race', 'sex', 'capital-gain', 
     'capital-loss', 'hours-per-week', 'native-country', 'income']
]
adult = pd.read_csv(
    DATA_PATH,
    header=None,
    names=column_names,
    na_values=' ?'
)

original_shape = adult.shape

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    return df

adult = strip_whitespace(adult)









# 2. TRAIN TEST SPLIT


features = adult.drop("income", axis=1)
target = adult["income"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# 3. FEATURE GROUPS


numerical_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

categorical_features = [
    col for col in features.columns
    if col not in numerical_features
]


# 4. NUMERICAL PIPELINE

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# 5. CATEGORICAL PIPELINE

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 6. COLUMN TRANSFORMER

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# 7. FULL PIPELINE


training_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42
    ))
])


# 8. HYPERPARAMETER GRID


param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}


# 9. STRATIFIED K-FOLD


cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 10. GRID SEARCH


"""
Why F1-score instead of Accuracy?

The Adult Income dataset is imbalanced:
roughly 76% of people earn <=50K
and 24% earn >50K.

A model can achieve high accuracy by
predicting the majority class most of the time.

F1-score balances:
- Precision
- Recall

making it more appropriate for imbalanced
classification problems.
"""

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=2
)


# 11. TRAIN


grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV F1 Score:")
print(grid_search.best_score_)


# 12. BEST MODEL


best_model = grid_search.best_estimator_


# 13. TEST PREDICTIONS


y_pred = best_model.predict(X_test)


# 14. CONFUSION MATRIX


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["<=50K", ">50K"]
)

disp.plot()
plt.show()


# 15. CLASSIFICATION REPORT


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["<=50K", ">50K"]
    )
)

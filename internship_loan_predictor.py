import pandas as pd
import numpy as np
from xgboost import XGBClassifier,plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
# df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\train_u6lujuX_CVtuZ9i (1).csv")
# y = df["Loan_Status"].map({'Y': 1, 'N': 0})
# X = df.drop(columns=["Loan_Status", "Loan_ID"], axis=1)

# num_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History", "ApplicantIncome", "CoapplicantIncome"]
# cat_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Education", "Property_Area"]

# num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
# cat_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])
# preprocessor = ColumnTransformer([
#     ("num", num_pipeline, num_cols),
#     ("cat", cat_pipeline, cat_cols)
# ])

# models = {
#     "XGBClassifier": XGBClassifier(),
#     "LogisticRegression": LogisticRegression(max_iter=1000),
#     "RandomForestClassifier": RandomForestClassifier(),
#     "SVC": SVC()
# }

# for name, model in models.items():
#     pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
#     cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
#     print(name)
#     print("CV Scores:", cv_scores)
#     print("CV Mean Accuracy:", cv_scores.mean())
#     print()

# df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\train_u6lujuX_CVtuZ9i (1).csv")
# y = df["Loan_Status"].map({'Y': 1, 'N': 0})
# final_num_cols = ["Credit_History", "CoapplicantIncome"]
# final_cat_cols = ["Married", "Dependents", "Self_Employed", "Education"]
# X_final = df[final_num_cols + final_cat_cols]

# # Calculate scale_pos_weight as (# of negatives)/(# of positives)
# scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
# print("scale_pos_weight:", scale_pos_weight)

# num_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="mean")),
#     ("scaler", StandardScaler())
# ])
# cat_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])
# preprocessor_final = ColumnTransformer([
#     ("num", num_pipeline, final_num_cols),
#     ("cat", cat_pipeline, final_cat_cols)
# ])
# final_model = Pipeline([
#     ("preprocessor", preprocessor_final),
#     ("classifier", XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss'))
# ])
# X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=32)
# final_model.fit(X_train, y_train)
# y_pred = final_model.predict(X_test)
# print("Final Model Accuracy:", accuracy_score(y_test, y_pred))
# print("Final Model F1 Score:", f1_score(y_test, y_pred))
# # Monkey-patch RandomOverSampler so that it has the required sklearn tags
# RandomOverSampler.__sklearn_tags__ = lambda self: {"requires_fit": True}

# df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\train_u6lujuX_CVtuZ9i (1).csv")
# y = df["Loan_Status"].map({'Y': 1, 'N': 0})
# final_num_cols = ["Credit_History", "CoapplicantIncome"]
# final_cat_cols = ["Married", "Dependents", "Self_Employed", "Education"]
# X_final = df[final_num_cols + final_cat_cols]

# num_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="mean")),
#     ("scaler", StandardScaler())
# ])
# cat_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])
# preprocessor_final = ColumnTransformer([
#     ("num", num_pipeline, final_num_cols),
#     ("cat", cat_pipeline, final_cat_cols)
# ])
# final_model = Pipeline([
#     ("preprocessor", preprocessor_final),
#     ("oversampler", RandomOverSampler(random_state=32)),
#     ("classifier", LogisticRegression(max_iter=1000))
# ])

# X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=32)
# final_model.fit(X_train, y_train)
# y_pred = final_model.predict(X_test)
# print("Final Model Accuracy:", accuracy_score(y_test, y_pred))
# print("Final Model F1 Score:", f1_score(y_test, y_pred))

# df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\train_u6lujuX_CVtuZ9i (1).csv")
# y = df["Loan_Status"].map({'Y': 1, 'N': 0})
# final_num_cols = ["Credit_History", "CoapplicantIncome"]
# final_cat_cols = ["Married", "Dependents", "Self_Employed", "Education"]
# X_final = df[final_num_cols + final_cat_cols]

# num_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="mean")),
#     ("scaler", StandardScaler())
# ])
# cat_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])
# preprocessor_final = ColumnTransformer([
#     ("num", num_pipeline, final_num_cols),
#     ("cat", cat_pipeline, final_cat_cols)
# ])
# final_model = Pipeline([
#     ("preprocessor", preprocessor_final),
#     ("classifier", LogisticRegression(max_iter=1000, class_weight='balanced'))
# ])
# X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=32)
# final_model.fit(X_train, y_train)
# y_pred = final_model.predict(X_test)
# print("Final Model Accuracy:", accuracy_score(y_test, y_pred))
# print("Final Model F1 Score:", f1_score(y_test, y_pred))
# sns.countplot(x=y)
# plt.title("Class Distribution")
# plt.xlabel("Loan Status (0 = Not Approved, 1 = Approved)")
# plt.ylabel("Count")
# plt.show()


# df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\train_u6lujuX_CVtuZ9i (1).csv")
# y = df["Loan_Status"].map({'Y': 1, 'N': 0})
# X = df.drop(columns=["Loan_Status", "Loan_ID"], axis=1)

# X["Total_Income"] = X["ApplicantIncome"] + X["CoapplicantIncome"]
# X["Income_to_Loan_Ratio"] = X["Total_Income"] / X["LoanAmount"]
# X["Income_to_Loan_Ratio"] = X["Income_to_Loan_Ratio"].fillna(X["Income_to_Loan_Ratio"].median())

# num_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History", "ApplicantIncome",
#             "CoapplicantIncome", "Total_Income", "Income_to_Loan_Ratio"]
# cat_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Education", "Property_Area"]

# num_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="mean")),
#     ("scaler", StandardScaler())
# ])

# cat_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])

# def objective(trial):
#     transformers = [
#         ("num", num_pipeline, num_cols),
#         ("cat", cat_pipeline, cat_cols)
#     ]
#     preprocessor = ColumnTransformer(transformers)

#     C = trial.suggest_float("C", 0.01, 10.0, log=True)
#     solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
    
#     classifier = LogisticRegression(C=C, solver=solver, max_iter=500)

#     pipe = Pipeline([
#         ("preprocessor", preprocessor),
#         ("classifier", classifier)
#     ])
    
#     score = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()
#     return score

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# print("Best trial:")
# trial = study.best_trial
# print("  Best CV Accuracy:", trial.value)
# print("  Hyperparameters:")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")


# def objective(trial):
#     transformers = [
#         ("num", num_pipeline, num_cols),
#         ("cat", cat_pipeline, cat_cols)
#     ]
#     preprocessor = ColumnTransformer(transformers)

#     num_leaves = trial.suggest_int("num_leaves", 20, 150)
#     learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
#     n_estimators = trial.suggest_int("n_estimators", 50, 300)
#     min_child_samples = trial.suggest_int("min_child_samples", 10, 100)
    
#     classifier = lgb.LGBMClassifier(
#         num_leaves=num_leaves,
#         learning_rate=learning_rate,
#         n_estimators=n_estimators,
#         min_child_samples=min_child_samples
#     )

#     pipe = Pipeline([
#         ("preprocessor", preprocessor),
#         ("classifier", classifier)
#     ])
    
#     score = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()
#     return score

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# print("Best trial:")
# trial = study.best_trial
# print("  Best CV Accuracy:", trial.value)
# print("  Hyperparameters:")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")


# df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\train_u6lujuX_CVtuZ9i (1).csv")
# y = df["Loan_Status"].map({'Y': 1, 'N': 0})
# X = df.drop(columns=["Loan_Status", "Loan_ID"], axis=1)
# X["Total_Income"] = X["ApplicantIncome"] + X["CoapplicantIncome"]
# X["Income_to_Loan_Ratio"] = X["Total_Income"] / X["LoanAmount"]
# X["Income_to_Loan_Ratio"] = X["Income_to_Loan_Ratio"].fillna(X["Income_to_Loan_Ratio"].median())

# num_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History", "ApplicantIncome",
#             "CoapplicantIncome", "Total_Income", "Income_to_Loan_Ratio"]
# cat_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Education", "Property_Area"]

# num_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="mean")),
#     ("scaler", StandardScaler())
# ])
# cat_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])

# def objective(trial):
#     selected_num_cols = []
#     selected_cat_cols = []
#     for col in num_cols:
#         if trial.suggest_categorical(f"use_{col}", [True, False]):
#             selected_num_cols.append(col)
#     for col in cat_cols:
#         if trial.suggest_categorical(f"use_{col}", [True, False]):
#             selected_cat_cols.append(col)
#     selected_cols = selected_num_cols + selected_cat_cols
#     if len(selected_cols) == 0:
#         selected_num_cols = num_cols
#         selected_cat_cols = cat_cols
#         selected_cols = num_cols + cat_cols
#     transformers = []
#     if selected_num_cols:
#         transformers.append(("num", num_pipeline, selected_num_cols))
#     if selected_cat_cols:
#         transformers.append(("cat", cat_pipeline, selected_cat_cols))
#     preprocessor_sel = ColumnTransformer(transformers)
    
#     max_depth = trial.suggest_int("max_depth", 3, 10)
#     learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
#     n_estimators = trial.suggest_int("n_estimators", 50, 300)
#     subsample = trial.suggest_float("subsample", 0.5, 1.0)
#     colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
#     scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 3.0)
    
#     classifier = XGBClassifier(use_label_encoder=False,
#                                eval_metric="logloss",
#                                max_depth=max_depth,
#                                learning_rate=learning_rate,
#                                n_estimators=n_estimators,
#                                subsample=subsample,
#                                colsample_bytree=colsample_bytree,
#                                scale_pos_weight=scale_pos_weight)
    
#     pipe = Pipeline([
#         ("preprocessor", preprocessor_sel),
#         ("classifier", classifier)
#     ])
#     score = cross_val_score(pipe, X[selected_cols], y, cv=5, scoring="accuracy").mean()
#     return score

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# print("Best trial:")
# trial = study.best_trial
# print("  Best CV Accuracy:", trial.value)
# print("  Feature Selection and Hyperparameters:")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")


# df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\train_u6lujuX_CVtuZ9i (1).csv")
# y = df["Loan_Status"].map({'Y': 1, 'N': 0})
# X = df.drop(columns=["Loan_Status", "Loan_ID"], axis=1)
# X["Total_Income"] = X["ApplicantIncome"] + X["CoapplicantIncome"]
# X["Income_to_Loan_Ratio"] = X["Total_Income"] / X["LoanAmount"]
# X["Income_to_Loan_Ratio"] = X["Income_to_Loan_Ratio"].fillna(X["Income_to_Loan_Ratio"].median())

# num_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History", "ApplicantIncome",
#             "CoapplicantIncome", "Total_Income", "Income_to_Loan_Ratio"]
# cat_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Education", "Property_Area"]

# num_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="mean")),
#     ("scaler", StandardScaler())
# ])
# cat_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])

# def objective(trial):
#     selected_num_cols = []
#     selected_cat_cols = []
#     for col in num_cols:
#         if trial.suggest_categorical(f"use_{col}", [True, False]):
#             selected_num_cols.append(col)
#     for col in cat_cols:
#         if trial.suggest_categorical(f"use_{col}", [True, False]):
#             selected_cat_cols.append(col)
#     selected_cols = selected_num_cols + selected_cat_cols
#     if len(selected_cols) == 0:
#         selected_num_cols = num_cols
#         selected_cat_cols = cat_cols
#         selected_cols = num_cols + cat_cols
#     transformers = []
#     if selected_num_cols:
#         transformers.append(("num", num_pipeline, selected_num_cols))
#     if selected_cat_cols:
#         transformers.append(("cat", cat_pipeline, selected_cat_cols))
#     preprocessor_sel = ColumnTransformer(transformers)
    
#     max_depth = trial.suggest_int("max_depth", 3, 10)
#     learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
#     n_estimators = trial.suggest_int("n_estimators", 50, 300)
#     subsample = trial.suggest_float("subsample", 0.5, 1.0)
#     colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
#     scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 3.0)
    
#     classifier = XGBClassifier(use_label_encoder=False,
#                                eval_metric="logloss",
#                                max_depth=max_depth,
#                                learning_rate=learning_rate,
#                                n_estimators=n_estimators,
#                                subsample=subsample,
#                                colsample_bytree=colsample_bytree,
#                                scale_pos_weight=scale_pos_weight)
    
#     pipe = Pipeline([
#         ("preprocessor", preprocessor_sel),
#         ("classifier", classifier)
#     ])
#     score = cross_val_score(pipe, X[selected_cols], y, cv=5, scoring="accuracy").mean()
#     return score

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# print("Best trial:")
# trial = study.best_trial
# print("  Best CV Accuracy:", trial.value)
# print("  Feature Selection and Hyperparameters:")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")


# df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\train_u6lujuX_CVtuZ9i (1).csv")
# y = df["Loan_Status"].map({'Y': 1, 'N': 0})

# df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
# df['Income_to_Loan_Ratio'] = df['Total_Income'] / df['LoanAmount']
# df['Income_to_Loan_Ratio'] = df['Income_to_Loan_Ratio'].fillna(df['Income_to_Loan_Ratio'].median())

# final_num_cols = ["Credit_History", "CoapplicantIncome", "Total_Income", "Income_to_Loan_Ratio"]
# final_cat_cols = ["Married", "Dependents", "Self_Employed", "Education", "Property_Area"]
# X_final = df[final_num_cols + final_cat_cols]

# num_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler())
# ])
# cat_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])
# preprocessor_final = ColumnTransformer([
#     ("num", num_pipeline, final_num_cols),
#     ("cat", cat_pipeline, final_cat_cols)
# ])

# X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=32, stratify=y)
# X_train_processed = preprocessor_final.fit_transform(X_train)
# X_test_processed = preprocessor_final.transform(X_test)

# final_model = XGBClassifier(
#     use_label_encoder=False,
#     eval_metric="logloss",
#     max_depth=5,
#     learning_rate=0.022373914062204868,
#     n_estimators=134,
#     subsample=0.5578078732239936,
#     colsample_bytree=0.8636043927135295,
#     scale_pos_weight=1.4688338969173822
# )
# final_model.fit(X_train_processed, y_train)

# y_pred = final_model.predict(X_test_processed)
# print("Final Model Accuracy:", accuracy_score(y_test, y_pred))
# print("Final Model F1 Score:", f1_score(y_test, y_pred))

# # plot_importance(final_model)
# # plt.show()

# # sns.countplot(x=y)
# # plt.title("Class Distribution")
# # plt.xlabel("Loan Status (0 = Not Approved, 1 = Approved)")
# # plt.ylabel("Count")
# # plt.show()
# features = preprocessor_final.get_feature_names_out()

# fraud_detector = IsolationForest(contamination=0.02, random_state=42)
# anomaly_scores = fraud_detector.fit_predict(X_train_processed)
# X_train_anomaly = X_train.copy()
# X_train_anomaly["Anomaly_Score"] = anomaly_scores
# suspicious_cases = X_train_anomaly[X_train_anomaly["Anomaly_Score"] == -1]
# print("Potential Fraudulent Cases (Anomalies) in Training Data:", len(suspicious_cases))
# def explaining(model,x_int,features):
#     explainer=shap.Explainer(final_model)
#     shap_values = explainer(x_int)
#     # shap.summary_plot(shap_values, X_test_processed)
#     feature_contri=shap_values.values[0]
#     base_values=shap_values.base_values[0]
#     final_pred=model.predict(x_int)
#     top_features=np.argsort(abs(feature_contri))[-3:]
#     explanation = f"Your loan application was {'approved' if final_pred== 1 else 'rejected'} because:\n"
#     for i in reversed(top_features):
#         sign = "increased" if feature_contri[i] > 0 else "decreased"
#         explanation += f"- {features[i]} {sign} your chances.\n"

#     return explanation
# i=3
# print(explaining(final_model, X_test_processed[i:i+1], features))
# print(X_train_anomaly["Anomaly_Score"].value_counts())

# anomaly_scores_test=fraud_detector.predict(X_test_processed)
# X_test_anomaly=X_test.copy()
# X_test_anomaly["Anomaly_Score"]=anomaly_scores_test
# print(X_test_anomaly["Anomaly_Score"].value_counts())


# df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\train_u6lujuX_CVtuZ9i (1).csv")

# y = df["Loan_Status"].map({'Y': 1, 'N': 0})

# df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
# df["Income_to_Loan_Ratio"] = df["Total_Income"] / df["LoanAmount"]
# df["Income_to_Loan_Ratio"] = df["Income_to_Loan_Ratio"].fillna(df["Income_to_Loan_Ratio"].median())

# final_num_cols = ["Credit_History", "CoapplicantIncome", "Total_Income", "Income_to_Loan_Ratio"]
# final_cat_cols = ["Married", "Dependents", "Self_Employed", "Education", "Property_Area"]

# X = df[final_num_cols + final_cat_cols].copy()

# num_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler())
# ])

# cat_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])

# preprocessor = ColumnTransformer([
#     ("num", num_pipeline, final_num_cols),
#     ("cat", cat_pipeline, final_cat_cols)
# ])

# models = {
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
#     "LightGBM": lgb.LGBMClassifier(),
#     "CatBoost": CatBoostClassifier(verbose=0)
# }

# for name, model in models.items():
#     pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
#     cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
#     print(f"{name} Model:")
#     print("CV Scores:", cv_scores)
#     print("CV Mean Accuracy:", cv_scores.mean())
#     print("-" * 50)

import optuna
import numpy as np
import shap
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\train_u6lujuX_CVtuZ9i (1).csv")
y = df["Loan_Status"].map({'Y': 1, 'N': 0})

df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_to_Loan_Ratio'] = df['Total_Income'] / df['LoanAmount']
df['Income_to_Loan_Ratio'] = df['Income_to_Loan_Ratio'].fillna(df['Income_to_Loan_Ratio'].median())

final_num_cols = ["Credit_History", "CoapplicantIncome", "Total_Income", "Income_to_Loan_Ratio"]
final_cat_cols = ["Married", "Dependents", "Self_Employed", "Education", "Property_Area"]
X_final = df[final_num_cols + final_cat_cols]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor_final = ColumnTransformer([
    ("num", num_pipeline, final_num_cols),
    ("cat", cat_pipeline, final_cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=32, stratify=y)
X_train_processed = preprocessor_final.fit_transform(X_train)
X_test_processed = preprocessor_final.transform(X_test)


# def objective(trial):
#     # Define hyperparameters to tune
#     n_estimators = trial.suggest_int("n_estimators", 50, 300)
#     max_samples = trial.suggest_float("max_samples", 0.5, 1.0)
#     contamination = trial.suggest_float("contamination", 0.01, 0.1)
#     max_features = trial.suggest_float("max_features", 0.5, 1.0)
    
#     # Train Isolation Forest
#     model = IsolationForest(
#         n_estimators=n_estimators,
#         max_samples=max_samples,
#         contamination=contamination,
#         max_features=max_features,
#         random_state=42
#     )
#     model.fit(X_train_processed)
#     predictions = model.predict(X_test_processed)
#     predictions = np.where(predictions == 1, 0, 1)  # Convert to binary (1 = fraud, 0 = normal)
    
#     # Evaluate using F1-score (for imbalanced fraud detection)
#     score = f1_score(y_test, predictions)
#     return score

# # Run Optuna optimization
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# # Best hyperparameters
# best_params = study.best_params
# print("Best Parameters:", best_params)

# # Train final model with best hyperparameters
# final_model = IsolationForest(**best_params, random_state=42)
# final_model.fit(X_train_processed)

# # SHAP Explanation Function
# def explaining(model, x_int, features):
#     explainer = shap.Explainer(model)
#     shap_values = explainer(x_int)
#     feature_contri = shap_values.values[0]
#     final_pred = model.predict(x_int)
    
#     top_features = np.argsort(abs(feature_contri))[-3:]
#     explanation = f"Your loan application was {'approved' if final_pred == 1 else 'rejected'} because:\n"
#     for i in reversed(top_features):
#         sign = "increased" if feature_contri[i] > 0 else "decreased"
#         explanation += f"- {features[i]} {sign} your chances.\n"
    
#     return explanation
best_params={'n_estimators': 56, 'max_samples': 0.6908277955472621, 'contamination': 0.09998167529183737, 'max_features': 0.6955017814682117}
fraud_predictor=IsolationForest(**best_params, random_state=42)
fraud_predictor.fit(X_train_processed)
test_predictions = fraud_predictor.predict(X_test_processed)
test_anomalies = np.sum(test_predictions == -1)
print(f"Detected Fraud Cases in Test Set: {test_anomalies}")
import joblib
with open("fraud_detector.pkl","wb") as f:
 joblib.dump(fraud_predictor,f)

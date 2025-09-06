import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load CSV
df = pd.read_csv("synthetic_health_dataset_v3.csv")

# Separate features & target
X = df.drop(columns=["deteriorated_health (0/1)"])
y = df["deteriorated_health (0/1)"]


# Identify categorical and numeric columns
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Encode categorical (yes/no, M/F, chest pain type, smoker, etc.)
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predictions
y_pred_prob = log_reg.predict_proba(X_test)[:,1]
y_pred = log_reg.predict(X_test)

# Evaluation
print("AUROC:", roc_auc_score(y_test, y_pred_prob))
print("AUPRC:", average_precision_score(y_test, y_pred_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))
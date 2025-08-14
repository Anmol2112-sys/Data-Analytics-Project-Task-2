import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# ...rest of your code...
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

# 1. Load Dataset
# generate_diabetes_data.py

import pandas as pd
import numpy as np
import random

# Number of rows to generate
num_rows = 768  # same as the famous Pima Indians dataset

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data
data = {
    'Pregnancies': np.random.randint(0, 17, size=num_rows),
    'Glucose': np.random.randint(70, 200, size=num_rows),
    'BloodPressure': np.random.randint(40, 122, size=num_rows),
    'SkinThickness': np.random.randint(10, 99, size=num_rows),
    'Insulin': np.random.randint(15, 846, size=num_rows),
    'BMI': np.round(np.random.uniform(18.0, 50.0, size=num_rows), 1),
    'DiabetesPedigreeFunction': np.round(np.random.uniform(0.05, 2.5, size=num_rows), 3),
    'Age': np.random.randint(21, 80, size=num_rows),
    'Outcome': np.random.randint(0, 2, size=num_rows)
}



url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
df = pd.read_csv(url, names=column_names)# Pima Indians Diabetes Dataset

# 2. Data Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    auc_score = roc_auc_score(y_test, preds)
    print(f"\n{name} Results:")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {auc_score:.3f}")

# 5. Feature Importance (Random Forest Example)
rf = models["Random Forest"]
importances = rf.feature_importances_
features = X.columns
plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance - Random Forest")
plt.show()

# 6. Confusion Matrix (Random Forest)
best_model = models["Random Forest"]
cm = confusion_matrix(y_test, best_model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. ROC Curve (Random Forest)
y_score = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Random Forest')
plt.legend(loc="lower right")
plt.show()
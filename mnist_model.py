# 🧠 MNIST Digit Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------
df = pd.read_csv("mnist_train_small.csv")

print("Dataset Shape:", df.shape)

# First column = label
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# ----------------------------
# 2️⃣ Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 3️⃣ Train Model
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 4️⃣ Predictions
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# 5️⃣ Evaluation
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}\n")

print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# 6️⃣ Confusion Matrix
# ----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d',
            cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------
# 7️⃣ Show Sample Predictions
# ----------------------------
plt.figure(figsize=(10,4))

for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_test.iloc[i].values.reshape(28,28), cmap='gray')
    plt.title(f"Actual: {y_test.iloc[i]}\nPred: {y_pred[i]}")
    plt.axis("off")

plt.suptitle("Sample Predictions")
plt.show()
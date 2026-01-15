from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

os.makedirs("Results", exist_ok=True)
with open("Results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")

os.makedirs("Model", exist_ok=True)
joblib.dump(model, "Model/model.pkl")

print("Training done")

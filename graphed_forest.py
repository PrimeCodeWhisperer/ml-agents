import joblib
import pandas as pd
import matplotlib.pyplot as plt

#Loads the trained model
data = joblib.load("model_steps_per_second.pkl")
model = data["model"]
columns = data["columns"]

#Loads the dataset
df = pd.read_csv("complete_training_logs.csv")

features = [
    "num_cores",
    "total_ram",
    "avg_tracked_cpu_percent",
    "avg_ram_usage",
    "batch_size"
]

X = df[features]
y_true = df["steps_per_second"]
y_pred = model.predict(X)
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()],
         linestyle="--")
plt.xlabel("Actual steps per second")
plt.ylabel("Predicted steps per second")
plt.title("Predicted vs Actual Steps per Second")
plt.tight_layout()
plt.show()

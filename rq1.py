import os
import matplotlib.pyplot as plt
from ml-models.models.linear_regressor import LinearRegressionModel
from ml-models.models.random_forrest import RandomForrestModel
from dotenv import load_dotenv
import numpy as np


load_dotenv()
DATA_PATH = os.getenv("TRAINING_DATA_PATH")
TARGET = "steps_to_threshold"

# Random Forest parameters
N_TREES = 100
MAX_DEPTH = None
SEED = 95

# Load & Train Linear Regression
linear_model = LinearRegressionModel(target=TARGET)
X, y = linear_model.load_data(DATA_PATH)
y_test_lin, lin_preds = linear_model.train(X, y)
lin_r2, lin_mae = linear_model.evaluate_predictions(y_test_lin, lin_preds)
linear_model.print_metrics(lin_r2, lin_mae)


# Load & Train Random Forest

rf_model = RandomForrestModel(target=TARGET, n_trees=N_TREES, seed=SEED, max_depth=MAX_DEPTH)
X, y = rf_model.load_data(DATA_PATH)
y_test_rf, rf_preds = rf_model.train(X, y)
rf_r2, rf_mae = rf_model.evaluate_predictions(y_test_rf, rf_preds)
rf_model.print_metrics(rf_r2, rf_mae)


# Scatter Plot: Predicted vs Actual

plt.figure(figsize=(8, 8))
plt.scatter(y_test_lin, lin_preds, alpha=0.6, label="Linear Regression")
plt.scatter(y_test_rf, rf_preds, alpha=0.6, label="Random Forest")

min_val = min(y_test_lin.min(), lin_preds.min(), rf_preds.min())
max_val = max(y_test_lin.max(), lin_preds.max(), rf_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black")

plt.xlabel("Actual Steps")
plt.ylabel("Predicted Steps")
plt.title("Predicted vs Actual Steps to Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.show()

# Bar Plot: R² and MAE Comparison



models = ["Linear Regression", "Random Forest"]
r2_scores = [lin_r2, rf_r2]
mae_scores = [lin_mae, rf_mae]

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 5))

# R² bars
ax1.bar(x - width/2, r2_scores, width, label="R²", color="skyblue")
ax1.set_ylabel("R² Score")
ax1.set_ylim(0, 1)

# MAE bars on second axis
ax2 = ax1.twinx()
ax2.bar(x + width/2, mae_scores, width, label="MAE", color="salmon")
ax2.set_ylabel("Mean Absolute Error")
ax2.set_ylim(0, max(mae_scores) * 1.2)

ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_title("Model Performance Comparison")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

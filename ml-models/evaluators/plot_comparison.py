import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.model_selection import learning_curve, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load environment variables
load_dotenv()
data_path = os.getenv('TRAINING_DATA_PATH')

if not data_path or not os.path.exists(data_path):
    print(f"Error: Could not find training data at {data_path}")
    exit(1)

# Load dataset
print(f"Loading {data_path}")
if data_path.endswith(".xlsx"):
    df = pd.read_excel(data_path)
else:
    df = pd.read_csv(data_path)

#Define the features and target
features = [
    'num_cores',
    'total_ram',
    'avg_tracked_cpu_percent',
    'avg_ram_usage',
    'batch_size'
]

target = 'steps_per_second'

# Check required columns
missing = [c for c in features + [target] if c not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
    exit(1)

# Prepare data
X = df[features].copy()
y = df[target].copy()

# Drop rows with missing values
valid_indices = X.notna().all(axis=1) & y.notna()
X = X[valid_indices]
y = y[valid_indices]

print(f"Using {len(X)} samples for training")

# Define which models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=95, n_jobs=-1),
    'Linear Regression': LinearRegression()
}

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training vs Validation Curves: Random Forest vs Linear Regression', fontsize=14, fontweight='bold')

# Define train sizes for learning curves
train_sizes = np.linspace(0.1, 1.0, 10)

# Plot learning curves for each model
for idx, (model_name, model) in enumerate(models.items()):
    print(f"\nGenerating learning curve for {model_name}...")

    # Get learning curve
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=5,
        train_sizes=train_sizes,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )

    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot
    ax = axes[idx]
    ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training score', linewidth=2)
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.plot(train_sizes_abs, val_mean, 's-', color='red', label='Validation score', linewidth=2)
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.2, 1.05])

plt.tight_layout()
#Now into the grpahs folder
full_path = os.path.join("Graphs", "model_comparison_curves.png")
plt.savefig(full_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as {full_path}")
plt.show()

for model_name, model in models.items():
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2',
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    print(f"\n{model_name}:")
    print(f"  Training R² (mean): {train_mean[-1]:.4f}")
    print(f"  Validation R² (mean): {val_mean[-1]:.4f}")
    print(f"  Gap (Train - Val): {train_mean[-1] - val_mean[-1]:.4f}")

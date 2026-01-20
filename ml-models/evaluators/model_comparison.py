import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def load_model(model_path):
    """Loaded model and feature names from file"""
    data = joblib.load(model_path)
    return data['model'], data['columns']

def prepare_data(data_path, target, model_columns, model_type):
    """prepare data using the same preprocessing as the trained model"""
    if data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    
    df = df.dropna(subset=[target])

    if model_type == 'rf':
        # Random Forest: use split strategy features with get_dummies
        features = ['scene', 'batch_size', 'algorithm', 'num_cores',
                   'total_ram', 'cpu_model', 'hyper_learning_rate', 'operating_system']
        df = df.dropna(subset=features)
        X = df[features]
        X = pd.get_dummies(X)

        # Align with model's training columns
        for col in model_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[model_columns]
    else:
        # Linear Regression: use LR features with get_dummies
        features = ['scene', 'batch_size', 'algorithm', 'num_cores', 'total_ram', 'cpu_model', 'hyper_learning_rate', 'operating_system']
        df = df.dropna(subset=features)
        X = df[features]
        X = pd.get_dummies(X, drop_first=True)


        # Align with model's training columns
        for col in model_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[model_columns]

    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def compare_models(rf_model, lr_model, X_test_rf, X_test_lr, y_test_rf, y_test_lr):
    """ predictions from both models"""
    rf_preds = rf_model.predict(X_test_rf)
    rf_r2 = r2_score(y_test_rf, rf_preds)
    rf_mae = mean_absolute_error(y_test_rf, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test_rf, rf_preds))

    lr_preds = lr_model.predict(X_test_lr)
    lr_r2 = r2_score(y_test_lr, lr_preds)
    lr_mae = mean_absolute_error(y_test_lr, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test_lr, lr_preds))

    return {
        'rf': {'r2': rf_r2, 'mae': rf_mae, 'rmse': rf_rmse, 'preds': rf_preds, 'actual': y_test_rf.values},
        'lr': {'r2': lr_r2, 'mae': lr_mae, 'rmse': lr_rmse, 'preds': lr_preds, 'actual': y_test_lr.values}
    }

def create_metrics_graph(target, results):
    """ comparison graph for R² and MAE"""
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['Random Forest', 'Linear Regression']
    r2_scores = [results['rf']['r2'], results['lr']['r2']]
    mae_scores = [results['rf']['mae'], results['lr']['mae']]

    x = np.arange(len(models))
    width = 0.35
    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, r2_scores, width, label='R²', alpha=0.8, color='steelblue')
    bars2 = ax2.bar(x + width/2, mae_scores, width, label='MAE', alpha=0.8, color='coral')

    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_title(f'{target} - Model Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, max(1.1, max(r2_scores) * 1.2))
    ax2.set_ylim(0, max(mae_scores) * 1.3)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'comparison_{target}_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: comparison_{target}_metrics.png")
    plt.close()

def create_scatter_plot(target, model_name, actual, preds, r2):
    """scatter plot for predictions vs actual."""
    fig, ax = plt.subplots(figsize=(10, 8))

    color = 'steelblue' if model_name == 'Random Forest' else 'coral'
    ax.scatter(actual, preds, alpha=0.6, s=50, color=color)

    min_val = min(actual.min(), preds.min())
    max_val = max(actual.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')

    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'{model_name}: {target} (R² = {r2:.4f})', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()

    filename = f'comparison_{target}_{model_name.lower().replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def create_combined_scatter_plot(target, results):
    """combined scatter plot comparing both models (as in rq1.py)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(results['rf']['actual'], results['rf']['preds'],
               alpha=0.6, label='Random Forest', color='steelblue', s=50)
    ax.scatter(results['lr']['actual'], results['lr']['preds'],
               alpha=0.6, label='Linear Regression', color='coral', s=50)

    min_val = min(results['rf']['actual'].min(), results['rf']['preds'].min(),
                  results['lr']['actual'].min(), results['lr']['preds'].min())
    max_val = max(results['rf']['actual'].max(), results['rf']['preds'].max(),
                  results['lr']['actual'].max(), results['lr']['preds'].max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', linewidth=2)

    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'Predicted vs Actual: {target}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    filename = f'comparison_{target}_combined.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def print_results(target, results):
    """Print comparison results."""

    print("\n" + "="*70)
    print(f"RESULTS: {target}")
    print("="*70)
    print(f"{'Model':<20} {'R² Score':<15} {'MAE':<15} {'RMSE':<15}")
    print("-"*70)
    print(f"{'Random Forest':<20} {results['rf']['r2']:<15.4f} {results['rf']['mae']:<15.2f} {results['rf']['rmse']:<15.2f}")
    print(f"{'Linear Regression':<20} {results['lr']['r2']:<15.4f} {results['lr']['mae']:<15.2f} {results['lr']['rmse']:<15.2f}")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Compare Random Forest vs Linear Regression')
    parser.add_argument('--target', type=str, required=True, help='Target variable')
    args = parser.parse_args()
    target = args.target

    load_dotenv()
    data_path = os.getenv('TRAINING_DATA_PATH')

    if not data_path or not os.path.exists(data_path):
        print("Error: TRAINING_DATA_PATH not set")
        sys.exit(1)

    # Load models
    try:
        rf_model, rf_columns = load_model(f'model_{target}.pkl')
        lr_model, lr_columns = load_model(f'linear_model_{target}.pkl')
    except Exception as e:
        print(f"Error: Could not load models for {target}: {e}")
        sys.exit(1)

    # Prepare data
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = prepare_data(data_path, target, rf_columns, 'rf')
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = prepare_data(data_path, target, lr_columns, 'lr')

    # Compare
    results = compare_models(rf_model, lr_model, X_test_rf, X_test_lr, y_test_rf, y_test_lr)

    # Print and save
    print_results(target, results)

    print("Generating graphs...")
    create_metrics_graph(target, results)
    create_scatter_plot(target, 'Random Forest', results['rf']['actual'], results['rf']['preds'], results['rf']['r2'])
    create_scatter_plot(target, 'Linear Regression', results['lr']['actual'], results['lr']['preds'], results['lr']['r2'])
    create_combined_scatter_plot(target, results)

if __name__ == "__main__":
    main()

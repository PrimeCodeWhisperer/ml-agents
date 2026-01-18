import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


class RandomForestTrainer:
    """Train Random Forest models with different cross-validation strategies."""

    def __init__(self, target, n_trees=100, seed=95, max_depth=None):
        self.target = target
        self.n_trees = n_trees
        self.seed = seed
        self.max_depth = max_depth
        self.model = None
        self.model_columns = []

    def load_data(self, data_path):
        """Load dataset from Excel or CSV."""
        print(f"Loading {data_path}")
        try:
            if data_path.endswith(".xlsx"):
                return pd.read_excel(data_path, engine="openpyxl")
            else:
                return pd.read_csv(data_path)
        except Exception as e:
            print(f"File cannot be loaded: {e}")
            print(f"Suggest to check TRAINING_DATA_PATH in your .env file.")
            return None

    def train_with_split(self, data_path):
        """Train using simple 80-20 train-test split."""
        df = self.load_data(data_path)
        if df is None:
            return

        df = df.dropna(subset=[self.target])

        features = ['scene', 'batch_size', 'algorithm', 'num_cores',
                   'total_ram', 'cpu_model', 'hyper_learning_rate', 'operating_system']

        # Check for missing columns
        missing = [c for c in features + [self.target] if c not in df.columns]
        if missing:
            print(f"Missing columns: {missing}")
            return

        X = pd.get_dummies(df[features])
        y = df[self.target]
        self.model_columns = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Training model...")
        self.model = RandomForestRegressor(
            n_estimators=self.n_trees,
            random_state=self.seed,
            max_depth=self.max_depth,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        r2_acc = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        print(f"Accuracy (R²): {r2_acc:.3f}")
        print(f"Average Error ({self.target}): {mae:.3f}")

    def train_with_kfold(self, data_path, n_splits=5):
        """Train using K-Fold cross-validation."""
        df = self.load_data(data_path)
        if df is None:
            return

        df = df.dropna(subset=[self.target])

        features = ["num_cores", "total_ram", "avg_tracked_cpu_percent",
                   "avg_ram_usage", "batch_size"]

        missing = [c for c in features + [self.target] if c not in df.columns]
        if missing:
            print(f"Missing columns: {missing}")
            return

        X = df[features]
        y = df[self.target]
        self.model_columns = list(X.columns)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        r2_scores = []
        mae_scores = []

        print(f"Running {n_splits}-Fold cross-validation...")
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = RandomForestRegressor(
                n_estimators=self.n_trees,
                random_state=self.seed,
                max_depth=self.max_depth,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            r2_scores.append(r2)
            mae_scores.append(mae)

            print(f"Fold {fold}: R²={r2:.3f}, MAE={mae:.1f}")

        print(f"Mean R²: {sum(r2_scores)/len(r2_scores):.3f}")
        print(f"Mean MAE: {sum(mae_scores)/len(mae_scores):.1f}")

        # Train final model on full dataset
        print("Training final model on full dataset...")
        self.model = RandomForestRegressor(
            n_estimators=self.n_trees,
            random_state=self.seed,
            max_depth=self.max_depth,
            n_jobs=-1
        )
        self.model.fit(X, y)

    def train_with_groupkfold(self, data_path, n_splits=5):
        """Train using GroupKFold cross-validation by CPU model."""
        df = self.load_data(data_path)
        if df is None:
            return

        df = df.dropna(subset=["cpu_model", self.target])

        features = ["num_cores", "total_ram", "avg_tracked_cpu_percent",
                   "avg_ram_usage", "batch_size"]

        missing = [c for c in features + [self.target] if c not in df.columns]
        if missing:
            print(f"Missing columns: {missing}")
            return

        X = df[features]
        y = df[self.target]
        groups = df["cpu_model"]
        self.model_columns = list(X.columns)

        n_groups = groups.nunique()
        if n_groups < 3:
            print(f"Not enough CPU groups for GroupKFold (found {n_groups})")
            return

        n_splits = min(n_splits, n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        r2_scores = []
        mae_scores = []

        print(f"Running GroupKFold cross-validation (by cpu_model, {n_splits} splits)...")
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = RandomForestRegressor(
                n_estimators=self.n_trees,
                random_state=self.seed,
                max_depth=self.max_depth,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            r2_scores.append(r2)
            mae_scores.append(mae)

            print(f"Fold {fold}: R²={r2:.3f}, MAE={mae:.1f}")

        print(f"Mean R²: {sum(r2_scores)/len(r2_scores):.3f}")
        print(f"Mean MAE: {sum(mae_scores)/len(mae_scores):.1f}")

        # Train final model on full dataset
        print("Training final model on full dataset...")
        self.model = RandomForestRegressor(
            n_estimators=self.n_trees,
            random_state=self.seed,
            max_depth=self.max_depth,
            n_jobs=-1
        )
        self.model.fit(X, y)

    def check_feature_importance(self):
        """Analyze and display aggregated feature importance."""
        if self.model is None:
            print("Error: No trained model found.")
            return

        if not hasattr(self.model, 'feature_importances_'):
            print("Error: This model doesn't support feature importance.")
            return

        # Get raw importances from all dummy columns
        raw_importances = pd.DataFrame({
            'Feature': self.model_columns,
            'Importance': self.model.feature_importances_
        })

        # Define original feature groups to aggregate
        base_features = [
            'batch_size',
            'num_cores',
            'total_ram',
            'cpu_model',
            'hyper_learning_rate',
            'operating_system',
            'avg_tracked_cpu_percent',
            'avg_ram_usage',
            'scene',
            'algorithm'
        ]

        aggregated_data = []

        # Sum up importance for each feature group (handles dummy variables)
        for base in base_features:
            mask = raw_importances['Feature'].str.startswith(base)
            total_score = raw_importances.loc[mask, 'Importance'].sum()

            if total_score > 0:  # Only include features present in the model
                aggregated_data.append({'Feature': base, 'Importance': total_score})

        # Create and sort final table
        if aggregated_data:
            agg_df = pd.DataFrame(aggregated_data)
            agg_df = agg_df.sort_values(by='Importance', ascending=False)

            print("\n" + "="*40)
            print(" AGGREGATED FEATURE IMPORTANCE")
            print("="*40)
            print(agg_df.to_string(index=False, formatters={'Importance': '{:.1%}'.format}))
            print("="*40)
            print("\n(Note: Features with 0% importance may be constant across all samples.)")
        else:
            print("No feature importance data available.")

    def save_model(self, filename):
        """Save trained model to file."""
        if self.model is None:
            print("No model to save. Train a model first.")
            return

        payload = {"model": self.model, "columns": self.model_columns}
        joblib.dump(payload, filename)
        print(f"Saved model to {filename}")


def main():
    load_dotenv()
    data_path = os.getenv("TRAINING_DATA_PATH")

    parser = argparse.ArgumentParser(
        description="Train Random Forest models with various CV strategies"
    )
    parser.add_argument("--strategy", type=str, default="kfold",
                       choices=["split", "kfold", "groupkfold"],
                       help="CV strategy: split, kfold, or groupkfold")
    parser.add_argument("--target", type=str, default="steps_per_second",
                       help="Target column name (default: steps_per_second)")
    parser.add_argument("--trees", type=int, default=100,
                       help="Number of trees in the forest (default: 100)")
    parser.add_argument("--seed", type=int, default=95,
                       help="Random seed for reproducibility (default: 95)")
    parser.add_argument("--depth", type=int, default=None,
                       help="Max depth of trees (default: None/Unlimited)")

    args = parser.parse_args()

    if not data_path:
        print("Error: TRAINING_DATA_PATH not found in .env file")
        return

    print(f"Configuration: Strategy={args.strategy}, Target={args.target}, "
          f"Trees={args.trees}, Seed={args.seed}, Max_depth={args.depth}\n")

    trainer = RandomForestTrainer(args.target, args.trees, args.seed, args.depth)

    if args.strategy == "split":
        trainer.train_with_split(data_path)
    elif args.strategy == "kfold":
        trainer.train_with_kfold(data_path)
    elif args.strategy == "groupkfold":
        trainer.train_with_groupkfold(data_path)

    trainer.check_feature_importance()
    trainer.save_model(f"model_{args.target}.pkl")


if __name__ == "__main__":
    main()

import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


class random_forest_with_kFold:
    def __init__(self, target, n_trees, seed):
        self.model = RandomForestRegressor(
            n_estimators=n_trees,
            random_state=seed,
            n_jobs=-1
        )
        self.model_columns = []
        self.target = target

    def train(self, data_path):
        print(f"Loading {data_path}")

        #To load the dataset
        try:
            if data_path.endswith(".xlsx"):
                df = pd.read_excel(data_path, engine="openpyxl")
            else:
                df = pd.read_csv(data_path)
        except Exception as e:
            print(f"File cannot be loaded: {e}")
            return

       
        df = df.dropna(subset=[self.target])

        #features
        features = [
            "num_cores",
            "total_ram",
            "avg_tracked_cpu_percent",
            "avg_ram_usage",
            "batch_size"
        ]

        #Check required columns
        missing = [c for c in features + [self.target] if c not in df.columns]
        if missing:
            print(f"Missing columns: {missing}")
            return

        X = df[features]
        y = df[self.target]

        self.model_columns = list(X.columns)


        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        r2_scores = []
        mae_scores = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = RandomForestRegressor(
                n_estimators=self.model.n_estimators,
                random_state=self.model.random_state,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)

            r2_scores.append(r2)
            mae_scores.append(mae)

            print(f"Fold {fold}: R²={r2:.3f}, MAE={mae:.1f}")

        print(f"Mean R² : {sum(r2_scores) / len(r2_scores):.3f}")
        print(f"Mean MAE ({self.target}): {sum(mae_scores) / len(mae_scores):.1f}")

    
        self.model.fit(X, y)

    def save_model(self, filename):
        os.makedirs("PKL", exist_ok=True)
        full_path = os.path.join("PKL", filename)
        payload = {
            "model": self.model,
            "columns": self.model_columns
        }
        joblib.dump(payload, full_path)
        print(f"Saved model to {full_path}")


if __name__ == "__main__":
    load_dotenv()
    path = os.getenv("TRAINING_DATA_PATH")

    parser = argparse.ArgumentParser(
        usage="python random_forrest.py --target steps_per_second --trees 100 --seed 95"
    )
    parser.add_argument("--target", type=str, default="steps_per_second")
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=95)

    args = parser.parse_args()

    if path:
        print(
            f"Configuration: "
            f"Target={args.target}, "
            f"Trees={args.trees}, "
            f"Seed={args.seed}"
        )

        algo = random_forest_with_kFold(args.target, args.trees, args.seed)
        algo.train(path)
        algo.save_model(f"model_{args.target}.pkl")

    else:
        print("You need to have the TRAINIG_DATA_PATH specified in .env file")

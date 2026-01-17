import argparse
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class LinearRegressionModel:
    def __init__(self, target):
        self.target = target
        self.model = LinearRegression() 
        self.model_columns = [] 

    def train(self, data_path):
        print(f"Loading {data_path}")
        
        try:
            if data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path, engine='openpyxl')
            else:
                df = pd.read_csv(data_path)
        except Exception as e:
            print(f"File cannot be loaded: {e}")
            print(f"Suggest to check TRAINING_DATA_PATH in your .env file.")
            return

        features = [
            'scene', 
            'batch_size', 
            'algorithm', 
            'num_cores', 
            'total_ram',
            'cpu_model',
            'hyper_learning_rate',
            'operating_system'
        ]
        

        missing = [c for c in features + [self.target] if c not in df.columns]
        if missing:
            print(f"Those columns are missing: {missing}")
            return

        X = df[features]
        y = df[self.target]

        # so this drop_first only applies to the categorical columns. i leaves all the numerical ones alone.
        # so the get_dummies puts it in a binary representation and the nthe drop_first drops the first one. 
        # This is needed for linear regression to avoid multicolinearity which basically means redundant info which will mess with the formula.
        X = pd.get_dummies(X, drop_first=True)

        self.model_columns = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Model training in progress...")
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        
        acc = r2_score(y_test, predictions)
        err = mean_absolute_error(y_test, predictions)

        # if R2 is negative we know its not linear so the random forest will be better. 
        print(f"Accuracy (R2), negative means the data is not linear: {acc:.2f}")
        print(f"Average Error: {err:.1f} seconds")

    def save_model(self, filename):
        payload = {
            'model': self.model,
            'columns': self.model_columns
        }
        joblib.dump(payload, filename)
        print(f"Saved model to {filename}")


if __name__ == "__main__":
    load_dotenv()
    path = os.getenv('TRAINING_DATA_PATH')

    parser = argparse.ArgumentParser(
        usage="python linear_regression.py --target <exact name of target column>",
    )
    
    parser.add_argument("--target", type=str, default="training_time_s", help="The column name to predict (default: training_time_s)")
    
    args = parser.parse_args()

    if path:
        print(f"Configuration: Target={args.target}")

        algo = LinearRegressionModel(target=args.target)
        algo.train(path)
        
        algo.save_model(f"linear_model_{args.target}.pkl")
    else:
        print("You need to have TRAINING_DATA_PATH specified in .env file")
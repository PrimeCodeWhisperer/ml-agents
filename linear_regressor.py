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

    def load_data(self, data_path):
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

        cols_to_check = features + [self.target]

        numeric_cols = ['batch_size', 'num_cores', 'total_ram', 'hyper_learning_rate']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_count = len(df)
        df = df.dropna(subset=cols_to_check)

        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            print(f"Skipped {dropped_count} rows containing missing (NaN) or incorrect values.")
        
        if df.empty:
            print("Error: No data left after cleaning! Check your dataset.")
            return


        X = df[features]
        y = df[self.target]

        # so this drop_first only applies to the categorical columns. i leaves all the numerical ones alone.
        # so the get_dummies puts it in a binary representation and the nthe drop_first drops the first one. 
        # This is needed for linear regression to avoid multicolinearity which basically means redundant info which will mess with the formula.
        X = pd.get_dummies(X, drop_first=True)

        self.model_columns = list(X.columns)

        return X, y

    def train(self, X,y):
    

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Model training in progress...")
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        
        r2_acc = r2_score(y_test, predictions)
        mae_err = mean_absolute_error(y_test, predictions)

        return r2_acc, mae_err

        
    def print_metrics(self, r2_acc, mae_err):
       
        print(f"Accuracy (R2), negative means the data is not linear: {r2_acc:.3f}")
        print(f"Average Error: {mae_err:.3f}")

        

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

        linear_model = LinearRegressionModel(target=args.target)
        X, y = linear_model.load_data(path)
        r2_acc, mae_err =linear_model.train(X, y)
        linear_model.print_metrics(r2_acc, mae_err)
        
        linear_model.save_model(f"linear_model_{args.target}.pkl")
    else:
        print("You need to have TRAINING_DATA_PATH specified in .env file")

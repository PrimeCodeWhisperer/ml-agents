import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class random_forrest:
    def __init__(self, target, n_trees, seed):
        #regressor is useful because we predict continuous time values
        self.model = RandomForestRegressor(n_estimators=n_trees, random_state=seed) #choose number of trees and random seed
        self.model_columns = [] 
        self.target = target

    def train(self, data_path):
        print(f"Loading {data_path}")
        
        #check for excel or csv
        try:
            if data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path, engine='openpyxl')
            else:
                df = pd.read_csv(data_path)
        except Exception as e:
            print(f"File cannot be loaded: {e}")
            print(f"Suggest to check TRAINING_DATA_PATH in your .env file.")
            return

        #features we want to use for training
        features = ['batch_size', 'num_cores', 'total_ram', 'scene', 'algorithm', 'total_steps']
        #output we want to predict
    

        #check is all columns exist
        missing = [c for c in features + [self.target] if c not in df.columns]
        if missing:
            print(f"Those columns are missing: {missing}")
            return

        X = df[features]
        y = df[self.target]

        #convert collumns into numbers
        X = pd.get_dummies(X)

        #save columns list so we can match the structure later when predicting
        self.model_columns = list(X.columns)

        #split data into  80% for training and 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Model training in progress...")
        self.model.fit(X_train, y_train)

        #check performance
        predictions = self.model.predict(X_test)
        r2_acc = r2_score(y_test, predictions)
        err = mean_absolute_error(y_test, predictions)

        print(f"Accuracy: {r2_acc * 100:.1f}%")
        print(f"Average Error ({self.target}): {err:.1f}")

    #save the model to file
    def save_model(self, filename):
        payload = {
            'model': self.model,
            'columns': self.model_columns
        }
        joblib.dump(payload, filename)
        print(f"Saved model to {filename}")


if __name__ == "__main__":
    #getting path to data
    load_dotenv()
    path = os.getenv('TRAINING_DATA_PATH')
    
    parser = argparse.ArgumentParser(
        usage="python random_forrest.py --target <exact name of target column> --trees <number of decision trees> --seed <random seed number for reproducibility>",
    )
    
    parser.add_argument("--target", type=str, default="training_time_s", help="The column name to predict (default: training_time_s)")
    parser.add_argument("--trees", type=int, default=100, help="Number of trees in the forest (default: 100)")
    parser.add_argument("--seed", type=int, default=95, help="Random seed for reproducibility (default: 95)")
    
    args = parser.parse_args()

    if path:
        print(f"Configuration: Target={args.target}, Trees={args.trees}, Seed={args.seed}")
        
        #initialize with arguments
        algo = random_forrest(args.target, args.trees, args.seed)
        algo.train(path)
        
        #save sklearn model (decision tree itself) to file. The file can be loaded later for predictions.
        algo.save_model(f"model_{args.target}.pkl")

    else:
        print("You need to have TRAINING_DATA_PATH specified in .env file")
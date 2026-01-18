import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class RandomForrestModel:
    def __init__(self, target, n_trees, seed, max_depth):
        #regressor is useful because we predict continuous time values
        self.model = RandomForestRegressor(
            n_estimators=n_trees, 
            random_state=seed,
            max_depth=max_depth) #choose number of trees and random seed
        self.model_columns = [] 
        self.target = target

    def load_data(self, data_path):
        
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
        #output we want to predict
    

        #check is all columns exist
        missing = [c for c in features + [self.target] if c not in df.columns]
        if missing:
            print(f"Those columns are missing: {missing}")
            return


        df = df.dropna(subset=[self.target]) #drop NaN rows
        X = df[features]
        y = df[self.target]

        #convert collumns into numbers
        X = pd.get_dummies(X)

        #save columns list so we can match the structure later when predicting
        self.model_columns = list(X.columns)
        return X,y



    def train(self, X, y):
        
        #split data into  80% for training and 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Model training in progress...")
        self.model.fit(X_train, y_train)

        #check performance
        predictions = self.model.predict(X_test)
        return y_test, predictions
       
    
    def evaluate_predictions(self, y_test, predictions):
        r2_acc = r2_score(y_test, predictions)
        mae_err = mean_absolute_error(y_test, predictions)

        return r2_acc, mae_err
    
    def print_metrics(self, r2_acc, mae_err):
        print(f"r2_score: {r2_acc:.3f}")
        print(f"Average Error ({self.target}): {mae_err:.1f}")


    def check_feature_importance(self):
        if not hasattr(self.model, 'feature_importances_'):
            print("Error: This model doesn't support feature importance.")
            return

        # 1. Get raw importances
        raw_importances = pd.DataFrame({
            'Feature': self.model_columns,
            'Importance': self.model.feature_importances_
        })

        # 2. Define your original feature groups
        # These must match the names in your 'features' list exactly
        base_features = [
            #'scene', 
            'batch_size', 
            #'algorithm', 
            'num_cores', 
            'total_ram',
            'cpu_model',
            'hyper_learning_rate',
            'operating_system'
        ]

        aggregated_data = []

        # 3. Sum up the scores for each group
        for base in base_features:
            # We look for columns that start with the base name (for categorical)
            # OR match exactly (for numerical like 'batch_size')
            
            # This filter finds all dummy columns belonging to this feature
            # e.g. finds 'cpu_model_i386' and 'cpu_model_arm' for 'cpu_model'
            mask = raw_importances['Feature'].str.startswith(base)
            
            # Calculate total importance for this base feature
            total_score = raw_importances.loc[mask, 'Importance'].sum()
            
            aggregated_data.append({'Feature': base, 'Importance': total_score})

        # 4. Create the final clean table
        agg_df = pd.DataFrame(aggregated_data)
        agg_df = agg_df.sort_values(by='Importance', ascending=False)

        print("\n" + "="*40)
        print(" AGGREGATED FEATURE IMPORTANCE")
        print("="*40)
        # Formats it as a percentage (e.g., 0.48 -> 48.0%)
        print(agg_df.to_string(index=False, formatters={'Importance': '{:.1%}'.format}))
        print("="*40)

        # Optional: Print raw top contributor just in case
        print("\n(Note: Aggregated from specific One-Hot encoded columns)")

        

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
    parser.add_argument("--trees", type=int, default=1000, help="Number of trees in the forest (default: 100)")
    parser.add_argument("--seed", type=int, default=95, help="Random seed for reproducibility (default: 95)")
    parser.add_argument("--depth", type=int, default=None, help="Max depth of trees (default: None/Unlimited)")

    args = parser.parse_args()

    if path:
        print(f"Configuration: Target={args.target}, Trees={args.trees}, Seed={args.seed}, Max_depth={args.depth}")
        
        #initialize with arguments
        forest_model = RandomForrestModel(args.target, args.trees, args.seed, args.depth)
        X,y = forest_model.load_data(path)
        y_test, predictions = forest_model.train(X,y)
        r2_acc, mae_error = forest_model.evaluate_predictions(y_test, predictions)
        forest_model.print_metrics(r2_acc, mae_error)

        forest_model.check_feature_importance()
        
        #save sklearn model (decision tree itself) to file. The file can be loaded later for predictions.
        forest_model.save_model(f"model_{args.target}.pkl")

    else:
        print("You need to have TRAINING_DATA_PATH specified in .env file")

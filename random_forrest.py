import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class random_forrest:
    def __init__(self):
        #regressor is useful because we predict continuous time values
        self.model = RandomForestRegressor(n_estimators=100, random_state=95) #choose number of trees and random seed
        self.model_columns = [] 

    def train(self, data_path):
        print(f"Loading data from {data_path}...")
        
        #specific check for excel vs csv
        try:
            if data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path, engine='openpyxl')
            else:
                df = pd.read_csv(data_path)
        except Exception as e:
            print(f"Could not load file: {e}")
            return

        #define what we want to use for training
        features = ['batch_size', 'num_cores', 'total_ram', 'scene', 'algorithm']
        target = 'training_time_s'

        #basic validation to make sure columns exist
        missing = [c for c in features + [target] if c not in df.columns]
        if missing:
            print(f"Missing columns in dataset: {missing}")
            return

        X = df[features]
        y = df[target]

        #convert text columns (like 'scene') into numbers
        X = pd.get_dummies(X)

        #save columns list so we can match the structure later when predicting
        self.model_columns = list(X.columns)

        #split: 80% for training, 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training model...")
        self.model.fit(X_train, y_train)

        #check how well it did
        predictions = self.model.predict(X_test)
        acc = r2_score(y_test, predictions)
        err = mean_absolute_error(y_test, predictions)

        print("Done.")
        print(f"Accuracy (R2): {acc:.2f}")
        print(f"Avg Error: +/- {err:.1f} seconds")

    def save_model(self, filename="time_predictor_model.pkl"):
        #save both the model and column layout
        payload = {
            'model': self.model,
            'columns': self.model_columns
        }
        joblib.dump(payload, filename)
        print(f"Saved model to {filename}")

    def load_model(self, filename="time_predictor_model.pkl"):
        try:
            data = joblib.load(filename)
            self.model = data['model']
            self.model_columns = data['columns']
            print(f"Loaded model from {filename}")
        except FileNotFoundError:
            print("Model file not found.")

    def predict_new(self, batch_size, cores, ram, scene, algorithm):
        if not self.model_columns:
            print("Model not ready.")
            return None

        #create a single row dataframe for the new input
        input_row = pd.DataFrame([{
            'batch_size': batch_size,
            'num_cores': cores,
            'total_ram': ram,
            'scene': scene,
            'algorithm': algorithm
        }])

        #convert to numbers and align columns with the trained model
        input_row = pd.get_dummies(input_row)
        input_row = input_row.reindex(columns=self.model_columns, fill_value=0)

        return self.model.predict(input_row)[0]


if __name__ == "__main__":
    #getting path to data
    load_dotenv()
    path = os.getenv('TRAINING_DATA_PATH')

    if path:
        ai = random_forrest()
        ai.train(path)
        ai.save_model("trained_data.pkl")

    else:
        print("You need to have TRAINING_DATA_PATH specified in .env file")
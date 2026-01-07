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
        print(f"Loading {data_path}")
        
        #check for excel or csv
        try:
            if data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path, engine='openpyxl')
            else:
                df = pd.read_csv(data_path)
        except Exception as e:
            print(f"File cannot be loaded: {e}")
            return

        #features we want to use for training
        features = ['batch_size', 'num_cores', 'total_ram', 'scene', 'algorithm']
        #output we want to predict
        target = 'training_time_s'

        #check is all columns exist
        missing = [c for c in features + [target] if c not in df.columns]
        if missing:
            print(f"Those columns are missing: {missing}")
            return

        X = df[features]
        y = df[target]

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
        acc = r2_score(y_test, predictions)
        err = mean_absolute_error(y_test, predictions)

        print(f"Accuracy: {acc:.2f}")
        print(f"Average Error: {err:.1f} seconds")

    def save_model(self, filename="trained_data.pkl"):
        payload = {
            'model': self.model,
            'columns': self.model_columns
        }
        joblib.dump(payload, filename)
        print(f"Saved model to {filename}")

    def load_model(self, filename="trained_data.pkl"):
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

        #convert to numbers
        input_row = pd.get_dummies(input_row)
        input_row = input_row.reindex(columns=self.model_columns, fill_value=0)

        return self.model.predict(input_row)[0]


if __name__ == "__main__":
    #getting path to data
    load_dotenv()
    path = os.getenv('TRAINING_DATA_PATH')

    if path:
        algo = random_forrest()
        algo.train(path)
        algo.save_model("trained_data.pkl")

    else:
        print("You need to have TRAINING_DATA_PATH specified in .env file")
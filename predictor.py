import joblib
import pandas as pd

class Predictor:
    def __init__(self, model_path="trained_data.pkl"):
        data = joblib.load(model_path)
        self.model = data["model"]
        self.model_columns = data["columns"]

    def predict(self, batch_size, num_cores, total_ram, learning_rate, total_steps):
        input_df = pd.DataFrame([{
            "batch_size": batch_size,
            "num_cores": num_cores,
            "total_ram": total_ram,
            "total_steps": total_steps,
            "learning_rate": learning_rate
            
        }])

        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=self.model_columns, fill_value=0)
        prediction = self.model.predict(input_df)[0]

        print(f"Batch size: {batch_size}")
        print(f"CPU cores: {num_cores}")
        print(f"Total RAM: {total_ram}")
        print(f"Total steps: {total_steps}")
        print(f"Learning rate: {learning_rate}")
        print(f"Estimated training time: {prediction:.1f} seconds")

        return prediction
if __name__ == "__main__":
    predictor = Predictor("trained_data.pkl")

    predictor.predict(
        batch_size=1000000,
        num_cores=12,
        total_ram=24,
        total_steps=394000,
        learning_rate= 0.0004
    )

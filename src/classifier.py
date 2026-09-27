import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model(csv_path="data/outputs/results_summary.csv", save_path="first_touch_model.pkl"):
    print("Loading data...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Please run `main.py` first.")
        return

    # Filter out errors or extremely weird values
    # For some videos, speed ratio is massive due to missing touch frame tracking
    df = df[df['speed_ratio'] < 100]

    # Map categories to "Good", "Average", "Poor"
    # Given the data, 'good_touch' and 'pressure_good' represent positive examples
    # 'poor_touch' represents negative.
    def map_category(cat):
        if 'good' in cat:
            return "Good"
        elif 'poor' in cat:
            return "Poor"
        return "Average"

    df['label'] = df['category'].apply(map_category)

    # Features for the model
    features = ['score', 'speed_ratio', 'direction_change', 'control_distance', 'pressure_factor']
    X = df[features]
    y = df['label']

    if len(X) < 10:
        print("Not enough valid data points to train a model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {save_path}")

def predict_quality(metrics_dict, model_path="first_touch_model.pkl"):
    """
    Predicts the quality class (Good/Poor) given a dictionary of metrics.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return "Unknown (Model not trained)"

    # Must match the order of `features` array above
    features = ['score', 'speed_ratio', 'direction_change', 'control_distance', 'pressure_factor']
    input_data = [metrics_dict.get(f, 0) for f in features]
    
    pred = model.predict([input_data])
    return pred[0]

if __name__ == "__main__":
    train_model()

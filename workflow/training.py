import joblib # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from preprocessing import load_data, preprocess_data # type: ignore

# Train and evaluate the model
def train_model(data):
    X = data.drop("target", axis=1)
    y = data["target"]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a RandomForestClassifier with hyperparameter tuning
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, "iris_model.pkl")
    print("Model saved as 'iris_model.pkl'.")
    
    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    
    return model

if __name__ == "__main__":
    # Load and preprocess your data here
    data = load_data()  # Assuming this function loads your dataset
    data = preprocess_data(data)  # Assuming this function preprocesses your dataset
    train_model(data)
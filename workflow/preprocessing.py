import pandas as pd
from sklearn.datasets import load_iris # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# Load the Iris dataset
def load_data():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data

# Preprocess the data
def preprocess_data(data):
    # Add a new feature: the ratio of sepal to petal length
    data['sepal_petal_ratio'] = data['sepal length (cm)'] / data['petal length (cm)']
    return data

# Standardize features (exclude the 'target' column and 'epal_petal_ratio')
def standardize_features(data):
    scaler = StandardScaler()
    features = data.drop(columns=['target', 'epal_petal_ratio'])
    scaled_features = scaler.fit_transform(features)
    data[features.columns] = scaled_features
    return data

# Ensure that the target variable is of type integer (it's categorical)
def ensure_integer_target(data):
    data['target'] = data['target'].astype(int)
    return data

# Example usage
if __name__ == "__main__":
    data = load_data()
    preprocessed_data = preprocess_data(data)
    standardized_data = standardize_features(preprocessed_data)
    cleaned_data = ensure_integer_target(standardized_data)
    cleaned_data.to_csv("cleaned_data.csv", index=False)
    print("Preprocessed data saved as 'cleaned_data.csv'")
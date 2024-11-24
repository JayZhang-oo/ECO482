import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def split_data(data, target_column, timestamp_column, test_size=0.2):
    """
    Split data into training and testing sets while maintaining 
    temporal order with timestamp_column
    """
    data = data.sort_values(by = timestamp_column)
    split_index = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    X_train, y_train = train_data.drop(columns=[target_column, timestamp_column]), train_data[target_column]
    X_test, y_test = test_data.drop(columns=[target_column, timestamp_column]), test_data[target_column]
    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and compare Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting models
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)  
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            "Model": model,
            "Accuracy": accuracy,
            "Report": classification_report(y_test, y_pred)
        }
        print(f"{name} Accuracy: {accuracy}\n")
        print(classification_report(y_test, y_pred))
    return results
    
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and show the metrics
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", report)
    return y_pred



if __name__ == "__main__":
    file_path= None #enter the file path
    data = pd.read_csv(file_path)
    target_column = None #enter the name of the target column
    timestamp = None #eneter the name of timestamp column
    #################################
    X_train, X_test, y_train, y_test = split_data(data, target_column, timestamp)
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    best_model = max(results, key=lambda name: results[name]["Accuracy"])
    print(f"Best Model: {best_model}")

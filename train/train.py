import mlflow
import mlflow.sklearn
import optuna
from optuna.pruners import HyperbandPruner
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configure MLflow
mlflow.set_tracking_uri("http://ec2-184-72-140-253.compute-1.amazonaws.com:5000")
mlflow.set_experiment("RandomForest_Classification")

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)

    # Create the RandomForest model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Log the metrics and model
    with mlflow.start_run():
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('min_samples_split', min_samples_split)
        mlflow.log_param('min_samples_leaf', min_samples_leaf)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.sklearn.log_model(model, "random_forest_model")

    return accuracy

# Create and run the study with Optuna
study = optuna.create_study(direction='maximize', pruner=HyperbandPruner())
study.optimize(objective, n_trials=100)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)


best_model_params = study.best_params
best_model = RandomForestClassifier(**best_model_params)
best_model.fit(X_train, y_train)

mlflow.sklearn.log_model(best_model, "best_random_forest_model", registered_model_name="RandomForestClassifier")
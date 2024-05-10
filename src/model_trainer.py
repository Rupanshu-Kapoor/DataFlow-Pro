#  Contains classes and functions for model 
# building, hyperparameter tuning, and training models.


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump  # For saving models

class ModelTrainer:
    def __init__(self, json_content: dict):
        self.json_content = json_content
    
    def build_and_tune_model(self, X_train, y_train, problem_type: str):
        """Builds and tunes a model according to the problem type."""
        # Choose model based on problem type
        if problem_type == 'classification':
            model = RandomForestClassifier()
        else:  # 'regression'
            model = RandomForestRegressor()
        
        # Define hyperparameters and grid for tuning
        param_grid = self.json_content.get('hyperparameters', {})
        
        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        
        # Get the best model from the grid search
        best_model = grid_search.best_estimator_
        
        return best_model
    
    def save_model(self, model, file_path: str):
        """Saves the trained model to a file."""
        dump(model, file_path)

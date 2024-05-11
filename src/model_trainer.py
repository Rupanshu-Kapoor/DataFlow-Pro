#  Contains classes and functions for model 
# building, hyperparameter tuning, and training models.

import numpy as np
from sklearn.model_selection import GridSearchCV
from joblib import dump  # For saving models
from config import model_dict

class ModelTrainer:
    def __init__(self, json_content: dict):
        self.json_content = json_content
        self.k_fold = json_content["design_state_data"]["train"]["k_fold"] 
    
    
    
    def build_and_tune_model(self, X_train, y_train, problem_type, selected_models, model_parameters):
        
        for model_name in selected_models:
            if model_name == "SVM" and problem_type == "Regression":
                model = model_dict["SVR"]
            elif model_name == "SVM" and problem_type == "Classification":
                model = model_dict["SVC"]
            else:
                model = model_dict[model_name]

            if (model_name == "RandomForestClassifier" or model_name == "RandomForestRegressor"):
                best_params = self.tune_random_forest(model, X_train, y_train, model_name, model_parameters)
            
            elif (model_name == "LinearRegression" or model_name == "LogisticRegression" 
                  or model_name == "ElasticNetRegression"):
                best_params = self.tune_linear_logistic_elasticnet_regression(model, X_train, y_train, model_name, model_parameters)
            
            elif (model_name == "RidgeRegression" or model_name == "LassoRegression"):
                best_params = self.tune_ridge_lasso_regression(model, X_train, y_train, model_name, model_parameters)
            
            elif (model_name == "DecisionTreeRegressor" or model_name == "DecisionTreeClassifier"):
                best_params = self.tune_decision_tree(model, X_train, y_train, model_name, model_parameters)
                        
            elif model_name == "SVM":
                best_params = self.tune_svm(model, X_train, y_train, model_parameters)
            
            elif model_name == "KNN":
                best_params = self.tune_knn(model, X_train, y_train, model_parameters)
            
            elif model_name == "neural_network":
                best_params = self.tune_neural_network(model, X_train, y_train, model_parameters)
        
        
        
        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        
        # Get the best model from the grid search
        best_model = grid_search.best_estimator_
        
        return best_model
    
    
    def save_model(self, model, file_path: str):
        """Saves the trained model to a file."""
        dump(model, file_path)

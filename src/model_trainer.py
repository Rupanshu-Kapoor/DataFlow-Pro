#  Contains classes and functions for model 
# building, hyperparameter tuning, and training models.

import numpy as np
from sklearn.model_selection import GridSearchCV
from joblib import dump  # For saving models
from config import model_dict
import streamlit as st
class ModelTrainer:
    def __init__(self, json_content: dict):
        self.json_content = json_content
        self.k_fold = json_content["design_state_data"]["train"]["k_fold"]
        if not self.k_fold:
            self.k_fold = None
        self.random_state = [42]
        self.num_iter = 3
        
    
    def tune_random_forest(self, model, X_train, y_train, model_name, model_parameters):
        params = {"random_state": self.random_state}
        min_trees = model_parameters[model_name]["min_trees"]
        max_trees = model_parameters[model_name]["max_trees"]
        params["n_estimators"] = np.linspace(min_trees, max_trees, self.num_iter, dtype=int)

        min_depth = model_parameters[model_name]["min_depth"]
        max_depth = model_parameters[model_name]["max_depth"]
        params["max_depth"] = np.linspace(min_depth, max_depth, self.num_iter, dtype=int)

        min_samples_per_leaf = model_parameters[model_name]["min_samples_per_leaf_min_value"]
        max_samples_per_leaf = model_parameters[model_name]["min_samples_per_leaf_max_value"]
        params["min_samples_leaf"] = np.linspace(min_samples_per_leaf, max_samples_per_leaf, self.num_iter, dtype=int)

        if model_parameters[model_name].get("random_state"):
            params["random_state"] = model_parameters[model_name]["random_state"]

        gcv = GridSearchCV(model, params, cv=self.k_fold) 
        gcv.fit(X_train, y_train)
        return gcv.best_estimator_

    def tune_linear_elasticnet_regression(self, model, X_train, y_train, model_name, model_parameters):
        params = {"random_state": self.random_state}
        if model_parameters[model_name].get("random_state"):
            params["random_state"] = model_parameters[model_name]["random_state"]

        min_iter = model_parameters[model_name]["min_iter"]
        max_iter = model_parameters[model_name]["max_iter"]
        params["max_iter"] = np.linspace(min_iter, max_iter, self.num_iter, dtype=int)

        min_reg = model_parameters[model_name]["min_regparam"]
        max_reg = model_parameters[model_name]["max_regparam"]
        params["alpha"] = np.logspace(min_reg, max_reg, self.num_iter)

        min_elasticnet = model_parameters[model_name]["min_elasticnet"]
        max_elasticnet = model_parameters[model_name]["max_elasticnet"]
        params["l1_ratio"] = np.linspace(min_elasticnet, max_elasticnet, self.num_iter)

        gcv = GridSearchCV(model, params, cv=self.k_fold)
        gcv.fit(X_train, y_train)
        return gcv.best_estimator_

  
    
    def tune_logistic_regression(self, model, X_train, y_train, model_parameters):
        params = {"random_state": self.random_state}
        if model_parameters["LogisticRegression"].get("random_state"):
            params["random_state"] = model_parameters["LogisticRegression"]["random_state"]
        
        min_iter = model_parameters["LogisticRegression"]["min_iter"]
        max_iter = model_parameters["LogisticRegression"]["max_iter"]
        params["max_iter"] = np.linspace(min_iter, max_iter, self.num_iter, dtype=int)

        min_reg = model_parameters["LogisticRegression"]["min_regparam"]
        max_reg = model_parameters["LogisticRegression"]["max_regparam"]
        params["C"] = np.logspace(min_reg, max_reg, self.num_iter)

        min_elasticnet = model_parameters["LogisticRegression"]["min_elasticnet"]
        max_elasticnet = model_parameters["LogisticRegression"]["max_elasticnet"]
        params["l1_ratio"] = np.linspace(min_elasticnet, max_elasticnet, self.num_iter)

        gcv = GridSearchCV(model, params, cv=self.k_fold)
        gcv.fit(X_train, y_train)
        return gcv.best_estimator_

    def tune_ridge_lasso_regression(self, model, X_train, y_train, model_name, model_parameters):
        params = {"random_state": self.random_state}
        if model_parameters[model_name].get("random_state"):
            params["random_state"] = model_parameters[model_name]["random_state"]
        
        min_iter = model_parameters[model_name]["min_iter"]
        max_iter = model_parameters[model_name]["max_iter"]
        params["max_iter"] = np.linspace(min_iter, max_iter, self.num_iter, dtype=int)

        min_regparam = model_parameters[model_name]["min_regparam"]
        max_regparam = model_parameters[model_name]["max_regparam"]
        params["alpha"] = np.logspace(min_regparam, max_regparam, self.num_iter)

        gcv = GridSearchCV(model, params, cv=self.k_fold)
        gcv.fit(X_train, y_train)
        return gcv.best_estimator_


    def tune_decision_tree(self, model, X_train, y_train, model_name, model_parameters):
        params = {"random_state": self.random_state}
        if model_parameters[model_name].get("random_state"):
            params["random_state"] = model_parameters[model_name]["random_state"]
        
        min_depth = model_parameters[model_name]["min_depth"]
        max_depth = model_parameters[model_name]["max_depth"]
        params["max_depth"] = np.linspace(min_depth, max_depth, self.num_iter, dtype=int)

        criterion = []
        if model_parameters[model_name]["use_gini"]:
            criterion.append("gini")
        if model_parameters[model_name]["use_entropy"]:
            criterion.append("entropy")
        params["criterion"] = criterion

        splitter = []
        if model_parameters[model_name]["use_random"]:
            splitter.append("random")
        if model_parameters[model_name]["use_best"]:
            splitter.append("best")
        params["splitter"] = splitter

        if model_parameters[model_name].get("min_samples_per_leaf"):
            params["min_samples_leaf"] = model_parameters[model_name]["min_samples_per_leaf"]

        gcv = GridSearchCV(model, params, cv=self.k_fold)
        gcv.fit(X_train, y_train)
        return gcv.best_estimator_
    
    def tune_svm(self, model, X_train, y_train, model_parameters):
        params = {}
        
        kernel = []
        if model_parameters["SVM"]["linear_kernel"]:
            kernel.append("linear")
        if model_parameters["SVM"]["rep_kernel"]:
            kernel.append("rbf")
        if model_parameters["SVM"]["polynomial_kernel"]:
            kernel.append("poly")
        if model_parameters["SVM"]["sigmoid_kernel"]:
            kernel.append("sigmoid")
        params["kernel"] = kernel

        params["C"] = model_parameters["SVM"]["c_value"]

        gamma = []
        if model_parameters["SVM"]["scale"]:
            gamma.append("scale")
        if model_parameters["SVM"]["auto"]:
            gamma.append("auto")

        params["gamma"] = gamma
        
        params["max_iter"] = model_parameters["SVM"]["max_iterations"]
        params["tol"] = model_parameters["SVM"]["tolerance"]

        gcv = GridSearchCV(model, params, cv=self.k_fold)
        gcv.fit(X_train, y_train)
        return gcv.best_estimator_

    def tune_knn(self, model, X_train, y_train, model_parameters):
        params = {}
        
        params["n_neighbors"] = model_parameters["KNN"]["k_value"]

        if model_parameters["KNN"].get("distance_weighting"):
            params["weights"] = ["distance"]

        if model_parameters["KNN"]["neighbour_finding_algorithm"] == "Automatic":
            params["algorithm"] = "auto"
        
        gcv = GridSearchCV(model, params, cv=self.k_fold)
        gcv.fit(X_train, y_train)
        return gcv.best_estimator_
        pass


    def tune_neural_network(self, model, X_train, y_train, model_parameters):
        parameters = model_parameters["neural_network"]
        params = {"random_state": self.random_state,
                  "hidden_layer_sizes": parameters["hidden_layer_sizes"],
                  "alpha": parameters["alpha_value"],
                  "max_iter": parameters["max_iterations"],
                  "tol": parameters["convergence_tolerance"],
                  "early_stopping": parameters["early_stopping"],
                  "solver": parameters["solver"],
                  "shuffle": parameters["shuffle_data"],
                  "learning_rate_init": parameters["initial_learning_rate"],
                  "batch_size": parameters["automatic_batching"],
                  "beta_1": parameters["beta_1"],
                  "beta_2": parameters["beta_2"],
                  "epsilon": parameters["epsilon"],
                  "power_t": parameters["power_t"],
                  "momentum": parameters["momentum"],
                  "nesterovs_momentum": parameters["use_nesterov_momentum"],
                  } 
        
        if parameters.get("random_state"):
            params["random_state"] = parameters["random_state"]
        
        if parameters.get("activation"):
            params["activation"] = parameters["activation"]
        
        gcv = GridSearchCV(model, params, cv=self.k_fold)
        gcv.fit(X_train, y_train)
        return gcv.best_estimator_
    
    def tune_xgb(self, model, X_train, y_train, model_name, model_parameters):
        params = {"random_state": self.random_state,
                  "booster": []
                  }
        if model_parameters["xg_boost"].get("random_state"):
            params["random_state"] = model_parameters["xg_boost"]["random_state"]

        if model_parameters["xg_boost"].get("use_gradient_boosted_tree"):
            params["booster"].append("gbtree")

        if model_parameters["xg_boost"].get("dart"):
            params["booster"].append("dart")

        params["n_estimators"] = model_parameters["xg_boost"]["max_num_of_trees"]
        params["tree_method"] = model_parameters["xg_boost"]["tree_method"]
        if model_parameters["xg_boost"]["early_stopping"]:
            params["early_stopping_rounds"] = model_parameters["xg_boost"]["early_stopping_rounds"]

        params["max_depth"] = model_parameters["xg_boost"]["max_depth_of_tree"]
        params["learning_rate"] = model_parameters["xg_boost"]["learningRate"]
        params["l1_regularization"] = model_parameters["xg_boost"]["l1_regularization"]
        params["l2_regularization"] = model_parameters["xg_boost"]["l2_regularization"]
        params["min_child_weight"] = model_parameters["xg_boost"]["min_child_weight"]
        params["gamma"] = model_parameters["xg_boost"]["gamma"]
        params["sub_sample"] = model_parameters["xg_boost"]["sub_sample"]
        params["col_sample_by_tree"] = model_parameters["xg_boost"]["col_sample_by_tree"]

        gcv = GridSearchCV(model, params, cv=self.k_fold)
        gcv.fit(X_train, y_train)
        return gcv.best_estimator_

    
    def build_and_tune_model(self, X_train, y_train, problem_type, selected_models, model_parameters):
        self.best_models = {}
        for model_name in selected_models:
            if model_name == "xg_boost":
                st.warning("As of now xg_boost is not supported")
                continue
            if model_name == "SVM" and problem_type == "Regression":
                model = model_dict["SVMRegressor"]
                best_model = self.tune_svm(model, X_train, y_train, model_parameters)

            elif model_name == "SVM" and problem_type == "Classification":
                model = model_dict["SVMClassifier"]
                best_model = self.tune_svm(model, X_train, y_train, model_parameters)

            elif model_name == "KNN" and problem_type == "Regression":
                model = model_dict["KNNRegressor"]
                best_model = self.tune_knn(model, X_train, y_train, model_parameters)

            elif model_name == "KNN" and problem_type == "Classification":
                model = model_dict["KNNClassifier"]
                best_model = self.tune_knn(model, X_train, y_train, model_parameters)

            elif model_name == "neural_network" and problem_type == "Regression":
                model = model_dict["neural_network"]
                best_model = self.tune_neural_network(model, X_train, y_train, model_parameters)

            elif model_name == "neural_network" and problem_type == "Classification":
                model = model_dict["neural_network"]
                best_model = self.tune_neural_network(model, X_train, y_train, model_parameters)

            elif model_name == "xg_boost" and problem_type == "Regression":
                model = model_dict["XGBoostRegressor"]
                best_model = self.tune_xgb(model, X_train, y_train, model_name, model_parameters)

            elif model_name == "xg_boost" and problem_type == "Classification":
                model = model_dict["XGBoostClassifier"]
                best_model = self.tune_xgb(model, X_train, y_train, model_name, model_parameters)
            else:
                model = model_dict[model_name]

            if (model_name == "RandomForestClassifier" or model_name == "RandomForestRegressor"):
                best_model = self.tune_random_forest(model, X_train, y_train, model_name, model_parameters)
            
            elif (model_name == "LinearRegression" or  model_name == "ElasticNetRegression"):
                best_model = self.tune_linear_elasticnet_regression(model, X_train, y_train, model_name, model_parameters)
            
            elif model_name == "LogisticRegression":
                best_model = self.tune_logistic_regression(model, X_train, y_train, model_parameters)

            elif (model_name == "RidgeRegression" or model_name == "LassoRegression"):
                best_model = self.tune_ridge_lasso_regression(model, X_train, y_train, model_name, model_parameters)
            
            elif (model_name == "DecisionTreeRegressor" or model_name == "DecisionTreeClassifier"):
                best_model = self.tune_decision_tree(model, X_train, y_train, model_name, model_parameters)
                        
            self.best_models[model_name] = best_model        
        
        
        return self.best_models
    
    

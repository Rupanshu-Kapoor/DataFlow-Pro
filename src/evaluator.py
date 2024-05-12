#  Contains classes and functions for evaluating trained 
# models using the specified metrics.


from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score, mean_squared_error,classification_report, confusion_matrix, ConfusionMatrixDisplay
import json
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

class Evaluator:
    def __init__(self, json_content, problem_type, target_variable):
        self.json_content = json_content
        self.problem_type = problem_type
    
    def evaluate_model(self, models, X_test, y_test):
        """Evaluates the model using specified metrics and returns results."""
        for model_name, model in models.items():
            print(f"Evaluating model: {model_name}")
            predictions = model.predict(X_test)
        
            # Choose evaluation metrics based on problem type
            st.subheader(f"Model: {model_name}")
            if self.problem_type == 'Classification':
                self.log_confusion_matrix(y_test, predictions, model_name, model)
                self.log_classification_report(y_test, predictions, model_name)
            else:  # 'regression'
                self.log_rmse(y_test, predictions, model_name)
                self.log_r2(y_test, predictions, model_name)
                self.log_adj_r2(X_test,y_test, predictions, model_name)
            
        # return metrics
    
    def save_metrics(self, metrics, file_path: str):
        """Saves evaluation metrics to a file."""
        with open(file_path, 'w') as file:
            json.dump(metrics, file)


    def log_confusion_matrix(self, y_test, predictions, model_name, model):
        """Logs the confusion matrix."""
        cm = confusion_matrix(y_test, predictions, labels=model.classes_)
        # st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown(f"#### Confusion matrix for : {model_name} ")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=ax)
        st.pyplot(fig)

    
    def log_classification_report(self, y_test, predictions, model_name):
        """Logs the classification report."""
        st.markdown(f"#### Classification report for: {model_name} ")
        accuracy = accuracy_score(y_test, predictions)
        cr = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(cr).T
        report_df = report_df.rename(columns={'precision': 'Precision', 'recall': 'Recall', 'f1-score': 'F1-Score', 'support': 'Support'})
        st.table(report_df)


    def log_rmse(self, y_test, predictions, model_name):
        """Logs the root mean squared error."""
        rmse = root_mean_squared_error(y_test, predictions)
        print(f"RMSE for model: {model_name}")
        print(rmse)

    def log_r2(self, y_test, predictions, model_name):
        """Logs the R-squared score."""
        r2 = r2_score(y_test, predictions)
        print(f"R-squared score for model: {model_name}")
        print(r2)

    def log_adj_r2(self,X_test, y_test, predictions, model_name):
        """Logs the adjusted R-squared score."""
        sample_size, n_variables = X_test.shape
        r2 = r2_score(y_test, predictions)
        adj_r2 = 1 - ((1 - r2) * (sample_size - 1)) / (sample_size - n_variables - 1)
        print(f"Adjusted R-squared score for model: {model_name}")
        print(adj_r2)
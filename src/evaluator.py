#  Contains classes and functions for evaluating trained 
# models using the specified metrics.


from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score, mean_squared_error,classification_report, confusion_matrix, ConfusionMatrixDisplay
import json
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

class Evaluator:
    def __init__(self, json_content, problem_type, target_variable):
        self.json_content = json_content
        self.problem_type = problem_type
    
    def evaluate_model(self, models, X_test, y_test):
        """Evaluates the model using specified metrics and returns results."""
        metrics = {}
        for model_name, model in models.items():
            metrics[model_name] = {}
            print(f"Evaluating model: {model_name}")
            predictions = model.predict(X_test)
        
            # Choose evaluation metrics based on problem type
            st.subheader(f"Model: {model_name}")
            if self.problem_type == 'Classification':
                self.log_confusion_matrix(y_test, predictions, model_name, model)
                accuracy = self.log_classification_report(y_test, predictions, model_name)
                metrics[model_name]["accuracy"] = accuracy
            else:  # 'regression'
                rmse_score = self.log_rmse(y_test, predictions, model_name)
                r2_score = self.log_r2(y_test, predictions, model_name)
                adj_r2_score = self.log_adj_r2(X_test,y_test, predictions, model_name)
                metrics[model_name]["rmse"] = rmse_score
                metrics[model_name]["r2"] = r2_score
                metrics[model_name]["adj_r2"] = adj_r2_score
                
        return metrics
            
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
        return round(accuracy,2)

    def log_rmse(self, y_test, predictions, model_name):
        """Logs the root mean squared error."""
        rmse = root_mean_squared_error(y_test, predictions)
        st.markdown(f"RMSE for {model_name}: {round(rmse,2)} ")
        return round(rmse,2)

    def log_r2(self, y_test, predictions, model_name):
        """Logs the R-squared score."""
        r2 = r2_score(y_test, predictions)
        st.markdown(f"R-squared score for {model_name}: {round(r2,2)} ")
        return round(r2,2)


    def log_adj_r2(self,X_test, y_test, predictions, model_name):
        """Logs the adjusted R-squared score."""
        sample_size, n_variables = X_test.shape
        r2 = r2_score(y_test, predictions)
        adj_r2 = 1 - ((1 - r2) * (sample_size - 1)) / (sample_size - n_variables - 1)
        print(f" model: {model_name}")
        st.markdown(f"Adjusted R-squared score for {model_name}: {round(r2,2)} ")
        return round(adj_r2,2)

    def display_metrics(self, metrics):
        
        available_metrics = list(next(iter(metrics.values())).keys())
        available_models = list(metrics.keys())

        num_models = len(available_models)
        hue_colors = plt.cm.tab10(np.linspace(0, 1, num_models))

        for metric in available_metrics:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            for i, model in enumerate(available_models):
                metric_value = metrics[model][metric]
                bar = ax.bar(model, metric_value, color=hue_colors[i], label=model)

                for rect in bar:
                    height = rect.get_height()
                    ax.annotate('{}'.format(round(height, 2)),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom')
            
            ax.set_xlabel('Algorithm Models')
            ax.set_ylabel(metric.upper())
            ax.legend()
            plt.xticks(rotation=45)
            plt.title(f'{metric.upper()}')
            st.pyplot(fig)


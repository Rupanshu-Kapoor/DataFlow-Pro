# Contains the logic for reading and parsing the RTF file and extracting JSON content
import pandas as pd
import json
from striprtf.striprtf import rtf_to_text
from sklearn.model_selection import train_test_split


class DataReader:
    def rtf_parser(self, file_path, encoding='utf-8'):
        # Read the RTF file
        with open(file_path, 'r', encoding=encoding) as file:
            rtf_content = file.read()
        
        # Convert the RTF content to text
        text_content = rtf_to_text(rtf_content)
        
        return text_content
    

    def rtf_to_json_parser(self, rtf_file_path):
        plain_text = self.rtf_parser(rtf_file_path)
        json_data = json.loads(plain_text)
        self.json_content = json_data
        return json_data
    
    def get_selected_features_and_details(self):
        selected_features  = []
        feature_details = {}
        design_state = self.json_content["design_state_data"]
        feature_handling = design_state["feature_handling"]
        target_variable = design_state["target"]["target"]
        for feature, details in feature_handling.items():
            if(details["is_selected"]):
                name = details["feature_name"]
                selected_features.append(name)
                feature_details[name] = details
        selected_features.remove(target_variable)
        return selected_features, feature_details
    
    
    def get_problem_type_and_target_variable(self):
        design_state = self.json_content["design_state_data"]
        problem_type  = design_state["target"]["prediction_type"]
        target_variable = design_state["target"]["target"]
        return problem_type,target_variable
    
    def get_selected_models(self):
        algorithms = self.json_content["design_state_data"]["algorithms"]
        selected_algorithms = []
        algo_hyperparameters = {}
        for algo, details in algorithms.items():
            if(details["is_selected"]):
                selected_algorithms.append(algo)      
                algo_hyperparameters[algo] = details["hyperparameters"]
                algo_hyperparameters[algo].pop("model_name")
                algo_hyperparameters[algo].pop("is_selected")

        return selected_algorithms, algo_hyperparameters
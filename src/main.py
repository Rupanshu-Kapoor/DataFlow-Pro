from src.data_reader import DataReader
from src.feature_handler import FeatureHandler
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator
from src.config import * 

def main():
    # Read the RTF file and parse the JSON content
    data_reader = DataReader()
    json_content = data_reader.rtf_to_json_parser(rtf_file_path=RTF_FILE_PATH)
    
    # Extract dataset information from JSON
    problem_type, target_variable = data_reader.get_problem_type_and_target_variable()

    # Extract feature names and target variable from JSON content
    selected_features, feature_details = data_reader.get_selected_features_and_details()

    # Transform features
    feature_handler = FeatureHandler(json_content)
    X_train, X_test, y_train, y_test = FeatureHandler.get_split_dataset(selected_features)    

    

if __name__ == '__main__':
    main()
from data_reader import DataReader
from feature_handler import FeatureHandler
from model_trainer import ModelTrainer
from evaluator import Evaluator
from config import * 

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
    X_train, X_test, y_train, y_test = feature_handler.get_split_dataset(selected_features)    

    X_train_transformed , X_test_transformed = feature_handler.transform_X_features(X_train, X_test, feature_details)
    y_train_transformed , y_test_transformed = feature_handler.transform_y_features(y_train, y_test, feature_details)
    
    # Model building and hyperparameter tuning
    selected_models, model_parameters = data_reader.get_selected_models()
    model_trainer = ModelTrainer(json_content)
    trained_model = model_trainer.build_and_tune_model(X_train_transformed, y_train_transformed, 
                                                       problem_type, selected_models, model_parameters)
    
    # Save trained model
    
    

if __name__ == '__main__':
    main()
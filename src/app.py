import streamlit as st
from data_reader import DataReader
from datetime import datetime
from feature_handler import FeatureHandler
from model_trainer import ModelTrainer
from evaluator import Evaluator
from config import * 

def main():
    main_heading = "<h1 style='text-align: center; color: #cce7ff; margin-bottom: 0; margin-top:-50px'>DataFlow Pro</h1>"
    tagline = "<h4 style='text-align: center; color: #cce7ff; margin-top: -25px;'>Automating ML Workflow with Ease</h4>"
    header_content = main_heading + tagline
    st.markdown(header_content, unsafe_allow_html=True)
    st.markdown("---")

    json_file = st.file_uploader("Upload RTF file", type=["rtf", "json", "txt"])
    
    if json_file is not None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        extension = json_file.name.split('.')[-1]
        file_name = f"{json_file.name.split('.')[0]}_{current_time}.{extension}"
        save_file_path = '../data/'+file_name
        
        with open(save_file_path, 'wb') as file:
            file.write(json_file.read())

            st.success("File uploaded successfully, mdoels are ready to be trained!")

    if st.button("Train Models"):
        if json_file is not None:
            with st.spinner('Hang On, Training Models For You...'):
                # Read the RTF file and parse the JSON content
                data_reader = DataReader(rtf_file_path=save_file_path)
                json_content = data_reader.rtf_to_json_parser()

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
                trained_models = model_trainer.build_and_tune_model(X_train_transformed, y_train_transformed, 
                                                                problem_type, selected_models, model_parameters)
                
                
                # Evaluate the model
                evaluator = Evaluator(json_content, problem_type, target_variable)
                evaluation_results = evaluator.evaluate_model(trained_models, X_test_transformed, y_test_transformed)
                
        else:
            st.error("Please upload a JSON file first.")

if __name__ == '__main__':
    main()
import streamlit as st
from data_reader import DataReader
from datetime import datetime
from feature_handler import FeatureHandler
from model_trainer import ModelTrainer
from evaluator import Evaluator
from config import * 
import pandas as pd
import json


def extract_column_info(df):
    column_info = {}
    for column in df.columns:
        column_info[column] = {
            "feature_name": column,
            "is_selected": True, 
            "feature_variable_type": str(df[column].dtype),
            "feature_details": {
                "numerical_handling": None,
                "rescaling": False,
                "scaling_type": None,
                "make_derived_feats": False,
                "missing_values": "Impute",
                "impute_with": None
            }
        }
    return column_info


def extract_algorithms_info(algo_list):
    algo_info = {}
    for algo in algo_list:
        algo_info[algo] = {
            "model_name" : algo,
            "is_selected" : False,
            "random_state" : [42]
        }
    return algo_info


def generate_json(session_name, dataset_name, target, train, feature_handling, algorithms):
    json_data = {
        "session_name": session_name,
        "session_description": session_name,
        "design_state_data": {
            "session_info": {
                "dataset": dataset_name,
                "session_name": session_name,
                "session_description": session_name
            },
            "target": target,
            "train": train,
            "feature_handling": feature_handling,
            "algorithms": algorithms
        }
    }
    return json_data



def train_models(save_file_path, json_file):
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
            y_train_transformed , y_test_transformed = feature_handler.transform_y_features(y_train, y_test, feature_details, target_variable)
            
            # Model building and hyperparameter tuning
            selected_models, model_parameters = data_reader.get_selected_models()
            model_trainer = ModelTrainer(json_content)
            trained_models = model_trainer.build_and_tune_model(X_train_transformed, y_train_transformed, 
                                                            problem_type, selected_models, model_parameters)
            
            
            # Evaluate the model
            evaluator = Evaluator(json_content, problem_type, target_variable)
            evaluation_results = evaluator.evaluate_model(trained_models, X_test_transformed, y_test_transformed)
            # display bar chart of evaluation results
            st.subheader("Different Model Comparison")
            evaluator.display_metrics(evaluation_results)


            
    else:
        st.error("Please upload a JSON file first.")


def create_json_and_train():
    
    st.write("### Upload Dataset: ")
    uploaded_file = st.file_uploader("Upload Dataset CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Sample Data:")
        st.write(df.head())

        # Extract column information
        column_info = extract_column_info(df)

        # take input for prediction_type
        st.write("### Select Prediction Parameters:")
        prediction_type = st.selectbox("Prediction Type", ["Regression", "Classification"], key="prediction_selectbox")

        # Checkbox for selecting target columns and feature details
        target_variable = st.selectbox("Target Variable", df.columns, key="target_selectbox")

        # add option to let user select how to encode target variable

        column_info[target_variable]["feature_details"] = {}
        # if target_variable is of category type, add option to label encode
        if column_info[target_variable]["feature_variable_type"] == "object":
            column_info[target_variable]["feature_details"]["text_handling"] = st.selectbox("Text Handling", ["Tokenize and hash", "Label Encoding"], key="text_handling_selectbox", index=0)

        train = {}
        train["k_fold"] = st.number_input("K-Fold", min_value=2, value=5, step=1, key="kfold")
        train["train_ratio"] = st.number_input("Train Ratio", min_value=0.0, max_value=1.0, value=0.8, step=0.1, key="train_ratio")
        train["random_seed"] = st.number_input("Random Seed", min_value=0, value=42, step=1, key="random_seed")

        target = {"prediction_type": prediction_type, 
                  "target": target_variable,
                  "type": prediction_type,
                  "partitioning": True}

        st.write("### Select Columns to Include:")
        for column in column_info:
            if column != target_variable:
                column_info[column]["is_selected"] = st.checkbox(column, key=f"{column}_checkbox", value=False)
                if column_info[column]["is_selected"]:
                    with st.expander(f"{column} Feature Handling", expanded=False):
                        column_info[column]["feature_details"]["rescaling"] = st.checkbox("Rescaling", key=f"{column}_scaling_checkbox")
                        if column_info[column]["feature_details"]["rescaling"] and column_info[column]["feature_variable_type"] != "object":
                            column_info[column]["feature_details"]["scaling_type"] = st.selectbox("Scaling Type", ["MinMaxScaler", "StandardScaler"], key=f"{column}_scaling_type_select")
                        column_info[column]["feature_details"]["missing_values"] = st.checkbox("Imputation", key=f"{column}_imputation_checkbox")
                        if column_info[column]["feature_details"]["missing_values"]:
                            column_info[column]["feature_details"]["impute_with"] = st.selectbox("Imputation With", ["Mean", "Median", "Mode", "Custom"], key=f"{column}_imputation_type_select")
                            if column_info[column]["feature_details"]["impute_with"] == "Custom":
                                column_info[column]["feature_details"]["custom_impute_value"] = st.text_input(f"Custom Impute Value", key=f"{column}_imputation_value_input")
                        if column_info[column]["feature_variable_type"] == "object":
                            column_info[column]["feature_details"]["encoding"] = st.selectbox("Encode Categorical Feature with", ["OridnalEncoder", "OneHotEncoder"], key = f"{column}_encoding_type")
        # Checkbox for selecting columns
        st.write(f"### Select {prediction_type} Algorithms:")
        if prediction_type == "Regression":
            algorithms_list = ["RandomForestRegressor", "LinearRegression", "RidgeRegression", "LassoRegression",
                               "ElasticNetRegression","xg_boost", "DecisionTreeRegressor", "SVM", "KNN", "neural_network"]
        else:
            algorithms_list = ["RandomForestClassifier", "LogisticRegression",  "xg_boost", 
                         "DecisionTreeClassifier", "SVM", "KNN", "neural_network"]
        
        algo_info = extract_algorithms_info(algorithms_list)
        for algo in algo_info:
            algo_info[algo]["is_selected"] = st.checkbox(algo, key=f"{algo}_checkbox")
            if algo_info[algo]["is_selected"]:
                with st.expander(f"{algo} HyperParameters", expanded=False):
                    if algo == "RandomForestClassifier" or algo == "RandomForestRegressor":
                        algo_info[algo]["min_trees"] = st.number_input("Minimum Trees", min_value=1, max_value=100, value=10, step=1, key=f"{algo}_min_trees")
                        algo_info[algo]["max_trees"] = st.number_input("Maximum Trees", min_value=1, max_value=100, value=30, step=1, key=f"{algo}_max_trees")
                        algo_info[algo]["min_depth"] = st.number_input("Minimum Depth", min_value=1, max_value=100, value=20, step=1, key=f"{algo}_min_depth")
                        algo_info[algo]["max_depth"] = st.number_input("Maximum Depth", min_value=1, max_value=100, value=30, step=1, key=f"{algo}_max_depth")
                        algo_info[algo]["min_samples_per_leaf_min_value"] = st.number_input("Minimum Samples Per Leaf", min_value=1, max_value=100, value=5, step=1, key=f"{algo}_min_samples_per_leaf")
                        algo_info[algo]["min_samples_per_leaf_max_value"] = st.number_input("Maximum Samples Per Leaf", min_value=1, max_value=100, value=50, step=1, key=f"{algo}_max_samples_per_leaf")
                    
                    elif algo == "LinearRegression" or algo == "LogisticRegression" or algo == "ElasticNetRegression":
                        algo_info[algo]["min_iter"] = st.number_input("Minimum Iterations", min_value=1, max_value=100, value=30, step=1, key=f"{algo}_min_iter")
                        algo_info[algo]["max_iter"] = st.number_input("Maximum Iterations", min_value=1, max_value=100, value=50, step=1, key=f"{algo}_max_iter")
                        algo_info[algo]["min_regparam"] = st.number_input("Minimum Regularization Parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key=f"{algo}_min_regparam")
                        algo_info[algo]["max_regparam"] = st.number_input("Maximum Regularization Parameter", min_value=0.0, max_value=1.0, value=0.8, step=0.1, key=f"{algo}_max_regparam")
                        algo_info[algo]["min_elasticnet"] = st.number_input("Minimum Elasticnet", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key=f"{algo}_min_elasticnet")
                        algo_info[algo]["max_elasticnet"] = st.number_input("Maximum Elasticnet", min_value=0.0, max_value=1.0, value=0.8, step=0.1, key=f"{algo}_max_elasticnet")

                    elif algo == "RidgeRegression" or algo == "LassoRegression":
                        algo_info[algo]["min_iter"] = st.number_input("Minimum Iterations", min_value=1, max_value=100, value=30, step=1, key=f"{algo}_min_iter")
                        algo_info[algo]["max_iter"] = st.number_input("Maximum Iterations", min_value=1, max_value=100, value=50, step=1, key=f"{algo}_max_iter")
                        algo_info[algo]["min_regparam"] = st.number_input("Minimum Regularization Parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key=f"{algo}_min_regparam")
                        algo_info[algo]["max_regparam"] = st.number_input("Maximum Regularization Parameter", min_value=0.0, max_value=1.0, value=0.8, step=0.1, key=f"{algo}_max_regparam")
                    
                    elif algo == "DecisionTreeClassifier" or algo == "DecisionTreeRegressor":
                        algo_info[algo]["min_depth"] = st.number_input("Minimum Depth", min_value=1, max_value=100, value=4, step=1, key=f"{algo}_min_depth")
                        algo_info[algo]["max_depth"] = st.number_input("Maximum Depth", min_value=1, max_value=100, value=7, step=1, key=f"{algo}_max_depth")
                        algo_info[algo]["use_gini"] = st.checkbox("Use Gini Index", value=False, key=f"{algo}_use_gini")
                        algo_info[algo]["use_entropy"] = st.checkbox("Use Entropy", value=True, key=f"{algo}_use_entropy")
                        algo_info[algo]["min_samples_per_leaf"] = st.text_input("Minimum Samples Per Leaf", placeholder="Enter comma separated list of values for min_samples_per_leaf", 
                                                                                key=f"{algo}_min_samples_per_leaf")
                        # check if min_samples_per_leaf is there
                        if algo_info[algo]["min_samples_per_leaf"]:
                            algo_info[algo]["min_samples_per_leaf"] = [int(x) for x in algo_info[algo]["min_samples_per_leaf"].split(",")]
                        else:
                            algo_info[algo]["min_samples_per_leaf"] = [12, 6]
                        algo_info[algo]["use_best"] = st.checkbox("Use Best", value=True, key=f"{algo}_use_best")
                        algo_info[algo]["use_random"] = st.checkbox("Use Random", value=True, key=f"{algo}_use_random")
                    
                    elif algo == "SVM":
                        algo_info[algo]["linear_kernel"] = st.checkbox("Linear Kernel", value=True, key=f"{algo}_linear_kernel")
                        algo_info[algo]["rep_kernel"] = st.checkbox("Rep Kernel", value=True, key=f"{algo}_rep_kernel")
                        algo_info[algo]["polynomial_kernel"] = st.checkbox("Polynomial Kernel", value=True, key=f"{algo}_polynomial_kernel")
                        algo_info[algo]["sigmoid_kernel"] = st.checkbox("Sigmoid Kernel", value=True, key=f"{algo}_sigmoid_kernel")
                        algo_info[algo]["c_value"] = st.text_input("C Value", placeholder="Enter comma separated list of values for C Value", key=f"{algo}_c_value")
                        # convert c values into list of integers
                        if algo_info[algo]["c_value"]:
                            algo_info[algo]["c_value"] = [int(x) for x in algo_info[algo]["c_value"].split(",")]   
                        else:
                            algo_info[algo]["c_value"] = [566, 79]
                        algo_info[algo]["auto"] = st.checkbox("Auto", value=True, key=f"{algo}_auto")
                        algo_info[algo]["scale"] = st.checkbox("Scale", value=True, key=f"{algo}_scale")
                        algo_info[algo]["custom_gamma_values"] = st.checkbox("Custom Gamma Values", value=True, key=f"{algo}_custom_gamma_values")
                        algo_info[algo]["tolerance"] = [st.number_input("Tolerance", min_value=0.0, max_value=1.0, value=0.001, step=0.001, key=f"{algo}_tolerance")]
                        algo_info[algo]["max_iterations"] = st.number_input("Maximum Iterations", min_value=1, max_value=100, value=10, step=1, key=f"{algo}_max_iterations")
                        if algo_info[algo]["max_iterations"]:
                            algo_info[algo]["max_iterations"] = [algo_info[algo]["max_iterations"]]

                    elif algo == "KNN":
                        algo_info[algo]["k_value"] = st.text_input("K Value", placeholder="Enter comma separated list of values for K Value", key=f"{algo}_k_value")
                        if algo_info[algo]["k_value"]:
                            algo_info[algo]["k_value"] = [int(x) for x in algo_info[algo]["k_value"].split(",")]
                        else:
                            algo_info[algo]["k_value"] = [78]
                        algo_info[algo]["distance_weighting"] = [st.checkbox("Distance Weighting", value=True, key=f"{algo}_distance_weighting")]
                        algo_info[algo]["neighbour_finding_algorithm"] = st.selectbox("Neighbour Finding Algorithm", ["auto", "ball_tree", "kd_tree", "brute"], key=f"{algo}_neighbour_finding_algorithm", index=0)
                        algo_info[algo]["p_value"] = st.number_input("P Value", min_value=1, max_value=2, value=1, step=1, key=f"{algo}_p_value")

                    elif algo == "neural_network":
                        algo_info[algo]["hidden_layer_sizes"] = st.text_input("Hidden Layer Sizes", placeholder="Enter comma separated list of values for Hidden Layer Sizes", key=f"{algo}_hidden_layer_sizes")
                        if algo_info[algo]["hidden_layer_sizes"]:
                            algo_info[algo]["hidden_layer_sizes"] = [int(x) for x in algo_info[algo]["hidden_layer_sizes"].split(",")]
                        else:
                            algo_info[algo]["hidden_layer_sizes"] = [67, 89]
                        algo_info[algo]["activation"] = ""
                        algo_info[algo]["alpha_value"] = [st.number_input("Alpha Value", min_value=0.0, max_value=1.0, value=0.01, step=0.0001, key=f"{algo}_alpha_value")]
                        algo_info[algo]["max_iterations"] = [st.number_input("Max Iterations", min_value=0, max_value=1000, value=10, step=100, key=f"{algo}_max_iterations")]
                        algo_info[algo]["convergence_tolerance"] = [st.number_input("Convergence Tolerance", min_value=0.0, max_value=1.0, value=0.1, step=0.0001, key=f"{algo}_convergence_tolerance")]
                        algo_info[algo]["early_stopping"] = [st.checkbox("Early Stopping", value=True, key=f"{algo}_early_stopping")]
                        algo_info[algo]["solver"] = [st.selectbox("Solver", ["lbfgs", "sgd", "adam"], key=f"{algo}_solver", index=2)]
                        algo_info[algo]["shuffle_data"] = [st.checkbox("Shuffle Data", value=True, key=f"{algo}_shuffle_data")]
                        algo_info[algo]["initial_learning_rate"] = [st.number_input("Initial Learning Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.001, key=f"{algo}_initial_learning_rate")]
                        algo_info[algo]["automatic_batching"] = [st.checkbox("Automatic Batching", value=True, key=f"{algo}_automatic_batching")]
                        algo_info[algo]["beta_1"] = [st.number_input("Beta 1", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key=f"{algo}_beta_1")]
                        algo_info[algo]["beta_2"] = [st.number_input("Beta 2", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key=f"{algo}_beta_2")]
                        algo_info[algo]["epsilon"] = [st.number_input("Epsilon", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key=f"{algo}_epsilon")]
                        algo_info[algo]["power_t"] = [st.number_input("Power T", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key=f"{algo}_power_t")]
                        algo_info[algo]["momentum"] = [st.number_input("Momentum", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key=f"{algo}_momentum")]
                        algo_info[algo]["use_nesterov_momentum"] = [st.checkbox("Use Nesterov Momentum", value=False, key=f"{algo}_use_nesterov_momentum")]
                    
                    elif algo == "xg_boost":
                        algo_info[algo]["use_gradient_boosted_tree"] = st.checkbox("Use Gradient Boosted Tree", value=True, key=f"{algo}_use_gradient_boosted_tree")
                        algo_info[algo]["dart"] = st.checkbox("DART", value=True, key=f"{algo}_dart")
                        algo_info[algo]["tree_method"] = [st.selectbox("Tree Method", ["exact", "approx", "hist"], key=f"{algo}_tree_method", index=1)]
                        algo_info[algo]["max_num_of_trees"] = [st.number_input("Max Number of Trees", min_value=0, max_value=1000, value=10, step=100, key=f"{algo}_max_num_of_trees")]
                        algo_info[algo]["early_stopping"] = st.checkbox("Early Stopping", value=True, key=f"{algo}_early_stopping")
                        if algo_info[algo]["early_stopping"]:
                            algo_info[algo]["early_stopping_rounds"] = [st.number_input("Early Stopping Rounds", min_value=0, max_value=1000, value=2, step=100, key=f"{algo}_early_stopping_rounds")]
                        algo_info[algo]["max_depth_of_tree"] = [st.number_input("Max Depth of Tree", min_value=0, max_value=1000, value=10, step=100, key=f"{algo}_max_depth_of_tree")]
                        algo_info[algo]["learningRate"] = [st.number_input("Learning Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.001, key=f"{algo}_learningRate")]
                        algo_info[algo]["l1_regularization"] = [st.number_input("L1 Regularization", min_value=0.0, max_value=1.0, value=0.1, step=0.001, key=f"{algo}_l1_regularization")]
                        algo_info[algo]["l2_regularization"] = [st.number_input("L2 Regularization", min_value=0.0, max_value=1.0, value=0.1, step=0.001, key=f"{algo}_l2_regularization")]
                        algo_info[algo]["gamma"] = [st.number_input("Gamma", min_value=0.0, max_value=1.0, value=0.1, step=0.001, key=f"{algo}_gamma")]
                        algo_info[algo]["min_child_weight"] = [st.number_input("Min Child Weight", min_value=0.0, max_value=1.0, value=0.1, step=0.001, key=f"{algo}_min_child_weight")]
                        algo_info[algo]["sub_sample"] = [st.number_input("Sub Sample", min_value=0.0, max_value=1.0, value=0.1, step=0.001, key=f"{algo}_sub_sample")]
                        algo_info[algo]["col_sample_by_tree"] = [st.number_input("Column Sample By Tree", min_value=0.0, max_value=1.0, value=0.1, step=0.001, key=f"{algo}_col_sample_by_tree")]
                        algo_info[algo]["replace_missing_values"] = st.checkbox("Replace Missing Values", value=True, key=f"{algo}_replace_missing_values")

        # Generate JSON
        if st.button("Generate JSON and train models"):
            session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_data = generate_json(session_name, uploaded_file.name, target, train, column_info, algo_info)
            # save json to file
            if json_data is not None:
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                extension = "json"
                file_name = f"uploaded_{current_time}.{extension}"
                save_file_path = '../data/'+file_name
                
                with open(save_file_path, 'w') as file:
                    # file.write(json_data.read())
                    json.dump(json_data, file)
                    st.success("JSON file generated successfully, models are being trained!")

                train_models(save_file_path, json_data)


def upload_json_and_train():
 
    st.write("### Upload JSON File")
    json_file = st.file_uploader("Upload RTF/JSON/TXT file", type=["rtf", "json", "txt"])
    
    if json_file is not None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        extension = json_file.name.split('.')[-1]
        file_name = f"{json_file.name.split('.')[0]}_{current_time}.{extension}"
        save_file_path = '../data/'+file_name
        
        with open(save_file_path, 'wb') as file:
            file.write(json_file.read())

            st.success("File uploaded successfully, mdoels are ready to be trained!")

    # create button to train models
    if st.button("Train Models"):
        if json_file is not None:
            train_models(save_file_path, json_file)
        else:
            st.warning("Please upload a JSON file")

def main():
    
    # 
    main_heading = "<h1 style='text-align: center; color: #cce7ff; margin-bottom: 0; margin-top:-50px'>DataFlow Pro</h1>"
    tagline = "<h4 style='text-align: center; color: #cce7ff; margin-top: -25px;'>Automating ML Workflow with Ease</h4>"
    header_content = main_heading + tagline
    st.markdown(header_content, unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Navigation")
    st.write("If you want to create a JSON and train a model, please click on the <u><b>Create Json and Train Model</b></u> button.", unsafe_allow_html=True)
    st.write("If you have an RTF/JSON/TXT file, please upload it and click on the <u><b>Upload Json and train model</b></u> button.", unsafe_allow_html=True)
    page = st.radio(" ", ("Create Json and Train Model", "Upload Json and train model"), index= None)

    if page == "Create Json and Train Model":
        create_json_and_train()
    elif page == "Upload Json and train model":
        upload_json_and_train()
    st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #000000;
            text-align: center;
            padding: 10px 0;
        }
    </style>
    <div class="footer">
        <p>Made with ❤️ by Rupanshu Kapoor.</p>
    </div>
""", unsafe_allow_html=True)
    

if __name__ == '__main__':
    main()
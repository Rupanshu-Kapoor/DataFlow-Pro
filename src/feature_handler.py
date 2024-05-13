# Contains classes and functions for handling and transforming 
# features based on the JSON file information.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
import streamlit as st

class FeatureHandler:
    def __init__(self, json_content):
        self.json_content = json_content

    def impute_missing_values(self, feature_details, X_train, X_test=None):
        mean_impute_features = []
        median_impute_features = []
        mode_impute_features = []
        custom_impute_features = []
        for feature in X_train.columns:
            details = feature_details[feature]['feature_details']
            if details["missing_values"]:
                if details["impute_with"].lower() == "mean":
                    mean_impute_features.append(feature)
                elif details["impute_with"].lower() == "median":
                    median_impute_features.append(feature)
                elif details["impute_with"].lower() == "mode":
                    mode_impute_features.append(feature)
                elif details["impute_with"].lower() == "custom":
                    custom_impute_features.append(feature)

        if mean_impute_features:
            X_train[mean_impute_features] = X_train[mean_impute_features].fillna(X_train[mean_impute_features].mean())
        if median_impute_features:
            X_train[median_impute_features] = X_train[median_impute_features].fillna(X_train[median_impute_features].median())
        if mode_impute_features:
            X_train[mode_impute_features] = X_train[mode_impute_features].fillna(X_train[mode_impute_features].mode().iloc[0])
        if custom_impute_features:
            for feature in custom_impute_features:
                X_train[feature] = X_train[feature].fillna(feature_details[feature]['feature_details']['custom_impute_value'])
        if X_test is not None:
            if mean_impute_features:
                X_test[mean_impute_features] = X_test[mean_impute_features].fillna(X_train[mean_impute_features].mean())
            if median_impute_features:
                X_test[median_impute_features] = X_test[median_impute_features].fillna(X_train[median_impute_features].median())
            if mode_impute_features:
                X_test[mode_impute_features] = X_test[mode_impute_features].fillna(X_train[mode_impute_features].mode().iloc[0])
            if custom_impute_features:
                for feature in custom_impute_features:
                    X_test[feature] = X_test[feature].fillna(feature_details[feature]['feature_details']['custom_impute_value'])
        return X_train, X_test

    

    # TODO: Add imputation for categorical features
    def scale_features(self, feature_details, X_train, X_test=None):
        min_max_scaler_features = []
        standard_scaler_features = []
        # for feature, details in feature_details.items():
        for feature in feature_details.keys():
            details = feature_details[feature]['feature_details']
            if details.get("rescaling"):
                if details["rescaling"]!= "No rescaling" and details["scaling_type"] == "MinMaxScaler" :
                    min_max_scaler_features.append(feature)
                elif details["rescaling"] != "No rescaling" and details["scaling_type"] == "StandardScaler" :
                    standard_scaler_features.append(feature)
        if min_max_scaler_features:
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train[min_max_scaler_features])
            X_train[min_max_scaler_features] = X_train_scaled
        if standard_scaler_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[standard_scaler_features])
            X_train[standard_scaler_features] = X_train_scaled
        if X_test is not None:
            if min_max_scaler_features:
                X_test_scaled = scaler.fit_transform(X_test[min_max_scaler_features])
                X_test[min_max_scaler_features] = X_test_scaled
            if standard_scaler_features:
                X_test_scaled = scaler.fit_transform(X_test[standard_scaler_features])
                X_test[standard_scaler_features] = X_test_scaled
        return X_train, X_test

    def encode_features(self, feature_details, X_train, X_test=None):
       
        ordinal_encoder_cols = []
        one_hot_encoder_cols = []
        for feature in X_train.columns:
            if feature_details[feature]["feature_variable_type"] == "object":
                details = feature_details[feature]['feature_details']
                if details["encoding"] == "OridnalEncoder":
                    ordinal_encoder_cols.append(feature)
                elif details["encoding"] == "OneHotEncoder":
                    one_hot_encoder_cols.append(feature)
        if ordinal_encoder_cols:
            ordinal_encoder = OrdinalEncoder()
            X_train[ordinal_encoder_cols] = ordinal_encoder.fit_transform(X_train[ordinal_encoder_cols])
            if X_test is not None:
                X_test[ordinal_encoder_cols] = ordinal_encoder.transform(X_test[ordinal_encoder_cols])
   
        if one_hot_encoder_cols:
            one_hot_encoder = OneHotEncoder( drop="first", sparse_output=False)
            temp_df = pd.DataFrame(one_hot_encoder.fit_transform(X_train[one_hot_encoder_cols]), 
                                   columns=one_hot_encoder.get_feature_names_out(),
                                   index=X_train.index)
            X_train = X_train.drop(one_hot_encoder_cols, axis=1)
            X_train = pd.concat([X_train, temp_df], axis=1)
            
            if X_test is not None:
                temp_df = pd.DataFrame(one_hot_encoder.transform(X_test[one_hot_encoder_cols]), 
                                       columns=one_hot_encoder.get_feature_names_out(), 
                                       index=X_test.index)
                X_test = X_test.drop(one_hot_encoder_cols, axis=1)
                X_test = pd.concat([X_test, temp_df], axis=1)
        return X_train, X_test

    def transform_X_features(self, X_train, X_test, feature_details):
        
        X_train_transformed, X_test_transformed = self.impute_missing_values(feature_details, X_train, X_test)
        X_train_transformed, X_test_transformed = self.encode_features(feature_details, X_train_transformed, X_test_transformed)
        X_train_transformed, X_test_transformed = self.scale_features(feature_details, X_train_transformed, X_test_transformed)
        return X_train_transformed, X_test_transformed
    # tokenize and hash the target variable
    def tokenize_target_variable(self, y_train, y_test):
        details = self.json_content["design_state_data"]["feature_handling"]
        feature_details =  details[y_train.name]["feature_details" ]
        if feature_details["text_handling"] == "Tokenize and hash":
        # tokenize the target variable
            label_encoder = LabelEncoder()
            y_train_tokenized = y_train.apply(lambda x: x.split("-")[1])
            y_train_encoded = label_encoder.fit_transform(y_train_tokenized)

            y_test_tokenized = y_test.apply(lambda x: x.split("-")[1])
            y_test_encoded = label_encoder.transform(y_test_tokenized)
            return y_train_encoded, y_test_encoded
        

    def label_encode_target_variable(self, y_train, y_test):
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        return y_train_encoded, y_test_encoded


    def transform_y_features(self, y_train, y_test, feature_details, target_variable):
        if feature_details[target_variable]["feature_variable_type"] == "object":
            if feature_details[target_variable]["feature_details"]["text_handling"] == "Tokenize and hash":
                y_train_transformed, y_test_transformed = self.tokenize_target_variable(y_train, y_test)
            elif feature_details[target_variable]["feature_details"]["text_handling"] == "Label Encoding":
                y_train_transformed, y_test_transformed = self.label_encode_target_variable(y_train, y_test)
            return y_train_transformed, y_test_transformed
        else:
            return y_train, y_test

    def get_split_dataset(self, selected_features):
        design_state = self.json_content["design_state_data"]
        dataset = design_state["session_info"]["dataset"]
        target_variable = design_state["target"]["target"]
        
        train_info = design_state["train"]
        train_ratio = train_info["train_ratio"]
        random_seed = train_info["random_seed"]

        DATASET_PATH = "../data/"+dataset
        df = pd.read_csv(DATASET_PATH)
        X = df[selected_features]
        Y = df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=train_ratio, 
                                                            random_state=random_seed)
        
        return X_train, X_test, y_train, y_test
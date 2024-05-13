# DataFlow Pro
Automating ML Workflows with Ease

## Introduction
The Automated ML is a Python application designed to automate the process of building, tuning, and evaluating machine learning models based on json provided in RTF/JSON?/TXT file format. <br>
This application follows a structured flow to read the json file, extract dataset information, transform features, split data, build and tune models, and evaluate their performance.


## Installation
To use the Automated ML Pipeline, follow these steps:

1. Clone this repository to your local machine: <br>
```git clone https://github.com/Rupanshu-Kapoor/AutomateML.git```

2. Install the required dependencies: <br>
   `pip install -r requirements.txt`

3. Run the application: <br>
   `streamlit run app.py`

 
 ## Steps to Use the Application:

 You can use the application in following two ways:

### (A). Create Json and Train Model

1. Upload the dataset on the tool on which you want to train the different model.
2. Once the data is uploaded, you can preview the dataset.
3. Select prediction parameters (prediction type, target variable, k-fold, etc.). 
4. Select features to be used for prediction.
5. When you select any feature, you can choose how to handle it. (rescaling, encoding, etc.)
6. Select the model to be used for prediction.
7. When you select any model, you can choose hyperparameters for tuning.
8. Once all the parameters are selected, click on `Generate Json and Train Model` button.
9. Application will generate the json file and train the model and display the results.

### (B). Upload Json and Train Model
 1. Upload the json file that contains all the dataset information.
 2. Click on Train Models.
 3. Application will train the model and display the results.

## Working of the Application:
The application performs the following tasks in sequence:
1. **Read the JSON File and Parse JSON Content**: The RTF/JSON file is read, converted to plain text, and JSON content is extracted.
2. **Extract Dataset Information**: Extract dataset information such as feature names, target variable, problem type (regression/classification), feature handling, etc.
3. **Transform Features**: Features are transformed based on the specified feature handling methods.
4. **Sample Data and Train-Test Split**: Data is sampled and split into training and testing sets.
5. **Model Building**: Models are built based on the problem type (regression/classification).
6. **Hyperparameter Tuning**: Hyperparameters of the models are tuned using grid search.
7. **Model Evaluation**: Trained models are evaluated using specified evaluation metrics.
<! --8. **Save Results**: Trained models and evaluation metrics are saved in the results/ directory. -->


## Use Cases

This application can be used for various use cases, including but not limited to:

- Automated machine learning (AutoML) pipelines.
- Data preprocessing and feature engineering tasks.
- Model training and evaluation for regression or classification problems.
- Hyperparameter tuning and model selection.
- Experimentation with different datasets and configurations.

## Future Work
Possible future enhancements for the application include:

- Adding support for additional data formats (e.g., CSV, Excel).
- Implementing more advanced feature engineering techniques.
- Incorporating more sophisticated model selection and evaluation methods.
- Enhancing the user interface for easier interaction.
- Integrating with external APIs or databases for data retrieval.

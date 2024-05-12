# Automated ML

## Introduction
The Automated ML is a Python application designed to automate the process of building, tuning, and evaluating machine learning 
models based on data provided in RTF format containing JSON metadata. <br>
This application follows a structured flow to read the RTF file, extract dataset information, transform features, split data, 
build and tune models, and evaluate their performance.


## Installation
To use the Automated ML Pipeline, follow these steps:

1. Clone this repository to your local machine: <br>
```git clone https://github.com/Rupanshu-Kapoor/AutomateML.git```

2. Install the required dependencies: <br>
   `pip install -r requirements.txt`

 ## Usage
 ### Step-by-Step Guide:
 Follow these steps to use the application:

 1. Place your RTF file containing JSON data in the `data/` directory.
 2. Run the application by executing the `main.py` file: <br>
 `python app/main.py`

## Working of the Application:
The application performs the following tasks in sequence:
1. **Read the RTF File and Parse JSON Content**: The RTF file is read, converted to plain text, and JSON content is extracted.
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

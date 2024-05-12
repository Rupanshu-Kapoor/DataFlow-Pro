# Optional. Contains configuration settings 
# for your application (e.g., paths, hyperparameter ranges).
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRegressor



RTF_FILE_PATH = "data/algoparams_from_ui1.json.rtf"


model_dict = {
    "RandomForestClassifier" : RandomForestClassifier(),
    "RandomForestRegressor" : RandomForestRegressor(),
    "LinearRegression": SGDRegressor(),
    "LogisticRegression": LogisticRegression(),
    "RidgeRegression": Ridge(),
    "LassoRegression": Lasso(),
    "ElasticNetRegression": ElasticNet(),
    "XGBoostClassifier": XGBClassifier(),
    "XGBoostRegressor": XGBRegressor(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "DecisionTreeClassifier":DecisionTreeClassifier(),
    "SVMClassifier": SVC(),
    "SVMRegressor": SVR(),
    "KNNRegressor": KNeighborsRegressor(),
    "KNNClassifier": KNeighborsClassifier(),
    "neural_network": MLPClassifier()    
}



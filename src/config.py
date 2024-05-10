# Optional. Contains configuration settings 
# for your application (e.g., paths, hyperparameter ranges).
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier



RTF_FILE_PATH = "data\algoparams_from_ui1.json.rtf"


model_dict = {
    "RandomForestClassifier" : RandomForestClassifier(random_state=random_seed),
    "RandomForestRegressor" : RandomForestRegressor(random_state=random_seed),
    "LinearRegression": LinearRegression(),
    "LogisticRegression": LogisticRegression(random_state=random_seed),
    "RidgeRegression": Ridge(),
    "LassoRegression": Lasso(),
    "ElasticNetRegression": ElasticNet(),
    "xg_boost": "ds",
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=random_seed),
    "DecisionTreeClassifier":DecisionTreeClassifier(random_state=random_seed),
    "SVM": SVC(random_state=random_seed),
    "KNN": KNeighborsClassifier(),
    "neural_network": MLPClassifier(random_state=random_seed)    
}
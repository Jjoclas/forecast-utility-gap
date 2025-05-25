from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier

AVAILABLE_MODELS = {
    "ridge": {
        "model": RidgeClassifier,
        "params": {"alpha": [0.1, 1.0, 10.0]}
    },
    "lr": {
        "model": LogisticRegression,
        "params": {"C": [0.1, 1.0, 10.0]}
    },
    "rf": {
        "model": RandomForestClassifier,
        "params": {"n_estimators": [10, 50, 100], "max_depth": [3, 6, 24]}
    },
    "lda": {
        "model": LinearDiscriminantAnalysis,
        "params": {"solver": ["svd", "lsqr", "eigen"]}
    },
    "ada_boost": {
        "model": AdaBoostClassifier,
        "params": {"n_estimators": [50, 100, 200]}
    },
    "gaussian_nb": {
        "model": GaussianNB,
        "params": {}
    },
    "lgbm": {
        "model": LGBMClassifier,
        "params": {"n_estimators": [50, 100, 200], "max_depth": [3, 6, None]}
    },
    "sgd": {
        "model": SGDClassifier,
        "params": {"alpha": [0.0001, 0.001, 0.01]}
    },
    "qda": {
        "model": QuadraticDiscriminantAnalysis,
        "params": {}
    },
    "mlp": {
        "model": MLPClassifier,
        "params": {"alpha": [0.0001, 0.001, 0.01]}
    },
    "decision_tree": {
        "model": DecisionTreeClassifier,
        "params": {"max_depth": [3, 5, 7]}
    },
    "dummy": {
        "model": DummyClassifier,
        "params": {}}
} 
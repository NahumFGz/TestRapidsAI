from catboost import CatBoostClassifier
from cuml.ensemble import RandomForestClassifier as cuMLRandomForestClassifier
from cuml.linear_model import LogisticRegression as cuMLLogisticRegression
from cuml.naive_bayes import GaussianNB as cuMLGaussianNB
from cuml.neighbors import KNeighborsClassifier as cuMLKNeighborsClassifier
from cuml.svm import SVC as cuMLSVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

class BaseModels:
    """
    Clase para seleccionar diferentes modelos de clasificación.

    Métodos:
    - 'logistic_regression': Regresión Logística.
    - 'random_forest': Bosque Aleatorio.
    - 'svm': Support Vector Machine.
    - 'knn': K-Nearest Neighbors.
    - 'naive_bayes': Naive Bayes.
    - 'lgbm': LightGBM Classifier.
    - 'catboost': CatBoost Classifier.
    - 'xgboost': XGBoost Classifier.
    """

    def __init__(self):
        pass

    def provider(self, method):
        """
        Devuelve un modelo de clasificación basado en el método especificado.

        Args:
            method (str): Método de clasificación.

        Returns:
            object: Modelo de clasificación seleccionado.

        Raises:
            ValueError: Si el método no es uno de los esperados.
        """
        if method == "logistic_regression":
            return cuMLLogisticRegression()
        elif method == "random_forest":
            return cuMLRandomForestClassifier()
        elif method == "svm":
            return cuMLSVC(probability=True)
        elif method == "knn":
            return cuMLKNeighborsClassifier()
        elif method == "naive_bayes":
            return cuMLGaussianNB()
        elif method == "lgbm":
            return LGBMClassifier()
        elif method == "catboost":
            return CatBoostClassifier(verbose=0)
        elif method == "xgboost":
            return XGBClassifier()
        else:
            raise ValueError("Invalid classification method specified.")

from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition
from sklearn.model_selection import train_test_split
import numpy as np


class CategoricalImputation(AutoSklearnPreprocessingAlgorithm):
    """
    Substitute missing values by 2
    """

    def __init__(self, strategy="constant", n_neighbors=5,
                       weights='uniform', training_fraction=0.5, random_state=None):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.random_stated = random_state
        self.training_fraction = training_fraction

    def fit(self, X, y=None):
        import sklearn.impute

        if self.strategy == 'constant':
            self.preprocessor = sklearn.impute.SimpleImputer(strategy='constant', fill_value=2, copy=False)
        elif self.strategy == 'most-frequent':
            self.preprocessor = sklearn.impute.SimpleImputer(strategy='most_frequent', copy=False)
        elif self.strategy == 'knn':
            self.preprocessor = sklearn.impute.KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights, copy=False)

        X_new = None
        try:
            min_training_instances = max(
                [self.training_fraction * len(X), 10 * len(np.unique(y)), self.n_neighbors + 1])
            X_new, _, _, _ = train_test_split(X, y, train_size=min_training_instances, random_state=42)
        except:
            X_new = X

        self.preprocessor.fit(X_new)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        X = self.preprocessor.transform(X).astype(int)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'CategoricalImputation',
                'name': 'Categorical Imputation',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        strategy = CategoricalHyperparameter("strategy", ["constant", "most-frequent", "knn"],
                                             default_value="constant")
        cs.add_hyperparameter(strategy)

        # knn hyperparameters
        n_neighbors = UniformIntegerHyperparameter(name="n_neighbors", lower=2, upper=100, log=True, default_value=5)
        weights = CategoricalHyperparameter(name="weights", choices=["uniform", "distance"], default_value="uniform")
        cs.add_hyperparameters([n_neighbors, weights])

        n_neighbors_depends_on_knn = EqualsCondition(n_neighbors, strategy, "knn")
        weights_depends_on_knn = EqualsCondition(weights, strategy, "knn")
        cs.add_conditions([n_neighbors_depends_on_knn, weights_depends_on_knn])

        training_fraction = UniformFloatHyperparameter("training_fraction", 0.0001, 1.0, log=True, default_value=0.5)
        cs.add_hyperparameter(training_fraction)

        return cs

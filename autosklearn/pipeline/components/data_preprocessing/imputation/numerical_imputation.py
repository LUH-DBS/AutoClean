from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
from autosklearn.pipeline.components.data_preprocessing.imputation.impute_new import IterativeImputer
from autosklearn.util.common import check_for_bool
from sklearn.model_selection import train_test_split
import numpy as np

class NumericalImputation(AutoSklearnPreprocessingAlgorithm):

    def __init__(self, strategy='mean',
                       n_neighbors=5,
                       weights='uniform',
                       max_iter=5,
                       sample_posterior='False',
                       tol=1e-3,
                       initial_strategy="mean",
                       imputation_order="ascending",
                       skip_complete='False',
                       training_fraction=0.5,
                   random_state=None):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.max_iter = max_iter
        self.sample_posterior = check_for_bool(sample_posterior)
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.skip_complete = check_for_bool(skip_complete)
        self.training_fraction = training_fraction

        self.random_state = random_state


    def fit(self, X, y=None):
        import sklearn.impute

        if self.strategy in ["mean", "median", "most_frequent"]:
            self.preprocessor = sklearn.impute.SimpleImputer(
                strategy=self.strategy, copy=False)
        elif self.strategy == 'knn':
            self.preprocessor = sklearn.impute.KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights, copy=False)
        elif self.strategy == 'iterative':
            self.preprocessor = IterativeImputer(sample_posterior=self.sample_posterior, max_iter=self.max_iter, skip_complete=self.skip_complete, imputation_order=self.imputation_order, initial_strategy=self.initial_strategy, tol=self.tol)
        else:
            pass

        X_new = None
        try:
            min_training_instances = max([self.training_fraction*len(X), 10*len(np.unique(y)), self.n_neighbors + 1])
            X_new, _, _, _ = train_test_split(X, y, train_size=min_training_instances, random_state=42)
        except:
            X_new = X

        #X_new = X
        self.preprocessor.fit(X_new)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'NumericalImputation',
                'name': 'Numerical Imputation',
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
                # TODO find out if this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        strategy = CategoricalHyperparameter("strategy", ["mean", "median", "most_frequent", "knn", "iterative"], default_value="mean")
        cs.add_hyperparameter(strategy)


        #knn hyperparameters
        n_neighbors = UniformIntegerHyperparameter(name="n_neighbors", lower=2, upper=100, log=True, default_value=5)
        weights = CategoricalHyperparameter(name="weights", choices=["uniform", "distance"], default_value="uniform")
        cs.add_hyperparameters([n_neighbors, weights])

        n_neighbors_depends_on_knn = EqualsCondition(n_neighbors, strategy, "knn")
        weights_depends_on_knn = EqualsCondition(weights, strategy, "knn")
        cs.add_conditions([n_neighbors_depends_on_knn, weights_depends_on_knn])


        #iterative hyperparameters
        max_iter = UniformIntegerHyperparameter(name="max_iter", lower=1, upper=100, log=True, default_value=5)
        sample_posterior = CategoricalHyperparameter("sample_posterior", choices=["True", "False"], default_value="False")
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, log=True, default_value=1e-3)
        initial_strategy = CategoricalHyperparameter("initial_strategy", ["mean", "median", "most_frequent", "constant"], default_value="mean")
        imputation_order = CategoricalHyperparameter("imputation_order", ["ascending", "descending", "roman", "arabic", "random"], default_value="ascending")
        skip_complete = CategoricalHyperparameter("skip_complete", choices=["True", "False"], default_value="False")
        
        
        max_iter_depends_on_iterative = EqualsCondition(max_iter, strategy, "iterative")
        sample_posterior_depends_on_iterative = EqualsCondition(sample_posterior, strategy, "iterative")
        tol_depends_on_iterative = EqualsCondition(tol, strategy, "iterative")
        initial_strategy_depends_on_iterative = EqualsCondition(initial_strategy, strategy, "iterative")
        imputation_order_depends_on_iterative = EqualsCondition(imputation_order, strategy, "iterative")
        skip_complete_depends_on_iterative = EqualsCondition(skip_complete, strategy, "iterative")
        

        cs.add_hyperparameters([max_iter,
                                sample_posterior,
                                tol,
                                initial_strategy,
                                imputation_order,
                                skip_complete])
        
        cs.add_conditions([max_iter_depends_on_iterative,
                           sample_posterior_depends_on_iterative,
                           tol_depends_on_iterative,
                           initial_strategy_depends_on_iterative,
                           imputation_order_depends_on_iterative,
                           skip_complete_depends_on_iterative])



        training_fraction = UniformFloatHyperparameter("training_fraction", 0.0001, 1.0, log=True, default_value=0.5)
        cs.add_hyperparameter(training_fraction)

        return cs

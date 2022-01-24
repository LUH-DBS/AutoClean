from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
from autosklearn.util.common import check_for_bool
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from autosklearn.workaround.Workaround import Workaround
import copy
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

class DetectionComponent(AutoSklearnPreprocessingAlgorithm):

    def __init__(self, **arguments):
        self.training_fraction = arguments['training_fraction']
        self.use_outlier_detection = {}
        self.strategy = {}

        #lof
        self.use_auto_threshold = {}
        self.contamination = {}
        self.n_neighbors = {}

        #isolation_forest
        self.use_auto_threshold_iso = {}
        self.contamination_iso = {}
        self.n_estimators_iso = {}
        self.bootstrap_iso = {}

        #svm
        self.kernel_svm = {}
        self.degree_svm = {}
        self.gamma_svm = {}
        self.coef0_svm = {}
        self.tol_svm = {}
        self.shrinking_svm = {}
        self.nu_svm = {}

        #EllipticEnvelope
        self.assume_centered_ell = {}
        self.support_fraction_ell = {}
        self.contamination_ell = {}


        number_of_features = Workaround.number_of_features
        for fi in range(number_of_features):
            if "use_outlier_detection" + str(fi) in arguments:
                self.use_outlier_detection[fi] = check_for_bool(arguments["use_outlier_detection" + str(fi)])
            else:
                self.use_outlier_detection[fi] = False

            if "strategy" + str(fi) in arguments:
                self.strategy[fi] = arguments["strategy" + str(fi)]
            else:
                self.strategy[fi] = 'lof'

            #lof
            if "use_auto_threshold" + str(fi) in arguments:
                self.use_auto_threshold[fi] = check_for_bool(arguments["use_auto_threshold" + str(fi)])
            else:
                self.use_auto_threshold[fi] = True
            if "contamination" + str(fi) in arguments:
                self.contamination[fi] = arguments["contamination" + str(fi)]
            else:
                self.contamination[fi] = 0.1
            if "n_neighbors" + str(fi) in arguments:
                self.n_neighbors[fi] = arguments["n_neighbors" + str(fi)]
            else:
                self.n_neighbors[fi] = 5

            #isolation forest
            if "use_auto_threshold_iso" + str(fi) in arguments:
                self.use_auto_threshold_iso[fi] = check_for_bool(arguments["use_auto_threshold_iso" + str(fi)])
            else:
                self.use_auto_threshold_iso[fi] = True
            if "contamination_iso" + str(fi) in arguments:
                self.contamination_iso[fi] = arguments["contamination_iso" + str(fi)]
            else:
                self.contamination_iso[fi] = 0.1
            if "n_estimators_iso" + str(fi) in arguments:
                self.n_estimators_iso[fi] = arguments["n_estimators_iso" + str(fi)]
            else:
                self.n_estimators_iso[fi] = 10
            if "bootstrap_iso" + str(fi) in arguments:
                self.bootstrap_iso[fi] = check_for_bool(arguments["bootstrap_iso" + str(fi)])
            else:
                self.bootstrap_iso[fi] = False

            #one class svm
            if "kernel_svm" + str(fi) in arguments:
                self.kernel_svm[fi] = arguments["kernel_svm" + str(fi)]
            else:
                self.kernel_svm[fi] = 'rbf'
            if "degree_svm" + str(fi) in arguments:
                self.degree_svm[fi] = arguments["degree_svm" + str(fi)]
            else:
                self.degree_svm[fi] = 3
            if "gamma_svm" + str(fi) in arguments:
                self.gamma_svm[fi] = arguments["gamma_svm" + str(fi)]
            else:
                self.gamma_svm[fi] = 0.1
            if "coef0_svm" + str(fi) in arguments:
                self.coef0_svm[fi] = arguments["coef0_svm" + str(fi)]
            else:
                self.coef0_svm[fi] = 0
            if "tol_svm" + str(fi) in arguments:
                self.tol_svm[fi] = arguments["tol_svm" + str(fi)]
            else:
                self.tol_svm[fi] = 1e-3
            if "shrinking_svm" + str(fi) in arguments:
                self.shrinking_svm[fi] = check_for_bool(arguments["shrinking_svm" + str(fi)])
            else:
                self.shrinking_svm[fi] = True
            if "nu_svm" + str(fi) in arguments:
                self.nu_svm[fi] = arguments["nu_svm" + str(fi)]
            else:
                self.nu_svm[fi] = 0.5

            # EllipticEnvelope
            if "assume_centered_ell" + str(fi) in arguments:
                self.assume_centered_ell[fi] = check_for_bool(arguments["assume_centered_ell" + str(fi)])
            else:
                self.assume_centered_ell[fi] = False
            if "support_fraction_ell" + str(fi) in arguments:
                self.support_fraction_ell[fi] = arguments["support_fraction_ell" + str(fi)]
            else:
                self.support_fraction_ell[fi] = 0.5
            if "contamination_ell" + str(fi) in arguments:
                self.contamination_ell[fi] = arguments["contamination_ell" + str(fi)]
            else:
                self.contamination_ell[fi] = 0.1


    def fit(self, X, y=None):
        self.feature2model = {}

        X_new = None
        try:
            my_list_all = [self.training_fraction * len(X), 10 * len(np.unique(y))]
            for fi in range(X_new.shape[1]):
                my_list_all.append(self.n_neighbors[fi] + 1)
            min_training_instances = max(my_list_all)
            X_new, _, _, _ = train_test_split(X, y, train_size=min_training_instances, random_state=42)
        except:
            X_new = X

        #fit LOF model for each feature
        for fi in range(X_new.shape[1]):
            if self.use_outlier_detection[fi]:
                detection_model =None
                if self.strategy[fi] == 'lof':
                    contamination = 'auto'
                    if self.use_auto_threshold[fi] == False:
                        contamination = self.contamination[fi]
                    detection_model = LocalOutlierFactor(contamination=contamination, n_neighbors=self.n_neighbors[fi], novelty=True)
                elif self.strategy[fi] == 'isolation_forest':
                    contamination_iso = 'auto'
                    if self.use_auto_threshold_iso[fi] == False:
                        contamination_iso = self.contamination_iso[fi]
                    detection_model = IsolationForest(contamination=contamination_iso, n_estimators=self.n_estimators_iso[fi], bootstrap=self.bootstrap_iso[fi])
                elif self.strategy[fi] == "one_class_svm":
                    detection_model = OneClassSVM(kernel=self.kernel_svm[fi],
                                                  degree=self.degree_svm[fi],
                                                  gamma=self.gamma_svm[fi],
                                                  coef0=self.coef0_svm[fi],
                                                  tol=self.tol_svm[fi],
                                                  shrinking=self.shrinking_svm[fi],
                                                  nu=self.nu_svm[fi])
                elif self.strategy[fi] == "elliptic":
                    detection_model = EllipticEnvelope(assume_centered=self.assume_centered_ell[fi],
                                                       support_fraction=self.support_fraction_ell[fi],
                                                       contamination=self.contamination_ell[fi])

                indices = np.argwhere(~np.isnan(X_new[:, fi]))
                detection_model.fit(X_new[indices, fi])
                self.feature2model[fi] = detection_model
        return self

    def transform(self, X):
        new_X = copy.deepcopy(X)
        for fi in range(X.shape[1]):
            if self.use_outlier_detection[fi]:
                #turn outliers to NaNs
                indices = np.argwhere(~np.isnan(X[:, fi]))
                outlier_indices = np.where(self.feature2model[fi].predict(X[indices, fi].reshape(-1, 1)) == -1)[0]
                new_X[indices[outlier_indices], fi] = np.NaN
        return new_X

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
        number_of_features = Workaround.number_of_features
        for fi in range(number_of_features):
            use_outlier_detection_fi = CategoricalHyperparameter("use_outlier_detection" + str(fi), choices=["True", "False"], default_value="False")
            cs.add_hyperparameter(use_outlier_detection_fi)

            strategy_fi = CategoricalHyperparameter("strategy" + str(fi), choices=["lof", "isolation_forest", "one_class_svm", "elliptic"], default_value="lof")
            cs.add_hyperparameter(strategy_fi)
            strategy_depends_on_detection = EqualsCondition(strategy_fi, use_outlier_detection_fi, "True")
            cs.add_condition(strategy_depends_on_detection)

            #lof parameters
            use_auto_threshold_fi = CategoricalHyperparameter("use_auto_threshold" + str(fi), choices=["True", "False"], default_value="True")
            cs.add_hyperparameter(use_auto_threshold_fi)
            use_auto_threshold_depends_on_lof = EqualsCondition(use_auto_threshold_fi, strategy_fi, "lof")
            cs.add_condition(use_auto_threshold_depends_on_lof)

            contamination_fi = UniformFloatHyperparameter("contamination" + str(fi), 0.0001, 0.5, log=True, default_value=0.1)
            cs.add_hyperparameter(contamination_fi)
            contamination_depends_on_auto_threshold = EqualsCondition(contamination_fi, use_auto_threshold_fi, "False")
            cs.add_condition(contamination_depends_on_auto_threshold)

            n_neighbors = UniformIntegerHyperparameter(name="n_neighbors" + str(fi), lower=2, upper=100, log=True, default_value=5)
            cs.add_hyperparameter(n_neighbors)
            n_neighbors_depends_on_lof = EqualsCondition(n_neighbors, strategy_fi, "lof")
            cs.add_condition(n_neighbors_depends_on_lof)

            #isolation forest:
            use_auto_threshold_fi = CategoricalHyperparameter("use_auto_threshold_iso" + str(fi), choices=["True", "False"], default_value="True")
            cs.add_hyperparameter(use_auto_threshold_fi)
            use_auto_threshold_depends_on_lof = EqualsCondition(use_auto_threshold_fi, strategy_fi, "isolation_forest")
            cs.add_condition(use_auto_threshold_depends_on_lof)

            contamination_fi = UniformFloatHyperparameter("contamination_iso" + str(fi), 0.0001, 0.5, log=True, default_value=0.1)
            cs.add_hyperparameter(contamination_fi)
            contamination_depends_on_auto_threshold = EqualsCondition(contamination_fi, use_auto_threshold_fi, "False")
            cs.add_condition(contamination_depends_on_auto_threshold)

            n_estimators = UniformIntegerHyperparameter(name="n_estimators_iso" + str(fi), lower=1, upper=200, log=True, default_value=10)
            cs.add_hyperparameter(n_estimators)
            n_estimators_depends_on_lof = EqualsCondition(n_estimators, strategy_fi, "isolation_forest")
            cs.add_condition(n_estimators_depends_on_lof)

            bootstrap_fi = CategoricalHyperparameter("bootstrap_iso" + str(fi), choices=["True", "False"],
                                                     default_value="False")
            cs.add_hyperparameter(bootstrap_fi)
            bootstrap_depends_on_lof = EqualsCondition(bootstrap_fi, strategy_fi, "isolation_forest")
            cs.add_condition(bootstrap_depends_on_lof)

            #one_class_svm
            kernel_svm = CategoricalHyperparameter(name="kernel_svm" + str(fi), choices=["rbf", "poly", "sigmoid"], default_value="rbf")
            cs.add_hyperparameter(kernel_svm)
            kernel_svm_depends_on_svm = EqualsCondition(kernel_svm, strategy_fi, "one_class_svm")
            cs.add_condition(kernel_svm_depends_on_svm)

            degree_svm = UniformIntegerHyperparameter("degree_svm" + str(fi), 2, 5, default_value=3)
            cs.add_hyperparameter(degree_svm)
            degree_svm_depends_on_poly = EqualsCondition(degree_svm, kernel_svm, "poly")
            cs.add_condition(degree_svm_depends_on_poly)

            gamma_svm = UniformFloatHyperparameter("gamma_svm" + str(fi), 3.0517578125e-05, 8, log=True, default_value=0.1)
            cs.add_hyperparameter(gamma_svm)
            gamma_svm_depends_on_svm = EqualsCondition(gamma_svm, strategy_fi, "one_class_svm")
            cs.add_condition(gamma_svm_depends_on_svm)

            coef0_svm = UniformFloatHyperparameter("coef0_svm" + str(fi), -1, 1, default_value=0)
            cs.add_hyperparameter(coef0_svm)
            #coef0_svm_depends_on_poly_sig = InCondition(coef0_svm, kernel_svm, ["poly", "sigmoid"])
            coef0_svm_depends_on = EqualsCondition(coef0_svm, strategy_fi, "one_class_svm")
            cs.add_condition(coef0_svm_depends_on)

            tol_svm = UniformFloatHyperparameter("tol_svm"+ str(fi), 1e-5, 1e-1, default_value=1e-3, log=True)
            cs.add_hyperparameter(tol_svm)
            tol_svm_depends_on_svm = EqualsCondition(tol_svm, strategy_fi, "one_class_svm")
            cs.add_condition(tol_svm_depends_on_svm)

            shrinking_svm = CategoricalHyperparameter("shrinking_svm"+ str(fi), ["True", "False"], default_value="True")
            cs.add_hyperparameter(shrinking_svm)
            shrinking_svm_depends_on_svm = EqualsCondition(shrinking_svm, strategy_fi, "one_class_svm")
            cs.add_condition(shrinking_svm_depends_on_svm)

            nu_svm = UniformFloatHyperparameter("nu_svm" + str(fi), 0.0, 1.0, log=False, default_value=0.5)
            cs.add_hyperparameter(nu_svm)
            nu_svm_depends_on_svm = EqualsCondition(nu_svm, strategy_fi, "one_class_svm")
            cs.add_condition(nu_svm_depends_on_svm)

            #EllipticEnvelope
            assume_centered_ell = CategoricalHyperparameter("assume_centered_ell" + str(fi), ["True", "False"], default_value="False")
            cs.add_hyperparameter(assume_centered_ell)
            assume_centered_ell_depends_on_ell = EqualsCondition(assume_centered_ell, strategy_fi, "elliptic")
            cs.add_condition(assume_centered_ell_depends_on_ell)

            support_fraction_ell = UniformFloatHyperparameter("support_fraction_ell" + str(fi), 0.0, 1.0, log=False, default_value=0.5)
            cs.add_hyperparameter(support_fraction_ell)
            support_fraction_ell_depends_on_ell = EqualsCondition(support_fraction_ell, strategy_fi, "elliptic")
            cs.add_condition(support_fraction_ell_depends_on_ell)

            contamination_ell = UniformFloatHyperparameter("contamination_ell" + str(fi), 0.0001, 0.5, log=True, default_value=0.1)
            cs.add_hyperparameter(contamination_ell)
            contamination_ell_depends_on_ell = EqualsCondition(contamination_ell, strategy_fi, "elliptic")
            cs.add_condition(contamination_ell_depends_on_ell)

        training_fraction = UniformFloatHyperparameter("training_fraction", 0.0001, 1.0, log=True, default_value=0.5)
        cs.add_hyperparameter(training_fraction)

        return cs

# -*- coding: utf-8 -*-
import re

import socket
import os
import pickle
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, UnParametrizedHyperparameter
import copy


class Config:
    config = dict()#pickle.load(open('/tmp/space.p', 'rb')) #dict()

    classifiers_copy = dict()
    preprocessors_copy = dict()
    ohes_copy = dict()
    mcs_copy = dict()
    rescalers_copy = dict()


    @staticmethod
    def get(key):
        if key in Config.config:
            #print(key + ': ' + str(Config.config[key].status))
            return Config.config[key].status
        else:
            return True

    @staticmethod
    def check_value(value, hyperparameter):
        if isinstance(hyperparameter, CategoricalHyperparameter):
            return value in hyperparameter.choices
        else:
            #return hyperparameter.value==value
            return False

    @staticmethod
    def config_module(module_components_copy, default_element):
        module_components = copy.deepcopy(module_components_copy)
        remove_keys = []
        for k, v in module_components.items():
            if not Config.get(k):
                remove_keys.append(k)

        # default value
        if len(remove_keys) == len(module_components):
            remove_keys.remove(default_element)

        for k in remove_keys:
            del module_components[k]
        return module_components

    @staticmethod
    def setup():
        import autosklearn.pipeline.components.classification
        import autosklearn.pipeline.components.feature_preprocessing
        import autosklearn.pipeline.components.data_preprocessing.categorical_encoding
        import autosklearn.pipeline.components.data_preprocessing.minority_coalescense
        import autosklearn.pipeline.components.data_preprocessing.rescaling

        if len(Config.classifiers_copy) == 0:
            Config.classifiers_copy = copy.deepcopy(autosklearn.pipeline.components.classification._classifiers)
            Config.preprocessors_copy = copy.deepcopy(autosklearn.pipeline.components.feature_preprocessing._preprocessors)
            Config.ohes_copy = copy.deepcopy(autosklearn.pipeline.components.data_preprocessing.categorical_encoding._ohes)
            Config.mcs_copy = copy.deepcopy(autosklearn.pipeline.components.data_preprocessing.minority_coalescense._mcs)
            Config.rescalers_copy = copy.deepcopy(autosklearn.pipeline.components.data_preprocessing.rescaling._rescalers)

        autosklearn.pipeline.components.classification._classifiers = Config.config_module(Config.classifiers_copy, 'random_forest')
        autosklearn.pipeline.components.feature_preprocessing._preprocessors = Config.config_module(Config.preprocessors_copy, 'no_preprocessing')
        autosklearn.pipeline.components.data_preprocessing.categorical_encoding._ohes = Config.config_module(Config.ohes_copy, 'one_hot_encoding')
        autosklearn.pipeline.components.data_preprocessing.minority_coalescense._mcs = Config.config_module(Config.mcs_copy, 'no_coalescense')
        autosklearn.pipeline.components.data_preprocessing.rescaling._rescalers = Config.config_module(Config.rescalers_copy, 'none')


    @staticmethod
    def get_value(class_name, hyperparameter):
        if Config.get(class_name + hyperparameter.name) == True:
            if isinstance(hyperparameter, CategoricalHyperparameter):
                list_of_choices = []
                for element in hyperparameter.choices:
                    if Config.get(class_name + hyperparameter.name + '##' + str(element)) == True:
                        list_of_choices.append(element)
                if len(list_of_choices) == 0:
                    return Constant(name=hyperparameter.name, value=hyperparameter.default_value)
                if len(list_of_choices) == 1:
                    return Constant(name=hyperparameter.name, value=list_of_choices[0])

                hyperparameter.choices = tuple(list_of_choices)
                if not hyperparameter.default_value in hyperparameter.choices:
                    hyperparameter.default_value = hyperparameter.choices[0]

            return hyperparameter
        else:
            return Constant(name=hyperparameter.name, value=hyperparameter.default_value)



# -*- coding: utf-8 -*-
import re

import socket
import os
import pickle
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, UnParametrizedHyperparameter


class Config:
    config = dict()#pickle.load(open('/tmp/space.p', 'rb')) #dict()

    @staticmethod
    def get(key):
        print('dict length: ' + str(len(Config.config)))
        return Config.config[key].status

    @staticmethod
    def check_value(value, hyperparameter):
        if isinstance(hyperparameter, CategoricalHyperparameter):
            return value in hyperparameter.choices
        else:
            #return hyperparameter.value==value
            return False

    @staticmethod
    def get_value(class_name, hyperparameter):
        if Config.get(class_name + hyperparameter.name) == True:
            if isinstance(hyperparameter, CategoricalHyperparameter):
                list_of_choices = []
                for element in hyperparameter.choices:
                    if Config.get(class_name + hyperparameter.name + '##' + element) == True:
                        list_of_choices.append(element)
                hyperparameter.choices = tuple(list_of_choices)
                if not hyperparameter.default_value in hyperparameter.choices:
                    hyperparameter.default_value = hyperparameter.choices[0]

                if len(list_of_choices) == 0:
                    return Constant(name=hyperparameter.name, value=hyperparameter.default_value)
                if len(list_of_choices) == 1:
                    return Constant(name=hyperparameter.name, value=list_of_choices[0])

            return hyperparameter
        else:
            return Constant(name=hyperparameter.name, value=hyperparameter.default_value)



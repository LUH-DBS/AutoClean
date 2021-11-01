# -*- coding: utf-8 -*-
import re

import socket
import os
import pickle
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, UnParametrizedHyperparameter


class Config:
    config = dict()

    file_path = '' #os.path.dirname(os.path.realpath(__file__)) + "/resources/space.properties"

    @staticmethod
    def load():
        with open(Config.file_path) as f:
            for line in f:
                splits = re.split('=|\n', line)
                Config.config[splits[0]] = splits[1] == 'True'

    @staticmethod
    def get(key):
        if len(Config.config) == 0:
            Config.load()
        return Config.config[key]

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



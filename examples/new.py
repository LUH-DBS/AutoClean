import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas as pd
from autosklearn.metrics import balanced_accuracy
from autosklearn.workaround.Workaround import Workaround
from sklearn.metrics import balanced_accuracy_score
import autosklearn.classification
from sklearn import preprocessing
import numpy as np

def get_X_y(data, target_label, drop_labels=[]):
    data_y = data[target_label]
    data_X = data.drop(target_label, 1)
    for drop_label in drop_labels:
        data_X = data_X.drop(drop_label, 1)
    return data_y, data_X

def to_str(data):
    return str(data)

def get_feat_type(data, target_label, drop_labels=[]):
    data_y, data_X = get_X_y(data, target_label, drop_labels)
    print(data_X.dtypes)
    feat_type = [
        'Categorical' if str(x) == 'object' else 'Numerical'
        for x in data_X.dtypes
    ]
    return feat_type


def main():
    #drop_labels = []
    #target = "SeriousDlqin2yrs"
    #dirty_path = '/home/neutatz/Software/CleanML/data/Credit/missing_values/dirty_train.csv'
    #clean_path = '/home/neutatz/Software/CleanML/data/Credit/missing_values/impute_holoclean_train.csv'

    target = "Income"
    drop_labels = []
    dirty_path = '/home/neutatz/Software/CleanML/data/USCensus/missing_values/dirty_train.csv'
    clean_path = '/home/neutatz/Software/CleanML/data/USCensus/missing_values/impute_holoclean_train.csv'

    data = pd.read_csv(dirty_path)
    data_y, data_X = get_X_y(data, target, drop_labels=drop_labels)

    clean_data = pd.read_csv(clean_path)
    feat_type = get_feat_type(clean_data, target, drop_labels=drop_labels)
    print(feat_type)

    for ci in range(len(feat_type)):
        if feat_type[ci] == 'Categorical':
            data_X[data_X.columns[ci]] = data_X[data_X.columns[ci]].apply(to_str)

    data_X_val = data_X.values

    for ci in range(len(feat_type)):
        if feat_type[ci] == 'Categorical':
            my_encoder = preprocessing.LabelEncoder()
            data_X_val[:, ci] = my_encoder.fit_transform(data_X_val[:, ci])
            for class_i in range(len(my_encoder.classes_)):
                if my_encoder.classes_[class_i] == 'nan':
                    data_X_val[data_X_val[:, ci] == class_i, ci] = np.NaN

    y_val = preprocessing.LabelEncoder().fit_transform(data_y.values)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data_X_val, y_val, random_state=1)

    Workaround.number_of_features = np.sum(np.array(feat_type) == 'Numerical')
    print('number of numerical: ' + str(Workaround.number_of_features))

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=60,
        delete_tmp_folder_after_terminate=False,
        initial_configurations_via_metalearning=0,
        n_jobs=5,
        ml_memory_limit=200072,
        ensemble_memory_limit=200072,
    )

    automl.fit(X_train.copy(), y_train.copy(), metric=balanced_accuracy, feat_type=feat_type)
    automl.refit(X_train.copy(), y_train.copy())

    print(automl.sprint_statistics())
    print(automl.show_models())

    predictions = automl.predict(X_test.copy())
    print("Accuracy score", balanced_accuracy_score(y_test, predictions))

if __name__ == '__main__':
    main()

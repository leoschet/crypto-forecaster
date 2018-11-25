import codecs
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR

from . import classifiers, retriever, timeseries
from .gcForest import GCForest, config

# TODO: Create config json
#       - Use data header only
cryptocurrencies = ['bitcoin']
possible_headers = [
    'very_positive_reply', 'very_positive_topic', 'positive_reply', 
    'positive_topic', 'neutral_reply', 'neutral_topic', 'negative_reply', 
    'negative_topic', 'very_negative_reply', 'very_negative_topic', 'total_reply', 
    'total_topic', 'transactions', 'views'
]

combinations = []
combinations.extend(list(itertools.combinations(possible_headers, 5)))

label = 'price'

## Classifiers
aode = classifiers.AODE()
svclassifier = SVC(C=100000, kernel='rbf', degree=3, gamma=0.00001, probability=True)
mlp = MLPClassifier(learning_rate_init=0.001, max_iter=4000)
rfc = RandomForestClassifier(
    n_estimators=200, criterion='gini', max_depth=30,
    n_jobs=-1, min_samples_leaf=1
)

# GcForest
gc_config = config.load_config()

## Regressors
# Lower gamma helps higher lags
svregressor = SVR(C=100, kernel='rbf', degree=8, gamma=0.001)

def plot(timeseries, prediction):
    fig = plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 1)
    plt.title('Análise de Tendência')
    plt.plot(timeseries, label='Original series')
    plt.plot(prediction, 'r--', label='Prediction')
    plt.ylabel('z-score')
    plt.legend(loc='best')

    plt.show()

results = {}

for cryptocurrency in cryptocurrencies:
    # Print some info
    print('\nTesting for ' + cryptocurrency + ' data set')

    # Edit results dict
    results[cryptocurrency] = {}

    # Collect data
    df = retriever.get_data(cryptocurrency)

    # Get all headers but label headers
    features_headers = list(
        set(df.columns.values).symmetric_difference(
            [label] + ['transactions']
        )
    )

    # Standardize features series
    df[features_headers] = df[features_headers].apply(stats.zscore)

    # Edit results dict
    results[cryptocurrency][label] = {}

    # Extract label data
    labels = np.array(df[label])[1250:]
    
    for comb in combinations:
        comb = list(comb)
        # Extract features data
        data = np.array(df[comb])[1250:]

        # Print some status
        data_len = len(data)
        features_amount = data[0].size
        print('\tTotal data collected: ' + str(data_len))
        print('\tTotal of features per data: ' + str(features_amount))

        features = '+'.join(comb)

        # Edit results dict
        results[cryptocurrency][label][features] = {}

        for lag in range(1, 10):
            # Print some info
            print('\n\n\t\tPredicting with lag =', str(lag))

            # Edit results dict
            results[cryptocurrency][label][features][lag] = {}

            # Get rolling windows of data
            features_windows = timeseries.window_stack(data, window=lag, spatial=False)

            # Cut off unused labels, a.k.a.: before lag labels
            lagged_labels = labels[(lag - 1):]

            # assert both have same len
            assert len(features_windows) == len(lagged_labels)

            # Train/test partition point
            partition = int(len(features_windows) * 0.75)

            # Divide into train and test
            train_data, test_data = features_windows[:partition], features_windows[partition:]
            train_labels, test_labels = lagged_labels[:partition], lagged_labels[partition:]

            # ## GC FOREST
            # # Modify data for gcforest
            # train_data = train_data[:, np.newaxis, :, :]
            # test_data = test_data[:, np.newaxis, :, :]

            # # Modify labels for gcforest
            # lagged_labels = [int(float(l)) for l in lagged_labels]
            # test_labels = [int(float(l)) for l in test_labels]

            # lagged_labels = [0 if l == -1 else l for l in lagged_labels]
            # test_labels = [0 if l == -1 else l for l in test_labels]

            # # Set finegraining args for gcforest
            # series_amount = len(comb)
            # if lag > 31:
            #     small_win = int(lag/16)
            #     medium_win = int(lag/8)
            #     large_win = int(lag/4)
            #     stride = 2
            # elif lag > 15:
            #     small_win = int(lag/8)
            #     medium_win = int(lag/4)
            #     large_win = int(lag/2)
            #     stride = 2
            # elif lag > 7:
            #     small_win = int(lag/6)
            #     medium_win = int(lag/3)
            #     large_win = int(lag/2)
            #     stride = 1
            # else:
            #     small_win = 2
            #     medium_win = 3
            #     large_win = 4
            #     stride = 1

            # gc_config = config.set_finegraining_args(
            #     gc_config, series_amount, small_win, medium_win, large_win, stride
            # )

            # # Create gcforest classifier
            # gc = GCForest(gc_config)

            # Execute 30 times for each lag
            for execution in range(10):
                # Print some info
                print('\t\t\tExecution =', str(execution))

                # fit the classifier
                # aode.fit(train_data, train_labels, online=False)
                # svclassifier.fit(train_data, train_labels)
                # mlp.fit(train_data, train_labels)
                rfc.fit(train_data, train_labels)
                # svregressor.fit(train_data, train_labels)
                # gc.fit_transform(train_data, train_labels)

                # predict
                pred_labels = []
                # for element in test_data:
                #     pred_labels.append(aode.predict(element, estimation=estimation))
                # pred_labels = svclassifier.predict(test_data)
                # pred_labels = mlp.predict(test_data)
                pred_labels = rfc.predict(test_data)
                # pred_labels = svregressor.predict(test_data)
                # pred_labels = gc.predict(test_data)

                # Regressor execution:
                # plot(list(test_labels), list(pred_labels))
                # continue

                # Calculte prediction metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    test_labels, pred_labels, average='weighted'
                )
                accuracy = accuracy_score(test_labels, pred_labels)

                if 'precision' not in results[cryptocurrency][label][features][lag]:
                    results[cryptocurrency][label][features][lag]['precision'] = []
                    results[cryptocurrency][label][features][lag]['recall'] = []
                    results[cryptocurrency][label][features][lag]['f1'] = []
                    results[cryptocurrency][label][features][lag]['accuracy'] = []

                    set_labels = list(set(test_labels))
                    np_test_labels = np.array(test_labels)
                    results[cryptocurrency][label][features][lag]['test_labels'] = [
                        (l, np.where(np_test_labels == l)[0].size) for l in set_labels
                    ]
                    results[cryptocurrency][label][features][lag]['pred_labels'] = []

                results[cryptocurrency][label][features][lag]['precision'].append(precision)
                results[cryptocurrency][label][features][lag]['recall'].append(recall)
                results[cryptocurrency][label][features][lag]['f1'].append(f1)
                results[cryptocurrency][label][features][lag]['accuracy'].append(accuracy)

                set_labels = list(set(pred_labels))
                np_pred_labels = np.array(pred_labels)
                results[cryptocurrency][label][features][lag]['pred_labels'].append(
                    [
                        (l, np.where(np_pred_labels == l)[0].size) for l in set_labels
                    ]
                )

            # Overwrite the file every lag
            file_name = 'res/forecasting_results/' + cryptocurrency + '_gridsearch.txt'
            file_ = codecs.open(file_name, 'w+', 'utf-8')
            file_.write(json.dumps(results, indent=4))

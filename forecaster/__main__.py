import codecs
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
estimations = ['paper']# , 'count'
data_headers = {
    'bitcoin': {
        'price': ['positive_topic', 'positive_reply', 'very_positive_topic', 'very_positive_reply', 'total_topic', 'total_reply', 'negative_reply'],
        # 'price': ['positive_topic', 'total_topic', 'positive_reply', 'total_reply'],
        'transactions': ['total_topic', 'very_positive_topic', 'very_positive_reply'],
    }
}
label_headers = ['price'] #, 'transactions'

## Classifiers
aode = classifiers.AODE()
# svclassifier = SVC(C=10000, kernel='rbf', degree=8, gamma=0.0001, probability=True)
svclassifier = SVC(C=100000, kernel='rbf', degree=3, gamma=0.00001, probability=True)
# svclassifier = SVC(C=1000, kernel='rbf', degree=3, gamma=0.0001, probability=True)
mlp = MLPClassifier(learning_rate_init=0.001, max_iter=4000) # hidden_layer_sizes=(60, 100, 40), learning_rate='adaptive',
rfc = RandomForestClassifier(
    n_estimators=150, criterion='gini', max_depth=30,
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
            label_headers + ['transactions']
        )
    )

    # Standardize features series
    df[features_headers] = df[features_headers].apply(stats.zscore)

    for label in label_headers:
        # Edit results dict
        results[cryptocurrency][label] = {}

        # Extract features and labels data
        data = np.array(df[data_headers[cryptocurrency][label]])[1250:]
        labels = np.array(df[label])[1250:]

        # Print some status
        data_len = len(data)
        features_amount = data[0].size
        print('\tTotal data collected: ' + str(data_len))
        print('\tTotal of features per data: ' + str(features_amount))

        for estimation in estimations:
            # Print some info
            print('\tEstimating probabilities on \'' + estimation + '\' mode')

            # Edit results dict
            results[cryptocurrency][label][estimation] = {}

            for lag in range(19, 51):
                # Edit results dict
                results[cryptocurrency][label][estimation][lag] = {}

                # Print some info
                print('\n\n\t\tPredicting with lag =', str(lag))

                # Execute 30 times for each lag
                for execution in range(15):
                    # Print some info
                    print('\t\t\tExecution =', str(execution))

                    # Get rolling windows of data
                    features_windows = timeseries.window_stack(data, window=lag, spatial=True)

                    # Cut off unused labels, a.k.a.: before lag labels
                    lagged_labels = labels[(lag - 1):]

                    # assert both have same len
                    assert len(features_windows) == len(lagged_labels)

                    # Train/test partition point
                    partition = int(len(features_windows) * 0.75)

                    # Divide into train and test
                    train_data, test_data = features_windows[:partition], features_windows[partition:]
                    train_labels, test_labels = lagged_labels[:partition], lagged_labels[partition:]

                    ## GC FOREST
                    # Modify data for gcforest
                    train_data = train_data[:, np.newaxis, :, :]
                    test_data = test_data[:, np.newaxis, :, :]

                    # Modify labels for gcforest
                    lagged_labels = [int(float(l)) for l in lagged_labels]
                    test_labels = [int(float(l)) for l in test_labels]

                    lagged_labels = [0 if l == -1 else l for l in lagged_labels]
                    test_labels = [0 if l == -1 else l for l in test_labels]

                    # Set finegraining args for gcforest
                    series_amount = len(data_headers[cryptocurrency][label])
                    if lag > 31:
                        small_win = int(lag/16)
                        medium_win = int(lag/8)
                        large_win = int(lag/4)
                        stride = 2
                    elif lag > 15:
                        small_win = int(lag/8)
                        medium_win = int(lag/4)
                        large_win = int(lag/2)
                        stride = 2
                    elif lag > 7:
                        small_win = int(lag/6)
                        medium_win = int(lag/3)
                        large_win = int(lag/2)
                        stride = 1
                    else:
                        small_win = 2
                        medium_win = 3
                        large_win = 4
                        stride = 1

                    gc_config = config.set_finegraining_args(
                        gc_config, series_amount, small_win, medium_win, large_win, stride
                    )

                    # Create gcforest classifier
                    gc = GCForest(gc_config)

                    # fit the classifier
                    # aode.fit(train_data, train_labels, online=False)
                    # svclassifier.fit(train_data, train_labels)
                    # mlp.fit(train_data, train_labels)
                    # rfc.fit(train_data, train_labels)
                    # svregressor.fit(train_data, train_labels)
                    gc.fit_transform(train_data, train_labels)

                    # predict
                    pred_labels = []
                    # for element in test_data:
                    #     pred_labels.append(aode.predict(element, estimation=estimation))
                    # pred_labels = svclassifier.predict(test_data)
                    # pred_labels = mlp.predict(test_data)
                    # pred_labels = rfc.predict(test_data)
                    # pred_labels = svregressor.predict(test_data)
                    pred_labels = gc.predict(test_data)

                    # Regressor execution:
                    # plot(list(test_labels), list(pred_labels))
                    # continue

                    # Calculte prediction metrics
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        test_labels, pred_labels, average='weighted'
                    )
                    accuracy = accuracy_score(test_labels, pred_labels)

                    if 'precision' not in results[cryptocurrency][label][estimation][lag]:
                        results[cryptocurrency][label][estimation][lag]['precision'] = []
                        results[cryptocurrency][label][estimation][lag]['recall'] = []
                        results[cryptocurrency][label][estimation][lag]['f1'] = []
                        results[cryptocurrency][label][estimation][lag]['accuracy'] = []
                        results[cryptocurrency][label][estimation][lag]['test_labels'] = [
                            str(label) for label in test_labels
                        ]
                        results[cryptocurrency][label][estimation][lag]['pred_labels'] = []

                    results[cryptocurrency][label][estimation][lag]['precision'].append(precision)
                    results[cryptocurrency][label][estimation][lag]['recall'].append(recall)
                    results[cryptocurrency][label][estimation][lag]['f1'].append(f1)
                    results[cryptocurrency][label][estimation][lag]['accuracy'].append(accuracy)
                    results[cryptocurrency][label][estimation][lag]['pred_labels'].append(
                        [str(pred) for pred in pred_labels]
                    )

                # Overwrite the file every lag
                file_name = 'res/forecasting_results/' + cryptocurrency + '.txt'
                file_ = codecs.open(file_name, 'w+', 'utf-8')
                file_.write(json.dumps(results, indent=4))

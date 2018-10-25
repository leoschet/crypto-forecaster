import codecs

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVR
import matplotlib.pyplot as plt

from . import classifiers, retriever, timeseries
from .gcForest import config
from .gcForest import GCForest

# TODO: Create config json
#       - Use data header only
cryptocurrencies = ['bitcoin']
estimations = ['count']#, 'paper'
data_headers = {
    'bitcoin': {
        # 'price': ['positive_topic', 'total_topic', 'positive_reply', 'very_positive_topic'],
        'price': ['positive_topic', 'total_topic', 'positive_reply', 'total_reply'],
        # 'price': ['positive_topic', 'total_topic', 'positive_reply', 'very_positive_topic', 'total_reply'],
        # 'price': ['today_price'],
        'transactions': ['total_topic', 'very_positive_topic', 'very_positive_reply'],
    }
}
label_headers = ['price', 'transactions']

## Classifiers
aode = classifiers.AODE()
svclassifier = SVC(C=0.5, kernel='rbf', degree=8, gamma=0.01, probability=True)
mlp = MLPClassifier(learning_rate='adaptive', learning_rate_init=0.001)
rfc = RandomForestClassifier(
    n_estimators=800, criterion='gini', max_depth=30, random_state=0,
    n_jobs=-1, min_samples_leaf=1
)

# GcForest
gc_config = config.load_config()

## Regressors
# Lower gamma helps higher lags
svregressor = SVR(C=100, kernel='rbf', degree=8, gamma=0.001)

kf = KFold(n_splits=10, shuffle=True) # 90% for training, 10% for testing

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
    print('\nTesting for ' + cryptocurrency + ' data set')

    file_name = 'res/forecasting_results/' + cryptocurrency + '.txt'
    file_ = codecs.open(file_name, 'w+', 'utf-8')

    results[cryptocurrency] = {}

    # Collect data
    df = retriever.get_data(cryptocurrency)

    # Get all headers but label headers
    features_headers = list(set(df.columns.values).symmetric_difference(label_headers))

    # Standardize features series
    df[features_headers] = df[features_headers].apply(stats.zscore)
    # df = df.apply(stats.zscore) # Useful for regressor?
    # df = df.diff() # Useful for regressor?

    # Shift label headers so features can predict future values
    # df[label_headers] = df[label_headers].diff() # Useful for regressor?
    df[label_headers] = df[label_headers].shift(-1)

    # Drop rows with nan values
    df = df.dropna()

    for label in label_headers:
        results[cryptocurrency][label] = {}

        # Extract features and labels data
        data = np.array(df[data_headers[cryptocurrency][label]])
        labels = np.array(df[label])

        # Print some status
        data_len = len(data)
        features_amount = data[0].size
        print('\tTotal data collected: ' + str(data_len))
        print('\tTotal of features per data: ' + str(features_amount))

        for estimation in estimations:
            print('\tEstimating probabilities on \'' + estimation + '\' mode')

            file_.write(
                'Results for \'' + label + '\' ' + cryptocurrency +
                ' estimating probabilities on \'' + estimation + '\' mode:\n'
            )

            results[cryptocurrency][label][estimation] = {}

            for lag in range(1, 51):
                print('\n\n\t\tPredicting with lag =', str(lag))
                
                # Get rolling windows of data
                features_windows = timeseries.window_stack(data, window=lag, spatial=False)

                # Cut off unused labels, a.k.a.: before lag labels
                lagged_labels = labels[(lag - 1):]

                # assert both have same len
                assert len(features_windows) == len(lagged_labels)

                # Divide into train and test
                train_data, test_data = features_windows[:697], features_windows[697:]
                train_labels, test_labels = lagged_labels[:697], lagged_labels[697:]

                ## GC FOREST
                # Modify data for gcforest
                # train_data = train_data[:, np.newaxis, :, :]
                # test_data = test_data[:, np.newaxis, :, :]

                # # Modify labels for gcforest
                # lagged_labels = [int(float(l)) for l in lagged_labels]
                # test_labels = [int(float(l)) for l in test_labels]

                # lagged_labels = [0 if l == -1 else l for l in lagged_labels]
                # test_labels = [0 if l == -1 else l for l in test_labels]

                # # Set finegraining args for gcforest
                # series_amount = len(data_headers[cryptocurrency][label])
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

                # TODO: Print confusion matrix
                print(
                    '\t\t\t',
                    classification_report(test_labels, pred_labels).replace('\n', '\n\t\t\t')
                )
                print('\t\t\tAccuracy:', accuracy)

                #####

                results[cryptocurrency][label][estimation]['precision'] = precision
                results[cryptocurrency][label][estimation]['recall'] = recall
                results[cryptocurrency][label][estimation]['f1'] = f1
                results[cryptocurrency][label][estimation]['accuracy'] = accuracy

                file_.write('\tLag = ' + str(lag))

                file_.write('\n\t\tPrecisions: ' + str(precision))
                file_.write('\n\t\t\tMean: ' + str(np.mean(precision)))

                file_.write('\n\t\tRecalls: ' + str(recall))
                file_.write('\n\t\t\tMean: ' + str(np.mean(recall)))

                file_.write('\n\t\tF1s: ' + str(f1))
                file_.write('\n\t\t\tMean: ' + str(np.mean(f1)))

                file_.write('\n\t\tAccuracies: ' + str(accuracy))
                file_.write('\n\t\t\tMean: ' + str(np.mean(accuracy)) + '\n\n')

import codecs

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.svm import SVC

from . import classifiers, retriever, timeseries



def _split(data):
    ret = []
    for i in range(0, len(data), 60):
        if i+60 > len(data):
            break

        ret.append(
            (list(range(i, i+30)), list(range(i+30, i+60)))
        )
    return ret

# Auxiliary data
cryptocurrencies = ['bitcoin']
estimations = ['count', 'paper'] # 
data_headers = {
    'bitcoin': {
        'price': ['positive_topic', 'total_topic', 'positive_reply', 'total_reply'],
        'transactions': ['total_topic', 'very_positive_topic', 'very_positive_reply'],
    }
}
label_headers = ['price'] # , 'transactions'

aode = classifiers.AODE()
svclassifier = SVC(kernel='rbf', degree=8, probability=True)
kf = KFold(n_splits=10, shuffle=True) # 90% for training, 10% for testing

results = {}
for cryptocurrency in cryptocurrencies:
    print('\nTesting for ' + cryptocurrency + ' data set')

    file_name = 'res/forecasting_results/' + cryptocurrency + '.txt'
    file_ = codecs.open(file_name, 'w+', 'utf-8')

    results[cryptocurrency] = {}

    # Collect data
    df = retriever.get_data(cryptocurrency)#.apply(stats.zscore)
    # Binerize labels (Done in get_data)
    # df = retriever.categorize_labels(df, label_headers)

    for label in label_headers:
        results[cryptocurrency][label] = {}

        # Extract features and labels
        data = df.drop(columns=label_headers).apply(timeseries.standardize_laggedly).dropna()
        data = np.array(data[data_headers[cryptocurrency][label]])
        
        # Print some status
        data_len = len(data)
        features_amount = data[0].size
        print('\tTotal data collected: ' + str(data_len))
        print('\tTotal of features per data: ' + str(features_amount))

        for estimation in estimations:
            aode = classifiers.AODE()
            
            print('\tPerforming 10-Fold estimating probabilities on \'' + estimation + '\' mode')

            file_.write(
                'Results for \'' + label + '\' ' + cryptocurrency +
                ' estimating probabilities on \'' + estimation + '\' mode:\n'
            )

            results[cryptocurrency][label][estimation] = {}

            for lag in range(1, 14):
                # TODO: Time lag is a window not a single element
                labels = np.array(df[label][10:].shift(-1))
                print('\n\n\t\tPredicting with lag =', str(lag))
                
                file_.write('\tLag = ' + str(lag))

                # lagged_data = data[lag:]
                # lagged_labels = labels[lag:]

                precisions = []
                recalls = []
                f1s = []
                accuracies = []
                # for train_indices, test_indices in _split(lagged_data):
                train_data, test_data = lagged_data[:697], lagged_data[697:]
                train_labels, test_labels = lagged_labels[:697], lagged_labels[697:]

                # Training
                aode.fit(train_data, train_labels, online=False)
                #svclassifier.fit(train_data, train_labels)

                # Test
                pred_labels = []
                for element in test_data:
                    pred_labels.append(aode.predict(element, estimation=estimation))
                #pred_labels = svclassifier.predict(test_data)  

                # print('\n\n\n\n\n\n')
                # print(set(train_labels), set(pred_labels))

                metrics = precision_recall_fscore_support(
                    test_labels, pred_labels, average='weighted'
                )

                print(
                    '\t\t\t',
                    classification_report(test_labels, pred_labels).replace('\n', '\n\t\t\t')
                )

                precision, recall, f1, _ = metrics
                accuracy = accuracy_score(test_labels, pred_labels)

                print('\t\t\tAccuracy:', accuracy)

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                accuracies.append(accuracy)
                #####

                results[cryptocurrency][label][estimation]['precision'] = precisions
                results[cryptocurrency][label][estimation]['recall'] = recalls
                results[cryptocurrency][label][estimation]['f1'] = f1s
                results[cryptocurrency][label][estimation]['accuracy'] = accuracies

                file_.write('\n\t\tPrecisions: ' + str(precisions))
                file_.write('\n\t\t\tMean: ' + str(np.mean(precisions)))

                file_.write('\n\t\tRecalls: ' + str(recalls))
                file_.write('\n\t\t\tMean: ' + str(np.mean(recalls)))

                file_.write('\n\t\tF1s: ' + str(f1s))
                file_.write('\n\t\t\tMean: ' + str(np.mean(f1s)))

                file_.write('\n\t\tAccuracies: ' + str(accuracies))
                file_.write('\n\t\t\tMean: ' + str(np.mean(accuracies)) + '\n\n')

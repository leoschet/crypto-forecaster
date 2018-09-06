import operator
from collections import defaultdict

import numpy as np


class AODE(object):
    def __init__(self, minfreq=1):
        self.minfreq = minfreq
        self.freq_counts = {}
        self.label_data_amount = {}
        self.total_data_amount = 0
        self.feature_values = defaultdict(list)

    def fit(self, data, labels, online=False):
        if not online:
            self.freq_counts = {}
            self.label_data_amount = {}
            self.total_data_amount = 0
            self.feature_values = defaultdict(list)

        unique_labels = set(labels)
        for label in unique_labels:
            label_freq_count = {}
            label_indices = np.where(labels == label)[0]

            # Save amount of data for each class
            self.label_data_amount[label] = label_indices.size
            self.total_data_amount += label_indices.size

            for index, feature in enumerate(data[label_indices].T):
                feature_value_indices = defaultdict(list)
                for value in feature:
                    # Count feature value's frequency
                    feature_value_indices[value].append(index)

                label_freq_count[index] = feature_value_indices

                # Save all possible values for every feature (no matter the class)
                self.feature_values[index] += list(feature_value_indices.keys())

            # Save label feature value's frequency
            self.freq_counts[label] = label_freq_count

    def predict(self, data, estimation='laplace', probabilities=False):
        """
        Given an input vector x, of size n, the classification goes as following:
        For each class and each value of x:
        - Calculate the probability of the current class and value
        - Calculate the productory of the probability of every value of x
        given the current class and value.
        
        The classification can be done by just finding the class that 
        maximize the product between the step's results.
        
        For providing the class probability estimates directly, the product of
        the two steps presented above must be normalized by the sum of the product
        across all classes.

        If 'probabilities' is set to true, an dictionary with the probability of all classes
        given the data is returned. The keys are the classes, the values are the probabilities
        """
        labels_probability = {}
        for label in self.freq_counts.keys():
            labels_probability[label] = 0
            for feature in self.freq_counts[label]:
                # Estimate the probability of the current class and value
                label_feature_prob = self._estimate_label_feature_probability(
                    label, feature, data[feature], estimation
                )

                productory = 1
                for independent_feature in self.freq_counts[label]:
                    # Estimate probability of every value of x given the current class and value
                    productory *= self._estimate_probability_density(
                        label, feature, data[feature], independent_feature,
                        data[independent_feature], estimation
                    )

                labels_probability[label] += label_feature_prob * productory

        if probabilities:
            prob_sum = sum([item[1] for item in labels_probability.items()])
            for label in labels_probability:
                labels_probability[label] /= prob_sum

            # Return the probability for every class given the data
            return labels_probability

        if estimation == 'count':
            # When using counter mode, the classifier obtains better results when
            # the class with lower probability is returned
            return max(labels_probability.items(), key=operator.itemgetter(1))[0]

        # Return the label with higher probability
        return min(labels_probability.items(), key=operator.itemgetter(1))[0]

    def _estimate_label_feature_probability(self, label, feature, value, estimation):
        """
        The estimation may be done in two different modes:
            - The one suggested in AODE's paper
            - Frequency count with laplace smoothing

        As suggested from "Not So Naive Bayes: Aggregating One-Dependence Estimators",
        AODE paper, the probability of a class and a feature value shall be estimated using
        the frequency of the class and the value as well as the count of all elements where
        the class and the feature are known (the same as all training data provided).

        For estimating the probability, it's suggested to use laplace smoothing with a tiny 
        modification. One is added to the frequency of the class and the value and divided by
        the count of all elements where the class and the feature are known added to the number of
        possible values for this feature.

        Alternatively, the probability of a class and a feature value can be estimated simply with 
        the add-one laplace smoothing
        """
        try:
            # Frequency of the class and the value
            value_freq = len(self.freq_counts[label][feature][value])
        except KeyError:
            value_freq = 0

        # Count of all elements where the class and the feature are known
        # label_feature_freq = self.label_data_amount[label]
        label_feature_freq = self.total_data_amount

        # Paper smoothing
        if estimation == 'paper':
            value_freq += 1
            label_feature_freq += len(self.feature_values[feature])

        # Laplace smoothing
        elif estimation == 'laplace' or estimation == 'count':
            value_freq += 1
            label_feature_freq += 1

        return value_freq/label_feature_freq

    def _estimate_probability_density(self, label, dependent_feature, dependent_value,
                                      feature, value, estimation):
        """
        The estimation may be done in two different modes:
            - The one suggested in AODE's paper
            - Frequency count with laplace smoothing

        As suggested from "Not So Naive Bayes: Aggregating One-Dependence Estimators",
        AODE paper, the probability of a class and a feature value shall be estimated using
        the frequency of the class, the dependent value and the value as well as the count
        of all elements where the class and the features are known (the same as all training
        data provided).

        For estimating the probability, it's suggested to use laplace smoothing with a tiny 
        modification. One is added to the frequency of the class and the value and divided by
        the count of all elements where the class and the feature are known added to the number of
        possible values for this feature.

        Alternatively, the probability of a class and a feature value can be estimated simply with 
        the add-one laplace smoothing
        """
        try:
            dependent_value_indices = self.freq_counts[label][dependent_feature][dependent_value]
        except KeyError:
            dependent_value_indices = []

        try:
            value_indices = self.freq_counts[label][feature][value]
        except KeyError:
            value_indices = []

        common_indices = [index for index in value_indices if index in dependent_value_indices]

        # Frequency of the class, the dependent value and the value
        values_freq = len(common_indices)

        # Count of all elements where the class and the feature are known
        # label_feature_freq = self.label_data_amount[label]
        label_feature_freq = self.total_data_amount
        
        # Paper smoothing
        if estimation == 'paper':
            values_freq += 1
            
            feature_possible_values = len(self.feature_values[feature])
            dependent_possible_values = len(self.feature_values[dependent_feature])
            label_feature_freq += feature_possible_values * dependent_possible_values

        # Laplace smoothing
        elif estimation == 'laplace':
            values_freq += 1
            label_feature_freq += 1

        # Frequency count with laplace smoothing
        elif estimation == 'count':
            values_freq += 1

            label_feature_freq = len(value_indices)
            label_feature_freq += 1

        return values_freq/label_feature_freq

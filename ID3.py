import numpy as np
import pandas
import math
from sklearn.model_selection import KFold
from matplotlib import pyplot


class Node:
    def __init__(self):
        self.Classification = 'B'
        self.threshold_val = -1
        self.featureIdx = -1
        self.is_leaf = False
        self.true_branch = None  # right son    row[node.idx] >= node.value
        self.false_branch = None  # left son


class ID3:

    def __init__(self, min_for_pruning=0):
        self.root = None
        self.min_for_pruning = min_for_pruning

    def calc_entropy(self, row):
        if len(row) == 0:
            return 0
        # find the counts of B and M
        B_count = np.count_nonzero(row == 'B')
        M_count = len(row) - B_count
        # calc the probity
        p_B = B_count / len(row)
        p_M = M_count / len(row)
        # calc the entropy
        if p_B == 0 or p_M == 0:
            return 0
        return -p_B * math.log(p_B, 2) - p_M * math.log(p_M, 2)

    def calc_info_gain(self, train, feature, thresholdVal, curr_IG):
        labels = train[:, 0]
        true_branch = train[train[:, feature] >= thresholdVal]
        false_branch = train[train[:, feature] < thresholdVal]
        IG_feature = (true_branch.shape[0] * self.calc_entropy(true_branch[:, 0])) / len(labels)
        IG_feature += (false_branch.shape[0] * self.calc_entropy(false_branch[:, 0])) / len(labels)
        return curr_IG - IG_feature

    def predict_sample(self, trainSet):
        B_counter = 0
        M_counter = 0
        for label in trainSet:
            if label == 'B':
                B_counter += 1
            else:
                M_counter += 1
        return 'B' if B_counter > M_counter else 'M'

    def is_leaf(self, trainSet):
        B_counter = 0
        M_counter = 0
        for label in trainSet:
            if label == 'B':
                B_counter += 1
            else:
                M_counter += 1
            if B_counter and M_counter:
                return False
        return True

    def findBestFeature(self, trainSet, node):
        best_gain = float("-inf")
        fatherInfoGain = self.calc_entropy(trainSet[:, 0])
        # loop over features
        for feature in range(1, len(trainSet[0])):
            # values = np.sort(np.unique(trainSet[:, feature]))
            feature_arr = np.unique(trainSet[:, feature])
            for value1, value2 in zip(feature_arr[:-1], feature_arr[1:]):
                value = (value1 + value2) / 2
                gain = self.calc_info_gain(trainSet, feature, value, fatherInfoGain)
                if gain > best_gain or (value > node.threshold_val and gain == best_gain):
                    best_gain = gain
                    node.threshold_val = value
                    node.featureIdx = feature
        return

    # the main function, build tree and prune
    def buildDT(self, trainSet, node):
        # create a Node
        if node is None:
            node = Node()

        if self.is_leaf(trainSet[:, 0]):
            node.is_leaf = True
            node.Classification = trainSet[0, 0]
            return node

        # find the best split
        self.findBestFeature(trainSet, node)

        # call recursive the function
        feature_true_branch = trainSet[:, node.featureIdx] >= node.threshold_val
        feature_false_branch = trainSet[:, node.featureIdx] < node.threshold_val

        if feature_true_branch.sum() < self.min_for_pruning or feature_false_branch.sum() < self.min_for_pruning:
            node.is_leaf = True
            node.Classification = self.predict_sample(trainSet[:, 0])
            return node

        node.true_branch = self.buildDT(trainSet[feature_true_branch], node.true_branch)
        node.false_branch = self.buildDT(trainSet[feature_false_branch], node.false_branch)

        return node

    def predict(self, row, node):
        if node.is_leaf:
            return node.Classification
        if row[node.featureIdx] >= node.threshold_val:
            return self.predict(row, node.true_branch)
        else:
            return self.predict(row, node.false_branch)

    def fit_predict(self, train, test):
        self.root = self.buildDT(train, self.root)
        y_pred = []
        for sample in test:
            if self.predict(sample, self.root) == 'M':
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred


def accuracy(y_prediction, test):
    counter = 0
    assert len(test[:, 0]) == len(y_prediction)
    for i in range(len(test[:, 0])):
        if test[i][0] == 'B' and y_prediction[i] == 0:
            counter += 1
        if test[i][0] == 'M' and y_prediction[i] == 1:
            counter += 1
    return counter / test[:, 0].size


def experiment(train):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=208136176)
    m_choices = [1, 5, 10, 15, 25]
    accuracies = []
    for m in m_choices:
        m_arr = []
        m_tree = ID3(m)
        for trainIdx, testIdx in k_fold.split(train):
            y_prediction = m_tree.fit_predict(train[trainIdx], train[testIdx])
            m_acc = accuracy(y_prediction, train[testIdx])
            m_arr.append(m_acc)
        accuracies.append(np.mean(m_arr))

    max_accuracy = np.max(accuracies)
    print("Best prune value is: ", end=" ")
    print(m_choices[accuracies.index(max_accuracy)], end=" ")
    print("with accuracy =", end=" ")
    print(max_accuracy)
    pyplot.plot(m_choices, accuracies)
    pyplot.title("pruning effect")
    pyplot.xlabel("m_choice")
    pyplot.ylabel("accuracy")
    pyplot.show()
    return


def experiment_loss_first_section(train):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=208136176)
    m_choices = [10]
    accuracies = []
    for m in m_choices:
        m_arr = []
        m_tree = ID3(m)
        for trainIdx, testIdx in k_fold.split(train):
            y_prediction = m_tree.fit_predict(train[trainIdx], train[testIdx])
            m_acc = accuracy_loss_first_section(y_prediction, train[testIdx])
            m_arr.append(m_acc)
        accuracies.append(np.mean(m_arr))
    print(accuracies)
    return


def accuracy_loss_first_section(y_prediction, test):
    FP = 0
    FN = 0
    assert len(test[:, 0]) == len(y_prediction)
    for i in range(len(test[:, 0])):
        if test[i][0] == 'B' and y_prediction[i] == 1:
            FP += 1
        if test[i][0] == 'M' and y_prediction[i] == 0:
            FN += 1
    return FP + 8 * FN


def experiment_loss_second_section(train):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=208136176)
    m_choices = [10]
    accuracies = []
    for m in m_choices:
        m_arr = []
        m_tree = ID3(m)
        for trainIdx, testIdx in k_fold.split(train):
            y_prediction = m_tree.fit_predict(train[trainIdx], train[testIdx])
            m_acc = accuracy_loss_second_section(y_prediction, train[testIdx])
            m_arr.append(m_acc)
        accuracies.append(np.mean(m_arr))
    print(accuracies)
    return


def accuracy_loss_second_section(y_prediction, test):
    FP = 0
    assert len(test[:, 0]) == len(y_prediction)
    for i in range(len(test[:, 0])):
        if test[i][0] == 'B':
            FP += 1
    return FP


if __name__ == '__main__':
    train = (pandas.read_csv('train.csv')).to_numpy()
    # test = (pandas.read_csv('test.csv')).to_numpy()
    # tree = ID3()
    # tree.fit_predict(train, test)

    # experiment
    experiment(train)

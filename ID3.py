import numpy as np
import pandas
import math
from sklearn.model_selection import KFold
from matplotlib import pyplot
# define consts
ID = 208136176


class Node:
    def __init__(self):
        self.Classification = 'B'
        self.threshold_val = -1
        self.featureIdx = -1
        self.is_leaf = False
        self.true_branch = None  # right son    row[node.idx] >= node.value
        self.false_branch = None  # left son


class ID3:

    def __init__(self, M=2):
        self.root = None
        self.M = M  # The prune value

    # ==== given a row return the entropy of it ====
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

    # ==== given a row return the prediction of a leaf based in its neighbors ====
    def predict_sample(self, row):
        B_counter = np.count_nonzero(row == 'B')
        M_counter = np.count_nonzero(row == 'M')
        return 'B' if B_counter > M_counter else 'M'

    def is_leaf(self, row):
        B_counter = np.count_nonzero(row == 'B')
        M_counter = np.count_nonzero(row == 'M')
        if B_counter and M_counter:
            return False
        return True

    def calc_info_gain(self, train, feature, thresholdVal, curr_IG):
        labels = train[:, 0]
        true_branch = train[train[:, feature] >= thresholdVal]
        false_branch = train[train[:, feature] < thresholdVal]
        IG_feature = (true_branch.shape[0] * self.calc_entropy(true_branch[:, 0])) / len(labels)
        IG_feature += (false_branch.shape[0] * self.calc_entropy(false_branch[:, 0])) / len(labels)
        return curr_IG - IG_feature

    def find_best_feature_to_split(self, trainSet, node):
        best_gain = 0
        parentInfoGain = self.calc_entropy(trainSet[:, 0])
        # loop over features
        for feature in range(1, len(trainSet[0])):
            values = np.sort(np.unique(trainSet[:, feature]))
            for i in range(len(values) - 1):
                value = (values[i] + values[i + 1]) / 2
                gain = self.calc_info_gain(trainSet, feature, value, parentInfoGain)
                if gain >= best_gain:
                    best_gain = gain
                    node.threshold_val = value
                    node.featureIdx = feature
        return best_gain

    # ==== the main function, build tree and prune ====
    def build_tree(self, trainSet, node):
        # create a Node
        if node is None:
            node = Node()

        # prune
        if len(trainSet[:, 0]) < self.M or self.is_leaf(trainSet[:, 0]):
            node.is_leaf = True
            node.Classification = self.predict_sample(trainSet[:, 0])
            return node

        # find the best split
        IG = self.find_best_feature_to_split(trainSet, node)

        # we didn't improve the branch, make it a leaf
        if IG == 0:
            node.is_leaf = True
            node.Classification = self.predict_sample(trainSet[:, 0])
            return node

        # call recursive the function
        node.true_branch = self.build_tree(trainSet[trainSet[:, node.featureIdx] >= node.threshold_val],
                                           node.true_branch)
        node.false_branch = self.build_tree(trainSet[trainSet[:, node.featureIdx] < node.threshold_val],
                                            node.false_branch)
        return node

    # ==== given a test sample (row) and the root of the DT return the prediction ====
    def predict(self, row, root):
        if root.is_leaf:
            return root.Classification
        if row[root.featureIdx - 1] >= root.threshold_val:
            return self.predict(row, root.true_branch)
        else:
            return self.predict(row, root.false_branch)

    # ==== build a tree and learn it (fit) according to the train, then test it with the test set
    # return list with the predictions ====
    def fit_predict(self, train, test):
        self.root = self.build_tree(train, self.root)
        y_prediction = []
        for sample in test:
            if self.predict(sample, self.root) == 'M':
                y_prediction.append(1)
            else:
                y_prediction.append(0)
        return y_prediction


"""
========================================================================
                              Experiments 
========================================================================
"""
# This function calculate the prune effect
# arguments: train - the train set we use to train the model
# to print remove the quotes

# calculate the accuracy of the tree we build
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
    k_fold = KFold(n_splits=5, shuffle=True, random_state=ID)
    # the result
    m_choices = [2, 8, 15, 23, 30]
    accuracies = []
    for m in m_choices:
        m_accuracy = []
        m_tree = ID3(m)
        for trainIdx, testIdx in k_fold.split(train):
            y_prediction = m_tree.fit_predict(train[trainIdx], train[testIdx])
            m_acc = accuracy(y_prediction, train[testIdx])
            m_accuracy.append(m_acc)
        accuracies.append(np.mean(m_accuracy))

    """
    max_accuracy = np.max(accuracies)
    print("Best prune value is:", end=" ")
    print(m_choices[accuracies.index(max_accuracy)], end=" ")
    print("with test accuracy =", end=" ")
    print(max_accuracy)
    pyplot.plot(m_choices, accuracies)
    pyplot.title("pruning effect")
    pyplot.xlabel("m_choice")
    pyplot.ylabel("accuracy")
    pyplot.show()
    """
    return


# ========================================================================
# This function calculate the loss for question 4 section 1
# arguments: train - the train set we use to train the model
# to print remove the quotes
def experiment_loss_first_section(train):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=ID)
    m_choices = [0]
    accuracies = []
    for m in m_choices:
        m_accuracy = []
        m_tree = ID3(m)
        for trainIdx, testIdx in k_fold.split(train):
            y_prediction = m_tree.fit_predict(train[trainIdx], train[testIdx])
            m_acc = accuracy_loss_first_section(y_prediction, train[testIdx])
            m_accuracy.append(m_acc)
        accuracies.append(np.mean(m_accuracy))
    """
    print(accuracies)
    """
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


# ========================================================================
# This function calculate the loss for question 4 section 2
# arguments: train - the train set we use to train the model
# to print remove the quotes
def experiment_loss_second_section(train):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=ID)
    m_choices = [8]
    accuracies = []
    for m in m_choices:
        m_accuracy = []
        m_tree = ID3(m)
        for trainIdx, testIdx in k_fold.split(train):
            y_prediction = m_tree.fit_predict(train[trainIdx], train[testIdx])
            m_acc = accuracy_loss_second_section(y_prediction, train[testIdx])
            m_accuracy.append(m_acc)
        accuracies.append(np.mean(m_accuracy))
    """
    print(accuracies)
    """
    return


def accuracy_loss_second_section(y_prediction, test):
    FP = 0
    assert len(test[:, 0]) == len(y_prediction)
    for i in range(len(test[:, 0])):
        if test[i][0] == 'B':
            FP += 1
    return FP

# -------------- #
# confmatrix.py  #
# Faerlin Pulido #
# 2020           #
# -------------- #
#
# This is a utility function used in jumio_challenge.ipynb
#

import numpy as np

def get_true_positive(class_index, confusion_matrix):
    return confusion_matrix[class_index][class_index]

def get_true_negative(class_index, confusion_matrix):
    N = len(confusion_matrix)
    M = len(confusion_matrix[0])
    top_left = confusion_matrix[:class_index, :class_index]

    top_right = []
    if (class_index < M-1):
        top_right = confusion_matrix[:class_index, (class_index+1):]

    bottom_left = []
    if (class_index < N-1):
        bottom_left = confusion_matrix[(class_index+1):, :class_index]

    bottom_right = []
    if (class_index < N-1 and class_index < M-1):
        bottom_right = confusion_matrix[(class_index+1):, (class_index+1):]

    true_positive = np.sum(top_left) + np.sum(top_right) + \
        np.sum(bottom_left) + np.sum(bottom_right)
    return true_positive

def get_false_positive(class_index, confusion_matrix):

    N = len(confusion_matrix)
    top = confusion_matrix[:class_index, class_index]
    bottom = []
    if (class_index < N-1):
        bottom = confusion_matrix[(class_index+1):, class_index]

    false_positive = np.sum(top) + np.sum(bottom)
    return false_positive

def get_false_negative(class_index, confusion_matrix):
    M = len(confusion_matrix[0])
    left = confusion_matrix[class_index, :class_index]
    right = []
    if (class_index < M-1):
        right = confusion_matrix[class_index, (class_index+1):]

    false_negative = np.sum(left) + np.sum(right)
    return false_negative

def get_true_positive_rate(class_index, confusion_matrix):
    tp = get_true_positive(class_index, confusion_matrix)
    fn = get_false_negative(class_index, confusion_matrix)
    tpr = tp/(tp+fn)
    return tpr

def get_false_positive_rate(class_index, confusion_matrix):
    tn = get_true_negative(class_index, confusion_matrix)
    fp = get_false_positive(class_index, confusion_matrix)
    fpr = fp/(fp+tn)
    return fpr
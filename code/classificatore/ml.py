import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import sklearn
import os
import logging

from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from sklearn.metrics import roc_curve, det_curve, auc


def get_apcer_op(apcer, bpcer, threshold, op):
    """Returns the value of the given FMR operating point
    Definition:
    ZeroFMR: is defined as the lowest FNMR at which no false matches occur.
    Others FMR operating points are defined in a similar way.
    @param apcer: =False Match Rates
    @type apcer: ndarray
    @param bpcer: =False Non-Match Rates
    @type bpcer: ndarray
    @param op: Operating point
    @type op: float
    @returns: Index, The lowest bpcer at which the probability of apcer == op
    @rtype: float
    """
    index = np.argmin(abs(apcer - op))
    return index, bpcer[index], threshold[index]


def get_bpcer_op(apcer, bpcer, threshold, op):
    """Returns the value of the given FNMR operating point
    Definition:
    ZeroFNMR: is defined as the lowest FMR at which no non-false matches occur.
    Others FNMR operating points are defined in a similar way.
    @param apcer: =False Match Rates
    @type apcer: ndarray
    @param bpcer: =False Non-Match Rates
    @type bpcer: ndarray
    @param op: Operating point
    @type op: float
    @returns: Index, The lowest apcer at which the probability of bpcer == op
    @rtype: float
    """
    temp = abs(bpcer - op)
    min_val = np.min(temp)
    index = np.where(temp == min_val)[0][-1]

    return index, apcer[index], threshold[index]


def get_eer_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1 = tpr + fpr - 1.0
    index = np.nanargmin(np.abs(differ_tpr_fpr_1))
    eer = fpr[index]

    return eer, index, threshold[index]


def performances_compute(prediction_scores, gt_labels, threshold_type, op_val, verbose):
    # fpr = apcer, 1-tpr = bpcer
    # op_val: 0 - 1
    # gt_labels: list of ints,  0 for attack, 1 for bonafide
    # prediction_scores: list of floats, higher value should be bonafide
    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    fpr, tpr, threshold = roc_curve(gt_labels, prediction_scores, pos_label=1, drop_intermediate=False)
    bpcer = 1 - tpr
    val_eer, _, eer_threshold = get_eer_threhold(fpr, tpr, threshold)
    val_auc = auc(fpr, tpr)

    if threshold_type == 'eer':
        threshold = eer_threshold
    elif threshold_type == 'apcer':
        _, _, threshold = get_apcer_op(fpr, bpcer, threshold, op_val)
    elif threshold_type == 'bpcer':
        _, _, threshold = get_bpcer_op(fpr, bpcer, threshold, op_val)
    else:
        threshold = 0.5

    num_real = len([s for s in data if s['label'] == 1])
    num_fake = len([s for s in data if s['label'] == 0])

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    threshold_APCER = type2 / num_fake
    threshold_BPCER = type1 / num_real
    threshold_ACER = (threshold_APCER + threshold_BPCER) / 2.0

    if verbose is True:
        logging.info(
            f'AUC@ROC: {val_auc}, threshold:{threshold}, EER: {val_eer}, APCER:{threshold_APCER}, BPCER:{threshold_BPCER}, ACER:{threshold_ACER}')
        print(
            f'AUC@ROC: {val_auc}, threshold:{threshold}, EER: {val_eer}, APCER:{threshold_APCER}, BPCER:{threshold_BPCER}, ACER:{threshold_ACER}')

    return val_auc, val_eer, [threshold, threshold_APCER, threshold_BPCER, threshold_ACER]


def compute_eer(label, pred):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, drop_intermediate=False)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


train = pd.read_csv('../output/feature_extraction/model_webmorph/train/train.csv')

x = train.drop(['image_path', "label"], axis=1)
y = []
for elem in train['label']:
    if elem == 'bonafide':
        y.append(1)
    elif elem == 'attack':
        y.append(0)

mappa_valori = {0: 1, 1: 0}  # Sostituzione dei valori
y = [mappa_valori[val] for val in y]
# scaler = MinMaxScaler()
# scaler.fit(x)
# train_scaled = scaler.transform(x)

sel = VarianceThreshold(threshold=0.017)

X_train_sel = sel.fit_transform(x)

# select the GaussianNB algorithm
# model = GaussianNB()

# select the RabdomForest algorithm
# model = RandomForestClassifier()

# select the DecisionTree algorithm
model = DecisionTreeClassifier()
model.fit(X_train_sel, y)

directory = '../output/feature_extraction/model_webmorph/test'
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        test = pd.read_csv(directory + '/' + filename)

        x_test = test.drop(['image_path', "label"], axis=1)
        y_test = []
        for elem in test["label"]:
            if elem == 'bonafide':
                y_test.append(1)
            elif elem == 'attack':
                y_test.append(0)

        mappa_valori = {0: 1, 1: 0}  # Sostituzione dei valori
        y_test = [mappa_valori[val] for val in y_test]

        # test_scaled  = scaler.transform(x_test)
        X_test_sel = sel.transform(x_test)
        prediction = model.predict(X_test_sel)
        # prediction = grid_search.predict(x_test)

        logging.basicConfig(filename='../output/classificator_DecisionTree/webmorph/info_webmorph.log', level=logging.INFO)
        logging.info(directory.split('/')[3] + '/' + filename)
        logging.info('The accuracy is: {:.2%}'.format(accuracy_score(prediction, y_test)))
        print(directory.split('/')[3] + '/' + filename)
        print('The accuracy is: ', accuracy_score(prediction, y_test))

        logging.info("Confusion Matrix:")
        logging.info(confusion_matrix(y_test, prediction))
        print("\nConfusion Matrix: ")
        print(confusion_matrix(y_test, prediction))

        logging.info("Classifiction Report: ")
        logging.info(classification_report(y_test, prediction))
        print("\nClassifiction Report: ")
        print(classification_report(y_test, prediction))

        logging.info("EER: {:.2%}".format(compute_eer(y_test, prediction)))
        print("EER: ", compute_eer(y_test, prediction))
        z = confusion_matrix(y_test, prediction)

        apcer = z[0][1] / (z[0][0] + z[0][1])
        bpcer = z[1][0] / (z[1][0] + z[1][1])
        eer = (apcer + bpcer) / 2

        logging.info("APCER: {:.2%}".format(apcer))
        logging.info("BPCER: {:.2%}".format(bpcer))
        logging.info("EER FAKE: {:.2%}".format(eer))
        print("APCER: ", '{:.2%}'.format(apcer))
        print("BPCER: ", '{:.2%}'.format(bpcer))
        print("EER FAKE: ", '{:.2%}'.format(eer))

        # Calcolare l'APCER e il BPCER per ogni punto di lavoro
        decision_scores = model.predict_proba(X_test_sel)[:, 1]
        # y_pred_morphed = decision_scores[:, 0]
        performances_compute(decision_scores, y_test, "apcer", 0.2, True)

        logging.info('\n\n')
        print('\n\n')

"""tuned_parametersSGD= [{'loss': ['hinge'], 'penalty': ['l1','l2'], 'alpha':[0.0005, 0.001, 0.002, 0.003,0.0001], 'tol':[ 1e-4, 1e-3, 1e-2],'n_jobs':[-1], 'max_iter': [1000000000],   'random_state':[42]},
                     {'loss': ['log'], 'penalty': ['l1','l2'], 'alpha': [0.0005, 0.001, 0.002, 0.003, 0.0001], 'tol':[  1e-4,1e-3, 1e-2], 'n_jobs': [-1], 'max_iter': [1000000000], 'random_state':[42]},
                     {'loss': ['modified_huber'], 'penalty': ['l1','l2'], 'alpha': [0.0005, 0.001, 0.002, 0.003, 0.0001], 'tol':[ 1e-4, 1e-3, 1e-2], 'n_jobs': [-1],'max_iter': [1000000000],'random_state':[42]},
                     {'loss': ['perceptron'], 'penalty': ['l1','l2'], 'alpha': [0.0005, 0.001, 0.002, 0.003, 0.0001], 'tol':[ 1e-4, 1e-3, 1e-2], 'n_jobs': [-1], 'max_iter': [1000000000], 'random_state':[42]},
                     ]

grid_search = GridSearchCV(SGDClassifier(), tuned_parametersSGD)
grid_search.fit(x, y)
print('')
print(grid_search.best_score_)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.refit_time_)
"""

"""def calculate_bpcer_apcer(genuine_scores, impostor_scores, threshold, apcer_target):
    # Calculate the BPCER and APCER
    genuine_accept = (genuine_scores >= threshold).sum()
    genuine_reject = (genuine_scores < threshold).sum()
    bpcer = 1 - (genuine_accept / float(len(genuine_scores)))
    
    impostor_accept = (impostor_scores < threshold).sum()
    impostor_reject = (impostor_scores >= threshold).sum()
    apcer = impostor_accept / float(len(impostor_scores))
    
    # Calculate the BPCER@APCER
    bpcers = np.linspace(0, 1, 1000)
    apcers = np.ones_like(bpcers) * apcer_target
    bpcer_apcer = np.mean((bpcers + apcers) / 2)
    
    return bpcer, apcer, bpcer_apcer
"""

# train = train.reset_index(drop=True)
# test = test.reset_index(drop=True)


# print(len(y[y==1]))
# print(len(y[y==0]))

"""
classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="rbf"),
    SVC(),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
"""

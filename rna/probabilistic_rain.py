print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD Style.

import numpy as np
import matplotlib.pyplot as plt

from utils.custom_losses import *
from utils.utils import save_model, load_model, loadmat, savemat
from utils.utils import binarize, StandardScaler, train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn_models import models, GaussianNB, rng
from sklearn_models import LinearSVC, SVC, LogisticRegression, MLPClassifier

# Create the dataset
path = 'data/train_data/'
ref_var = ['rain', 'mslp', 't2m', 'rh2m']

var_key = 'rain'
init_time = '00'
domain = 'd01'

ref_var.remove(var_key)
file_name = '_'.join([var_key] + ref_var + [domain, init_time]) + '.mat'

print('Loading data...')
data = loadmat(path + file_name, squeeze_me=True)

#
for key in ['__header__', '__version__', '__globals__']:
    _ = data.pop(key)

# values = data['stn_356']

# join all data
dataset = []
for stn, values in data.iteritems():
    indexs = values[:, 1] > 0.1
    print(stn, 1e2 * len(values[indexs]) / len(values))
    dataset.append(values)
if len(dataset) > 0:
    values = np.concatenate(dataset, axis=0)
else:
    exit()

scaler = StandardScaler()

# split data for validation and testing
X, y = scaler.fit_transform(values[:, 2:]), binarize(
    values[:, 1:2], threshold=0.1).squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=True)

# fitting model
print('----------------------------------------------------------------')
print('inputs: integer tensor of shape (samples, predictors)')
print('inputs_train shape:', X_train.shape)
print('inputs_test shape:', X_test.shape)
print('----------------------------------------------------------------')
print('outputs: float, tensor of shape (samples, 1)')
print('outputs_train shape:', y_train.shape)
print('outputs_test shape:', y_test.shape)
print('----------------------------------------------------------------')


def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10), dpi=300)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ref_brier_score = 1.0

    clfs = [lr, est, isotonic, sigmoid]
    names = ['Logistic', name, name + ' Isotonic', name + ' Sigmoid']
    for clf, name in zip(clfs, names):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=20)

        ax1.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(
            prob_pos, range=(0, 1), bins=20, label=name, histtype="step", lw=2)

        if clf_score < ref_brier_score:
            clf_name = name
            ref_brier_score = clf_score
            best_classifier = clf

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()

    # saving best classifier:
    save_model(best_classifier,
               'models/best_' + '_'.join(clf_name.split(' ')) + '.clf')

    return fig


if __name__ == '__main__':

    # Plot calibration curve for Gaussian Naive Bayes:
    models = [("Naive Bayes", GaussianNB()), ("MLPC", models['mlp_c']),
              ("LSVC", LinearSVC()), ("SVC", SVC(kernel="rbf"))]

    for i, (name, model) in enumerate(models):

        fig_name = '_'.join( name.lower().split(' ') ) + '.png'
        fig = plot_calibration_curve(model, name, i)
        fig.savefig('outputs/calibration/' + fig_name)

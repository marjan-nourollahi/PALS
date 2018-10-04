__author__ = 'M_Nour'

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score, recall_score
import dataset
from collections import Counter
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from imblearn.under_sampling import NearMiss,CondensedNearestNeighbour
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



# sensor_data = dataset.load_data()
# rng = np.random.RandomState(0)
# indices = np.arange(len(sensor_data.data))
# rng.shuffle(indices)
#
# sm = SMOTE(random_state=42)
# X, y = sm.fit_sample(sensor_data.data[indices[:6000]], sensor_data.target[indices[:6000]])
#
# cnn = CondensedNearestNeighbour(random_state=42)
# X_test, y_test = cnn.fit_sample(sensor_data.data[indices[6000:10000]], sensor_data.target[indices[6000:10000]])
# print(Counter(y))


def explore(X, y, file_name, mdl , d=None):
    n_total_samples = len(y)
    n_labeled_points = 300
    max_iterations = 1

    unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
    f_result =[]
    acc_result = []
    recall_result = []

    for i in range(max_iterations):
        if len(unlabeled_indices) == 0:
            print("No unlabeled items left to label.")
            break
        h = np.setdiff1d(np.arange(n_total_samples),unlabeled_indices, assume_unique=True)
        y_train = y[h]
        X_train = X[h]


        # model = SVC(gamma=2, C=1)
        # model = linear_model.LogisticRegression()
        # model = QuadraticDiscriminantAnalysis()#KNeighborsClassifier(3)
        # model  = AdaBoostClassifier()
        # model = RandomForestClassifier( n_estimators=185)
        if d!=None:
            model = mdl(**d)
        else:
            model = mdl()
        model.fit(X_train, y_train)

        # predicted_labels = lp_model.transduction_[unlabeled_indices]

        #      SVC(kernel="linear", C=0.025),
    #     SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # # DecisionTreeClassifier(max_depth=5),
    # # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # # AdaBoostClassifier(),
    # # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
        predicted_labels = model.predict(X[unlabeled_indices])
        true_labels = y[unlabeled_indices]

        cm = confusion_matrix(true_labels, predicted_labels,
                              labels=model.classes_)

        if i%1 ==0:
            print("Iteration %i %s" % (i, 70 * "_"))

            print(classification_report(true_labels, predicted_labels))
            print(accuracy_score(true_labels, predicted_labels))
            print(f1_score(true_labels, predicted_labels, average='binary'))
            f_result.append(f1_score(true_labels, predicted_labels, average='binary'))
            acc_result.append(accuracy_score(true_labels, predicted_labels))
            recall_result.append(recall_score(true_labels, predicted_labels, average='macro'))


            print("Confusion matrix")
            print(cm)
        from scipy import stats
        # compute the entropies of transduced label distributions

        # model.predict(X)
        p = model.predict_proba(X)
        e = stats.distributions.entropy(p.T)
        # pred_entropies = stats.distributions.entropy(
        #     model.p)
        # uncertainty_index = np.argsort(pred_entropies)[::-1]

        # pred_decision_function = np.abs([x for x in model.decision_function(X)])


        # select up to 5 digit examples that the classifier is most uncertain about
        # uncertainty_index = np.argsort(model.decision_function(X))[::-1]
        uncertainty_index = np.argsort(e)[::-1]

        # print(unlabeled_indices.shape)
        # print(uncertainty_index[:10])
        #
        uncertainty_index = uncertainty_index[
                                np.in1d(uncertainty_index, unlabeled_indices)][:40]


        # keep track of indices that we get labels for
        delete_indices = np.array([])
        for index, image_index in enumerate(uncertainty_index):
            # labeling 5 points, remote from labeled set
            delete_index, = np.where(unlabeled_indices == image_index)
            delete_indices = np.concatenate((delete_indices, delete_index))
        unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
        n_labeled_points += len(uncertainty_index)

    ###################################AUC-ROC##############################################
    # Compute ROC curve and ROC area for each class
    # p = model.predict_proba(X[unlabeled_indices])
    # true_label = y[unlabeled_indices]  # [unlabeled_indices > 4000]
    #
    # fpr, tpr, xxx = roc_curve(true_label, p[:, 1])
    # roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area

    ##############################################################################
    # Plot of a ROC curve for a specific class
    # fig = plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC')
    # plt.legend(loc="lower right")
    # ax2 = plt.gca().twinx()
    # ax2.plot(fpr, xxx, markeredgecolor='r', linestyle='dashed', color='r')
    # ax2.set_ylabel('Threshold', color='r')
    # ax2.set_ylim([xxx[-1], xxx[0]])
    # ax2.set_xlim([fpr[0], fpr[-1]])
    #
    # plt.show()
    # #############################################################################################
    # p = model.predict_proba(X[unlabeled_indices])
    # true_label = y[unlabeled_indices]  # [unlabeled_indices > 4000]
    # predicted_labels = model.predict(X[unlabeled_indices])
    # print('+' * 70)
    # print(classification_report(true_label, predicted_labels))
    # predicted_labels2 = [1 if x > 0.1 else 0 for x in p[:, 1]]
    # cm = confusion_matrix(true_label, predicted_labels,
    #                       labels=model.classes_)
    # print(cm)
    #
    # print('+' * 70)
    # print(classification_report(true_label, predicted_labels2))
    # cm = confusion_matrix(true_label, predicted_labels2,
    #                       labels=model.classes_)
    # print(cm)
    # print('+' * 70)
    # print(sum([1 if x > 1.00  else 0 for x in xxx]))

    # with open("f1_score_" + file_name+ ".csv", 'wb') as f_file:
    #     np.savetxt(f_file, f_result, delimiter=",", fmt='%10.5f')
    #
    # with open("accuracy_score_" + file_name + ".csv", 'wb') as acc_file:
    #     np.savetxt(acc_file, acc_result, delimiter=",", fmt='%10.5f')
    #
    #
    # with open("recall_score_" + file_name + ".csv", 'wb') as recall_file:
    #     np.savetxt(recall_file, recall_result, delimiter=",", fmt='%10.5f')
    #
    # s= file_name +'     :       ' + f1_score(true_labels, predicted_labels,average='macro')
    # with open("f_final_score.csv", 'wb') as f_final_file:
    #     np.savetxt(f_final_file, s, delimiter=",", fmt='%10.5f')
    #


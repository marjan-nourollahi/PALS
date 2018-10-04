__author__ = 'M_Nour'

import numpy as np
from scipy import stats
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score, recall_score
from collections import Counter
# import marjan_dataset
import dataset
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc




def load_data():
    sensor_data = dataset.load_data()
    X1, y1 = sensor_data.data[:6000], sensor_data.target[:6000]
    print(Counter(y1))


    rng = np.random.RandomState(0)
    indices = np.arange(len(X1))
    print(len(sensor_data.data[:]))
    print('*'*20)
    print(len(indices))
    rng.shuffle(indices)

    X,y = X1[indices], y1[indices]
    return X, y


# def PAL(method, file_name='PAL',use_default_lp_setting = True, d=None):
def PAL(X, y, method, file_name='PAL',use_default_lp_setting = True, d=None):

    n_total_samples = len(y)#+ len(y_test)
    print('#' * 70)
    print(len(y))
    print('#' * 70)
    n_labeled_points = 200
    max_iterations =50
    f_result =[]
    acc_result = []
    recall_result = []

    # all_y = np.concatenate((y, y_test), axis=0)
    # all_X = np.concatenate((X, X_test), axis=0)
    #
    # print(Counter(y_test))

    unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]

    for i in range(max_iterations):
        if len(unlabeled_indices) == 0:
            print("No unlabeled items left to label.")
            break

        sm = ADASYN(random_state=42)
        mask = np.setdiff1d(np.arange(len(X)), unlabeled_indices)
        X_res, y_res = sm.fit_sample(X[mask],y[mask])
        y_unlabeled = np.copy(y[unlabeled_indices])
        y_unlabeled[:] =-1
        X_unlabeled = X[unlabeled_indices]
        # y_train = np.copy(y[unlabeled_indices])#all_y)
        # y_train[:] = -1
        # y_train_2 = np.copy(y_test)
        # y_train_2[:] = -1
        # y_unlabeled = np.concatenate((y_train, y_train_2), axis=0)

        # h = np.setdiff1d(np.arange(n_total_samples), unlabeled_indices, assume_unique=True)

        if use_default_lp_setting:
            lp_model = label_propagation.LabelSpreading(kernel='knn', n_neighbors=10)
        else:
            lp_model = label_propagation.LabelSpreading(**d)


        y_f , X_f = np.concatenate((y_res, y_unlabeled), axis=0), np.concatenate((X_res, X_unlabeled), axis=0)


        lp_model.fit(X_f, y_f)
        ########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # model = QuadraticDiscriminantAnalysis()
        # model.fit(X[mask], y[mask])
        # QD_predicted = model.predict(X[unlabeled_indices])

        ########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4444

        ind = np.arange(len(X_f))
        # predicted_labels = lp_model.transduction_[ind[ind >= len(X_res)]]
        lp_predicted = lp_model.predict(X[unlabeled_indices])

        true_labels = y[unlabeled_indices] #[unlabeled_indices > 4000]

        predicted_labels = lp_predicted

        cm = confusion_matrix(true_labels, predicted_labels,
                              labels=lp_model.classes_)

        print("Iteration %i %s" % (i, 70 * "_"))
        print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
              % (n_labeled_points, n_total_samples - n_labeled_points,
                 n_total_samples))

        print(classification_report(true_labels, predicted_labels))
        print(accuracy_score(true_labels, predicted_labels))
        print("Confusion matrix")
        print(cm)
        ####################################  End Prints   ##############################################

        f_result.append(f1_score(true_labels, predicted_labels,average='binary'))
        acc_result.append(accuracy_score(true_labels, predicted_labels))
        recall_result.append(recall_score(true_labels, predicted_labels,average='macro'))

        uncertainty_index = method(lp_model,X, unlabeled_indices)

        # keep track of indices that we get labels for
        delete_indices = np.array([])
        eating_counter = 0
        non_eating_counter = 0

        for index in uncertainty_index:

            delete_index, = np.where(unlabeled_indices == index)
            delete_indices = np.concatenate((delete_indices, delete_index))
        unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
        n_labeled_points += len(delete_indices)

    ###################################AUC-ROC##############################################
    # Compute ROC curve and ROC area for each class
    # p = lp_model.predict_proba(X[unlabeled_indices])
    # from collections import Counter
    # print(max(p[:, 1]))
    # print(min(p[:, 1]))
    # prob= [round(x, 4) for x in p[:,1]]
    # print(max(prob))
    # print(min(prob))
    # not_nan_index = []
    #
    # for i, item in enumerate(prob):
    #     if np.isfinite(item):
    #         not_nan_index.append(i)
    #
    # true_label = y[unlabeled_indices]  # [unlabeled_indices > 4000]
    # true_label = true_label[not_nan_index]
    # prob = p[:,1][not_nan_index]
    #
    #
    # fpr, tpr, xxx = roc_curve(true_label, prob)
    # fpr = fpr[1:]
    # tpr = tpr[1:]
    # xxx = xxx[1:]
    #
    # roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area

    ##############################################################################
    # # Plot of a ROC curve for a specific class
    # fig =plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.plot([0, 1], [0.5, 0.5], markeredgecolor='b', linestyle='dashed', color='b')
    # plt.plot([0, 1], [0.52, 0.52], color='navy', lw=lw, linestyle='--')
    # plt.plot([0.5, 0.5], [0, 1], color='navy', lw=lw, linestyle='--')

    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC')
    # plt.legend(loc="lower right")
    # ax2 = plt.gca().twinx()
    # ax2.plot(fpr, xxx, markeredgecolor='r', linestyle='dashed', color='r')
    # ax2.set_ylabel('Threshold', color='r')
    # # ax2.set_ylim([xxx[-1], xxx[0]])
    # ax2.set_ylim([0.0, 1.05])
    # ax2.set_xlim([fpr[0], fpr[-1]])
    #
    # plt.show()
    #

    #############################################################################################
    # p = lp_model.predict_proba(X[unlabeled_indices])
    # true_label = y[unlabeled_indices]  # [unlabeled_indices > 4000]
    # predicted_labels=lp_model.predict(X[unlabeled_indices])
    # print('+'*70)
    # print(classification_report(true_label, predicted_labels))
    # predicted_labels2 = [1 if x>0.57 else 0 for x in p[:, 1]]
    # cm = confusion_matrix(true_label, predicted_labels,
    #                       labels=lp_model.classes_)
    # print(cm)
    #
    # print('+' * 70)
    # print(classification_report(true_label, predicted_labels2))
    # cm = confusion_matrix(true_label, predicted_labels2,
    #                       labels=lp_model.classes_)
    # print(cm)
    # print('+' * 70)
    # print(sum([1 if x>1.00  else 0 for x in xxx]))
    # ts_method(lp_model)
    ####################################   Prints   #################################################


    with open("f1_score_" + file_name+ ".csv", 'wb') as f_file:
        np.savetxt(f_file, f_result, delimiter=",", fmt='%10.5f')

    with open("accuracy_score_" + file_name + ".csv", 'wb') as acc_file:
        np.savetxt(acc_file, acc_result, delimiter=",", fmt='%10.5f')


    with open("recall_score_" + file_name + ".csv", 'wb') as recall_file:
        np.savetxt(recall_file, recall_result, delimiter=",", fmt='%10.5f')

    # s= file_name +'     :       ' + f1_score(true_labels, predicted_labels,average='macro')
    # with open("f_final_score.csv", 'wb') as f_final_file:
    #     np.savetxt(f_final_file, s, delimiter=",", fmt='%10.5f')


def entropy_method(mdl,X,  unlabeled_indices ):    # compute the entropies of transduced label distributions
    # uncertainty_index = np.argsort(stats.distributions.entropy(
    #     mld.label_distributions_.T))[::-1]

    p = mdl.predict_proba(X)
    uncertainty_index = np.argsort( stats.distributions.entropy(p.T))[::-1]

    uncertainty_index = uncertainty_index[
                            np.in1d(uncertainty_index, unlabeled_indices)][:40]
    return uncertainty_index

def ts_method(model):
    sensor_data = dataset.load_data()
    X, y = sensor_data.data[4000:], sensor_data.target[4000:]

    predicted_labels = model.predict(X)
    # predicted_labels = [1 if x > 0.5 else 0 for x in p[:, 1]]
    cm = confusion_matrix(y, predicted_labels,
                          labels=model.classes_)

    print('&' * 70)
    print(cm)
    print(classification_report(y, predicted_labels))
    # print('&' * 70)
    #
    # predicted_labels = [1 if x > 0.54 else 0 for x in p[:, 1]]
    # cm = confusion_matrix(y, predicted_labels,
    #                       labels=model.classes_)
    #
    # print('&' * 70)
    # print(cm)
    # print(classification_report(y, predicted_labels))
    # print('&' * 70)
    # predicted_labels = [1 if x > 0.52 else 0 for x in p[:, 1]]
    # cm = confusion_matrix(y, predicted_labels,
    #                       labels=model.classes_)
    #
    # print('&' * 70)
    # print(cm)
    # print(classification_report(y, predicted_labels))
    # print('&' * 70)

def random_method(mld,X, unlabeled_indices ):
    rng = np.random.RandomState(42)
    indices= np.copy(unlabeled_indices)
    rng.shuffle(indices)
    return indices[:40]


def least_confident_method(mld, unlabeled_indices):
    uncertainty_index = np.argsort(np.max(mld.label_distributions_, axis=1))
    uncertainty_index = uncertainty_index[
                            np.in1d(uncertainty_index, unlabeled_indices)]
    return uncertainty_index



if __name__ == '__main__':
    X, y = load_data()
    PAL(X, y, entropy_method, 'PAL_8')


__author__ = 'M_Nour'

import numpy as np
from scipy import stats
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score, recall_score
import dataset
from collections import Counter
import matplotlib.pyplot as plt
from os.path import join
from dataset import calc_features
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


def make_model3():
    model =  label_propagation.LabelSpreading(kernel='knn', n_neighbors=15)
    sensor_data = dataset.load_data()
    X, y = sensor_data.data[:200], sensor_data.target[:200]
    model.fit(X, y)
    np.savetxt("X.csv", X, delimiter=",", fmt='%10.5f')
    np.savetxt("y_train.csv", y, delimiter=",", fmt='%10.1f')
    return model

def make_model():
    model = RandomForestClassifier(n_estimators=185)#label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
    sensor_data = dataset.load_data()
    X, y = sensor_data.data[:200], sensor_data.target[:200]
    model.fit(X, y)
    np.savetxt("X.csv", X, delimiter=",", fmt='%10.5f')
    np.savetxt("y_train.csv", y, delimiter=",", fmt='%10.1f')
    return model

def make_model2():
    sensor_data = dataset.load_data()
    rng = np.random.RandomState(0)
    indices = np.arange(len(sensor_data.data))
    rng.shuffle(indices)
    print(len(sensor_data.data))
    sm = SMOTE(random_state=42)
    X, y  = sm.fit_sample(sensor_data.data[indices[:2000]], sensor_data.target[indices[:2000]])

    n_total_samples = len(y)
    print(len(y))
    n_labeled_points = 200
    max_iterations = 50
    unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
    lp_model = label_propagation.LabelSpreading(kernel='knn', n_neighbors=15)

    for i in range(max_iterations):
        if len(unlabeled_indices) == 0:
            print("No unlabeled items left to label.")
            break
        y_train = np.copy(y)
        y_train[unlabeled_indices] = -1
        lp_model.fit(X, y_train)
        p = lp_model.predict_proba(X[unlabeled_indices])
        # predicted_labels = [1 if x > 0.57 else 0 for x in p[:, 1]]
        predicted_labels = lp_model.predict(X[unlabeled_indices])

        true_labels = y[unlabeled_indices]
        # print("#"*20 + "Iteration :: " + str(i) + "#"*20)
        # print(classification_report(true_labels, predicted_labels))

        pred_entropies = stats.distributions.entropy(
            lp_model.label_distributions_.T)
        uncertainty_index = np.argsort(pred_entropies)[::-1]
        uncertainty_index = uncertainty_index[
                                np.in1d(uncertainty_index, unlabeled_indices)][:40]
        delete_indices = np.array([])
        for index in uncertainty_index:
            delete_index, = np.where(unlabeled_indices == index)
            delete_indices = np.concatenate((delete_indices, delete_index))
        unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
        n_labeled_points += len(uncertainty_index)




    np.savetxt("X.csv", X, delimiter=",", fmt='%10.5f')
    np.savetxt("y_train.csv", y_train, delimiter=",", fmt='%10.1f')
    return lp_model


def update_data(mdl , x, aq, entropy, threshold):
    x = x.reshape(1, -1)
    p = mdl.predict_proba(x)
    e = stats.distributions.entropy(p.T)
    entropy =np.append(entropy, e)
    if e > threshold and aq>0:
        aq-=1
    return aq,entropy





def customize_model(mdl, W_X, W_Y):
    print("@"*100)
    print(len(W_Y))
    sm = SMOTE(random_state=42)
    # mdl = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)

    X = np.genfromtxt('X.csv', delimiter=',')
    y_train = np.genfromtxt('y_train.csv', delimiter=',')
    # y_train = y_train.reshape(-1,1)
    # W_Y = W_Y.reshape(-1, 1)
    # h = np.arange(len(W_Y))[[i!= -1 for i in W_Y ]]
    # W_Y = W_Y[h]
    # W_X = W_X[h]
    y_train = np.append(y_train, W_Y, axis=0)
    X = np.append(X, W_X, axis=0)
    mdl.fit(X, y_train)

    return mdl




def update_threshold_adaptive_lambda(entropy, k , N, fold_count, AR):

    threshold = sorted(entropy)[::-1][int(AR*k*fold_count/N)]
    return threshold


def update_threshold_static_lambda():
    threshold= 0.65
    return threshold


def customize(mdl, method, name, fold_count, participant_id, AR = 60):
    f_result = []
    acc_result = []
    rec_result =[]
    f_macro_result = []
    rec_macro_result = []
    threshold = 0.691
    entropy = np.array([])

    total_query = 0
    aq = AR
    user_data = np.genfromtxt(join('C:/Users/Marjan/Desktop/Thesis/participants_wild',participant_id, 'wrist_ss.csv'), delimiter=',')
    user_data = user_data[:, 1:]
    data = calc_features(user_data)
    X = preprocessing.scale(data[:, :(user_data.shape[1] - 1) * 9])
    print("*"*20)
    print(X.shape)
    X = np.tile(X, (2, 1))
    #X = np.tile(X, 10) #repeat 10 times
    print(X.shape)
    print("*" * 20)
    y = data[:, -1]
    y = np.tile(y, 2)
    print(y.shape)
    W_Y = np.copy(y)
    print(W_Y.shape)

    N = len(X)
    unlabeled_indices = np.arange(N)
    W_Y [unlabeled_indices] = -1
    # size = int(N//fold_count)
    size = 3600 / 3  # number of instances per hour
    for k, row in enumerate(X):
        if k%size ==  size-1:
            total_query+=(AR- aq)
            aq = AR

            mdl = customize_model(mdl, X[:k+1], W_Y[:k+1])
            predicted_labels = mdl.predict(X[unlabeled_indices])
            # p = mdl.predict_proba(X[unlabeled_indices])
            # predicted_labels = [1 if x > 0.50 else 0 for x in p[:, 1]]

            true_labels = y[unlabeled_indices]
            from collections import  Counter
            print(Counter(true_labels))
            print("size is :: "+ str(size))
            print("k is :: " + str(k))
            print("Iteration :: "+ str((k+1) /size)+ " Queried :: "+ str(total_query))
            print(classification_report(true_labels, predicted_labels))
            f_macro_result.append(f1_score(true_labels, predicted_labels, average='macro'))
            acc_result.append(accuracy_score(true_labels, predicted_labels))
            rec_macro_result.append(recall_score(true_labels, predicted_labels, average='macro'))
            f_result.append(f1_score(true_labels, predicted_labels, average='binary'))
            rec_result.append(recall_score(true_labels, predicted_labels, average='binary'))

        a, entropy = update_data(mdl, row, aq, entropy,threshold)
        if a < aq:
            aq = a
            unlabeled_indices = np.delete(unlabeled_indices, np.where(unlabeled_indices == k))
            W_Y[k] = y[k]
        if method.__name__ == 'update_threshold_adaptive_lambda':
            threshold = method(entropy, k, N, fold_count, AR)
        else:
            threshold = method()
           # print("%.5f" % threshold)
    np.savetxt("en.csv", sorted(entropy), delimiter=",", fmt='%10.5f')
    with open("f1_macro_score_" + name+ '_'+ participant_id + ".csv", 'wb') as afile:
        np.savetxt(afile, f_macro_result, delimiter=",", fmt='%10.5f')
    with open("accuracy_score_" + name + '_' + participant_id + ".csv", 'wb') as afile:
        np.savetxt(afile, acc_result, delimiter=",", fmt='%10.5f')

    with open("recall_macro_score_" + name + '_' + participant_id + ".csv", 'wb') as afile:
        np.savetxt(afile, rec_macro_result, delimiter=",", fmt='%10.5f')

    with open("f1_score_" + name+ '_'+ participant_id + ".csv", 'wb') as afile:
        np.savetxt(afile, f_result, delimiter=",", fmt='%10.5f')

    with open("recall_score_" + name + '_' + participant_id + ".csv", 'wb') as afile:
        np.savetxt(afile, rec_result, delimiter=",", fmt='%10.5f')


    predicted_labels = mdl.predict(X[unlabeled_indices])
    # predicted_labels = [1 if x > 0.5 else 0 for x in p[:, 1]]

    true_labels = y[unlabeled_indices]
    print(classification_report(true_labels, predicted_labels))




def depict_uncertainty(model):
    sensor_data = dataset.load_data()
    y_validate = sensor_data.test_target
    X_validate= sensor_data.test_data
    print(len(X_validate))
    p = model.predict_proba(X_validate)
    e =np.sort(stats.distributions.entropy(p.T))[::-1]
    print(e[0])
    for i in range(10):
        print(e[(i+1)*100])

    plt.scatter(e[:100], np.zeros_like(e[:100]))
    plt.ylabel("Uncertainty")
    plt.legend(loc=2)
    plt.show()


def customize_with_perfect_lambda(mdl, name, fold_count, participant_id, AR=15):
    f_result = []
    acc_result = []

    user_data = np.genfromtxt(join('C:/Users/Marjan/Desktop/Thesis/participants_wild', participant_id,'wrist_ss.csv'), delimiter=',')
    user_data = user_data[:, 1:]
    data = calc_features(user_data)

    X_n = preprocessing.scale(data[:, :(user_data.shape[1] - 1) * 9])
    y_n = data[:, -1]
    rng = np.random.RandomState(0)
    indices = np.arange(len(X_n))
    rng.shuffle(indices)
    X,y = X_n[indices[:]],y_n[indices[:]]
    W_Y = np.copy(y)
    N = len(X)
    unlabeled_indices = np.arange(N)
    W_Y[unlabeled_indices] = -1
    size = int(N // fold_count)
    for i in range(fold_count):
        current_X = np.array(X[i*size: (i+1)*size])
        # current_Y = W_Y[i*size: (i+1)*size]
        p = mdl.predict_proba(current_X)
        e = np.sort(stats.distributions.entropy(p.T))[::-1]

        # select up to 5 digit examples that the classifier is most uncertain about
        uncertainty_index = np.argsort(e)[::-1]

        uncertainty_index = uncertainty_index[
                                np.in1d(uncertainty_index, unlabeled_indices)]

        # keep track of indices that we get labels for
        delete_indices = np.array([])
        eating_counter = 0
        non_eating_counter = 0
        for index in uncertainty_index:
            if eating_counter + non_eating_counter < AR:
                delete_index, = np.where(unlabeled_indices == index)

                is_changed = False
                if y[delete_index] ==0 and non_eating_counter < int(0.5*AR):
                    non_eating_counter += 1
                    is_changed = True
                elif y[delete_index] == 1:
                    eating_counter += 1
                    is_changed = True
                if is_changed:
                    delete_indices = np.concatenate((delete_indices, delete_index))


        unlabeled_indices = np.delete(unlabeled_indices, delete_indices)

        W_Y = np.copy(y)
        W_Y[unlabeled_indices] = -1
        mdl = customize_model(mdl, X[:(i + 1) * size], W_Y[:(i + 1) * size])
        # p = mdl.predict_proba(X[unlabeled_indices])
        # predicted_labels = [1 if x > 0.50 else 0 for x in p[:, 1]]
        predicted_labels = mdl.predict(X[unlabeled_indices])
        true_labels = y[unlabeled_indices]
        print("Iteration :: " + str((i + 1)))
        print(classification_report(true_labels, predicted_labels))
        f_result.append(f1_score(true_labels, predicted_labels, average='binary'))
        acc_result.append(accuracy_score(true_labels, predicted_labels))
    with open("f1_score_"+ name + '_' + participant_id +  ".csv", 'wb') as afile:
        np.savetxt(afile, f_result, delimiter=",", fmt='%10.5f')

if __name__ == "__main__":
    # model= make_model()
    # customize(model)
    print("finished")
    participants = [2, 4, 5, 7]
    model =make_model()

    for p in participants:
        customize(model, update_threshold_adaptive_lambda, 'spal', 10, str(p))


    #depict_uncertainty(model)
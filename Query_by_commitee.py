import numpy as np
from sklearn.semi_supervised import label_propagation
from sklearn.svm import SVC
from sklearn import utils
from sklearn.metrics import classification_report, accuracy_score
from sklearn import linear_model
import dataset
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib import pyplot as plt

max_iterations = 50

from scipy import stats
def vote_entropy(C, y):
    result = stats.entropy(y)
    result = np.array([])
    for x in y.T:
        d = Counter(x)
        disagreement = stats.entropy([*d.values()])

        # disagreement = -np.sum([i/C* np.log2(i/C) for i in d.values()])

        result = np.append(result, disagreement)

    # print(Counter(result))
    return result


def query_by_committee():
    n_labeled_points = 300

    sensor_data = dataset.load_data()
    rng = np.random.RandomState(0)
    indices = np.arange(len(sensor_data.data))
    rng.shuffle(indices)
    X = sensor_data.data[indices[:4000]]
    y = sensor_data.target[indices[:4000]]
    n_total_samples = len(y)
    unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]

    committee = [
            # KNeighborsClassifier(3),
            #      SVC(kernel="linear", C=0.025),
                 SVC(gamma=2, C=1),
                GaussianProcessClassifier(1.0 * RBF(1.0)),
                # DecisionTreeClassifier(max_depth=5),
                # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                # AdaBoostClassifier(),
                # GaussianNB(),
                QuadraticDiscriminantAnalysis(),
                 # linear_model.LogisticRegression()
    ]
    plt_x = []
    plt_y =[]
    plt_acc = []
    for iteration in range(max_iterations):
        print(iteration)
        y_train = y[:n_labeled_points]
        X_train = X[:n_labeled_points]
        X_test = X[n_labeled_points:]
        y_test = y[n_labeled_points:]

        predicted = np.array([])
        for i,model in enumerate(committee):
            model.fit(X_train, y_train)
            if i==0:
                predicted = model.predict(X_test)
            else:
                predicted = np.vstack((predicted,model.predict(X_test) ))
            print("_____________"+ str(i)+ "_____________")
            print(accuracy_score(y_test, model.predict(X_test)))
        voted = vote_entropy(len(committee), predicted)
        y_predicted= np.array([])

        for item in predicted.T:


            d = Counter(item)
            y_x = max(d.items(), key=lambda x: x[1])[0]
            y_predicted = np.append(y_predicted ,y_x )

        # print(classification_report(y_test, y_predicted))
        # print(accuracy_score(y_test, y_predicted))

        uncertain_points = (np.argsort(voted)[::-1])[:20]
        # _________________________________________________________
        plt_x.append(iteration)
        plt_acc.append(accuracy_score(y_test, y_predicted))
        plt_y.append(max(Counter(voted).values()))


        #__________________________________________________________

        delete_indices = np.array([])
        for index in uncertain_points:
            delete_index, = np.where(unlabeled_indices == index)
            delete_indices = np.concatenate((delete_indices, delete_index))
        unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
        n_labeled_points += len(uncertain_points)

    plt.plot(plt_x, plt_y)
    plt.show()


if __name__ == '__main__':
    query_by_committee()


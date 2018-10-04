from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics.classification import accuracy_score

__author__ = 'M_Nour'


from scipy import *
from scipy.stats import *
import numpy as np
from sklearn import utils, model_selection
from sklearn.model_selection import train_test_split
from os.path import dirname,join
from sklearn import preprocessing
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss,CondensedNearestNeighbour
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

def calc_features(Z, step_size = 75, frame_size= 150):
    number_of_inputs= Z.shape[1] - 1
    # print(number_of_inputs)
    for counter in range(0, len(Z), step_size):

        # Get the label
        L = Z[counter, number_of_inputs]

        # Get rows from which to calculate features
        R = Z[counter:counter + frame_size, :number_of_inputs]

        M = mean(R, axis=0)
        V = var(R, axis=0)
        SK = stats.skew(R, axis=0)
        K = stats.kurtosis(R, axis=0)
        RMS = sqrt(mean(R ** 2, axis=0))
        med = median(R, axis=0)
        maxvalue = R.max( axis=0)
        minvalue = R.min( axis=0)

        p2p = np.ptp(R, axis=0)
        # variance = sig.var(axis=0)
        # percent = np.percentile(sig, axis=0, q=[25, 50, 75])
        # # stdvalue = sig.std(axis=0)
        # # rms = np.sqrt(int(sum(np.square(sig)))/ len(sig))
        # a, b = sig.shape
        # hists = []
        # for i in range(b):
        #     hists.extend(np.histogram(sig[:, i], bins=np.arange(0, 99, 10))[0])


        H = np.hstack((M, V))
        H = np.hstack((H, SK))
        H = np.hstack((H, K))
        H = np.hstack((H, RMS))
        H = np.hstack((H, med))
        H = np.hstack((H, maxvalue))
        H = np.hstack((H, minvalue))
        H = np.hstack((H, p2p))

        H = np.hstack((H, L))
        if counter == 0:
            All = H
        else:
            All = np.vstack((All, H))
    # print(All.shape)
    return All


def calc_y(Z, step_size = 75):
    y = np.array([])
    number_of_inputs= Z.shape[1] - 1
    for counter in range(0, len(Z), step_size):

        # Get the label
        l = Z[counter, number_of_inputs]
        y = np.append(y, l)
    return y


def make_data():

    frame_size_seconds = 6
    step_size_seconds = int(frame_size_seconds / 2)
    sampling_rate = 25

    # Set the frame and step size
    frame_size = frame_size_seconds * sampling_rate
    step_size = step_size_seconds * sampling_rate

    # Get training file for each participant
    first_time_in_loop = 1
    module_path = dirname(__file__)

    for target_participant_counter in range(1,22):

        if target_participant_counter == 14:
            continue

        D = np.genfromtxt(join(module_path, 'participants',str(target_participant_counter),'datafiles' , 'waccel_tc_ss_label.csv'),
                      delimiter=',')

        if first_time_in_loop == 1:
            first_time_in_loop = 0
            Z = D
        else:
            Z = np.vstack((Z, D))

    # Z = np.vstack((Z,
    #                np.genfromtxt(join('C:/Users/Marjan/Desktop/Thesis/participants_wild/3', 'wrist_ss.csv'),
    #                              delimiter=',')))

    test_data  = np.genfromtxt(join( 'C:/Users/Marjan/Desktop/Thesis/participants_wild/2', 'wrist_ss.csv'),delimiter=',')
    # test_data = np.vstack((test_data ,np.genfromtxt(join( 'C:/Users/Marjan/Desktop/Thesis/participants_wild/', 'wrist_ss.csv'),delimiter=',') ))
    # Remove the relative timestamp
    Z = Z[:, 1:]
    print(Z.shape)
    test_data = test_data[:,1:]
    print(test_data.shape)

    # Number of inputs
    number_of_inputs = Z.shape[1] - 1

    # -----------------------------------------------------------------------------------
    #
    #									Training
    #
    # -----------------------------------------------------------------------------------

    # Calculate features for frame

    All = calc_features(Z, step_size, frame_size )
    All_Test = calc_features(test_data, step_size, frame_size )
    X_test = preprocessing.scale(All_Test[:, :number_of_inputs * 9])
    Y_test = All_Test[:, number_of_inputs * 9]




    X = preprocessing.scale(All[:, :number_of_inputs * 9])
    Y = All[:, number_of_inputs * 9]
    # sm = SMOTE(random_state=42)
    # #
    # X_res, y_res = sm.fit_sample(X, Y)


    np.savetxt("X_file.csv", X, delimiter=",", fmt='%10.5f')
    np.savetxt("Y_file.csv", Y, delimiter=",", fmt= '%10.1f')

    np.savetxt("X_test_file.csv", X_test, delimiter=",", fmt='%10.5f')
    np.savetxt("Y_test_file.csv", Y_test, delimiter=",", fmt='%10.1f')
    print('$'*200)
    print(len(X))

    return utils.Bunch(data=X,target=Y , test_data = X_test , test_target = Y_test)


def run_classifier():
    pass
    # cnn = NearMiss()
    # X_res, y_res = cnn.fit_sample(X, Y)


    # _________________
    # from collections import Counter
    # print(Counter(Y))
    # rng = np.random.RandomState(0)
    # indices = np.arange(len(X))
    # rng.shuffle(indices)
    # X = X_res[indices[:4000]]
    # y = y_res[indices[:4000]]
    #
    # X_train, X_test, y_train, y_test = train_test_split(X_res[:4000], y_res[:4000], random_state=42, test_size=0.33)
    #
    # # Run classifier
    # classifier = SVC(gamma=2, C=1)
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print(accuracy_score(y_test,y_pred))
    #
    # clf = RandomForestClassifier(n_estimators=185)
    # clf.fit(X_train, y_train)
    # y = clf.predict(X_test)
    # print(classification_report(y_test, y))
    # print(accuracy_score(y_test,y))
def load_data():

    X = np.genfromtxt ('X_file.csv', delimiter=",")

    Y = np.genfromtxt ('Y_file.csv', delimiter=",")

    test_data = np.genfromtxt('X_test_file.csv', delimiter=",")

    test_target = np.genfromtxt('Y_test_file.csv', delimiter=",")

    return utils.Bunch(data=X,target=Y, test_data= test_data, test_target = test_target)


# def example():
#
#     X, y = make_classification(n_classes=2, class_sep=2,weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
#                                n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
#     print('Original dataset shape {}'.format(Counter(y)))
#     # Original dataset shape Counter({1: 900, 0: 100})
#     sm = SMOTE(random_state=42)
#     X_res, y_res = sm.fit_sample(X, y)
#     print('Resampled dataset shape {}'.format(Counter(y_res)))
#     #Resampled dataset shape Counter({0: 900, 1: 900})

if __name__=='__main__':
    make_data()
    # example()



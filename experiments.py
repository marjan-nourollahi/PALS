import My_Active_Learning as pal
import Active_Learning_SVM as other_approaches
import stream_PAL as spal
import numpy as np
from matplotlib import pyplot as plt

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
from multiprocessing import Process



"""
Random vs. Entropy vs. Least Confident PAL
"""

def first_experiment():
    X, y = pal.load_data()
    # pal.PAL(pal.least_confident_method, 'PAL_Least_Confident')
    pal.PAL(X, y,pal.entropy_method, 'PAL_Entropy')
    pal.PAL(X, y,pal.random_method, 'PAL_Random')
    file_names = ['PAL_Entropy', 'PAL_Random']
    names = ['Entropy', 'Random']
    pal_depict_measurments(file_names, names)


#
#  knn vs rbf kernels"
#


def second_experiment():
    # X, y = pal.load_data()
    # pal.PAL(X,y ,pal.entropy_method, 'PAL_RBF',False,  {'gamma': .25, 'max_iter': 5})
    # # pal.PAL(pal.entropy_method, 'PAL_RBF_10', False, {'gamma': 10, 'max_iter': 15})
    # pal.PAL(X,y, pal.entropy_method, 'PAL_KNN',False, {'kernel':'knn','n_neighbors':30})
    # pal.PAL(pal.entropy_method, 'PAL_KNN_30', False, {'n_neighbors': 30})
    # file_names = ['PAL_200_1', 'PAL_4','PAL_6', 'PAL_8']

    file_names = ['PAL_RBF','PAL_KNN']
    names = ['RBF', 'KNN']
    pal_depict_measurments(file_names, names)

#
# PAL vs Other Approaches
# model = SVC(gamma=2, C=1)
# model = linear_model.LogisticRegression()
# model = QuadraticDiscriminantAnalysis()#KNeighborsClassifier(3)
# model  = AdaBoostClassifier()
# model = RandomForestClassifier( n_estimators=185)
#


def third_experiment():
    #
    X, y = pal.load_data()
    print('^'*40)
    print(len(X))

    #
    # p1 = Process(target=pal.PAL, args=(X, y,  pal.entropy_method,'PAL_200',))
    # p2 = Process(target=other_approaches.explore, args=(X, y,  'svm_200',SVC, {'gamma':2, 'C': 1, 'probability':True},))
    # p3 = Process(target=other_approaches.explore, args=(X, y,  'LR_200', linear_model.LogisticRegression,))
    # p4 = Process(target=other_approaches.explore, args=(X, y, 'RFC_200',RandomForestClassifier, {'n_estimators':185} ,))
    # p7 = Process(target=other_approaches.explore,
    #              args=(X, y, 'RFC_200_X_1', RandomForestClassifier,))
    #
    # p5 = Process(target=other_approaches.explore, args=(X, y, 'QDA_200',QuadraticDiscriminantAnalysis,))
    # p6 = Process(target=other_approaches.explore, args=(X, y,  'ABC_200',AdaBoostClassifier,))
    #
    # processes = [p1,p2, p3, p4, p5, p6, p7]
    # for p in processes:
    #     p.start()
    # for p in processes:
    #     p.join()

    # p1.start()
    # p1.join()

    # pal.PAL(X, y, pal.entropy_method,'PAL_20')
    # other_approaches.explore(X, y, 'svm_20',SVC, {'gamma':2, 'C': 1, 'probability':True} )
    # other_approaches.explore(X, y, 'LR_20', linear_model.LogisticRegression)
    other_approaches.explore(X, y, 'RFC_20',RandomForestClassifier, {'n_estimators':185} )
    # other_approaches.explore(X, y, 'QDA_20',QuadraticDiscriminantAnalysis)
    # other_approaches.explore(X, y, 'ABC_20',AdaBoostClassifier)

    # file_names = ['PAL_200', 'RFC_200','QDA_200', 'ABC_200', 'svm_200','LR_200']
    # names = ['PAL', 'RFC','QDA', 'ABC', 'svm','LR']
    #
    # pal_depict_measurments(file_names, file_names, type_of_result='recall_score')


"""
SPAL: 
Experimental upper bond: Best lambda
SPAL: Adaptive Lambda
Random: No Lambda
Static Lambda
"""


def spal_first_experiment():
    fold_count = 10
    # file_names = ["experimental_upper_bound",'adaptive_lambda', 'static_lambda' ]
    file_names = ['adaptive_lambda','simple','adaptive_lambda_no_model']#, 'adaptive_lambda','simple_model']#["experimental_upper_bound_15", 'adaptive_lambda_15', 'static_lambda_15']
    model_names = ['(1)','(2)','(3)']#,'PAL as base model', 'Simple Model'] #['Experimental Upper Bound', 'Adaptive Lambda', 'Static Lambda']
    participants = [2,4,5,7]

    # for p in participants:
    #     spal.customize(spal.make_model2(), spal.update_threshold_adaptive_lambda, file_names[0], fold_count, str(p))
        # spal.customize(spal.make_model3(), spal.update_threshold_adaptive_lambda, file_names[1], fold_count, str(p))
        # spal.customize(spal.make_model(), spal.update_threshold_adaptive_lambda, file_names[2], fold_count, str(p))


        # spal.customize(spal.make_model(), spal.update_threshold_static_lambda,file_names[2] ,fold_count, str(p))
        # spal.customize_with_perfect_lambda(spal.make_model(), file_names[0], fold_count ,str(p))


    # ************************chart**********************************


    depict_f_measure(file_names, model_names, max_iter=fold_count,xlabel='Number of time intervals' )


"""
Budget average for all subjects
"""


def spal_second_experiment():
    fold_count = 1
    file_names = ["SPAL_5",'SPAL_10', 'SPAL_15' ,'SPAL_20','SPAL_25','SPAL_30',"SPAL_35",'SPAL_40', 'SPAL_45' ,'SPAL_50','SPAL_55','SPAL_60']
    model_names= list(map(str , range(5, 61, 5)))
    delta = np.arange(5, 61, 5)
    participants = [2,4,5,7]

    # model = spal.make_model2()
    # for p in participants:
    #     for name, aq in zip(file_names, delta):
    #         spal.customize( model, spal.update_threshold_adaptive_lambda,name , fold_count, str(p),aq )
    # #
    # # ************************chart**********************************
    #

    depict_f_measure(file_names, model_names, max_iter=12 ,xlabel='Budget/Hour')

"""
fig, ax = plt.subplots()
plt.show()"""

def depict_f_measure_bar_chart(file_names, model_names, bins=6, xlabel='Budget'):
    y_axis_f= []
    bar_width = 0.25
    for model, name in zip(file_names, model_names):
        y = 0.0
        for i in [2,4,5,7]:
            y += np.genfromtxt(('f1_macro_score_'+model + '_'+ str(i) +".csv"), delimiter=',')

        y = np.divide(y, np.array([4.0], dtype=float))
        y_axis_f.append(y)

    y_axis_r = []
    for model, name in zip(file_names, model_names):
        y = 0.0
        for i in [2,4,5,7]:
            y += np.genfromtxt(('recall_macro_score_'+model + '_'+ str(i) +".csv"), delimiter=',')

        y = np.divide(y, np.array([4.0], dtype=float))
        y_axis_r.append(y)


    y_axis_a = []
    for model, name in zip(file_names, model_names):
        y = 0.0
        for i in [2, 4, 5, 7]:
            y += np.genfromtxt(('accuracy_score_' + model + '_' + str(i) + ".csv"), delimiter=',')

        y = np.divide(y, np.array([4.0], dtype=float))
        y_axis_a.append(y)

    index = np.arange(6)
    opacity = 0.8

    fig, ax = plt.subplots()

    rects1 = ax.bar(index, y_axis_f, bar_width,
                    alpha=opacity, color='b',
                    label='f-score')

    rects2 = ax.bar(index +1.1* bar_width, y_axis_r, bar_width,
                    alpha=opacity, color='r',
                    label='recall')

    # rects3 = ax.bar(index +2.2* bar_width, y_axis_a, bar_width,
    #                 alpha=opacity, color='g',
    #                 label='accuracy')

    ax.set_xlabel('Query Budget/ Hour',fontsize='x-large')
    ax.set_ylabel('Scores',fontsize='x-large')
    ax.set_xticks(index + bar_width )
    ax.set_xticklabels(np.arange(10, 161, 30))
    ax.legend()
    fig.tight_layout()

    plt.show()


def depict_f_measure(file_names, model_names, max_iter=50, xlabel='Number of Iterations'):
    ys= []
    for model, name in zip(file_names, model_names):
        y = np.zeros((4,))
        print(name)
        for i in [2,4,5,7]:
            print(i)
            y = np.sum([y, np.genfromtxt(('f1_macro_score_'+model + '_'+ str(i) +".csv"), delimiter=',')], axis=0)

        y = np.divide(y, np.array([4.0], dtype=float))
        ys.append(y)
    x = np.arange(5, max_iter*5+1, 5)
    plt.ylim([0.4, 0.7])
    plt.plot(x, ys )

    # for i in [2,4,5,7]:
    #     for model, name in zip(file_names, model_names):
    #         y = np.genfromtxt(('f1_score_'+model + '_'+ str(i) +".csv"), delimiter=',')
    #
    #         x = np.arange(1, max_iter+1)
    #         plt.plot(x, y, label=name+'_'+ str(i))
    plt.ylabel('F Score', fontsize='x-large')
    plt.legend(fontsize='x-large')
    plt.xlabel(xlabel, fontsize='x-large')
    plt.tight_layout()
    plt.show()


def pal_depict_measurments(file_names, model_names, max_iter=50, xlabel='Number of Iterations', type_of_result = 'f1_score'):
    for model, name in zip(file_names, model_names):
            y = np.genfromtxt((type_of_result + '_'+model + ".csv"), delimiter=',')
            x = np.arange(0, max_iter)
            plt.plot(x, y, label=name )
    plt.ylabel((' '.join(type_of_result.split('_')).title()), fontsize='x-large')
    plt.legend(fontsize='x-large')
    plt.xlabel(xlabel, fontsize='x-large')
    plt.tight_layout()
    plt.xticks(np.arange(0,max_iter+1, 5))
    plt.show()

def test_size():
    from os.path import join
    from dataset import calc_features
    from sklearn import preprocessing
    from collections import Counter
    participants = [2,4,5,7]
    for p in participants:
        user_data = np.genfromtxt(join('C:/Users/Marjan/Desktop/Thesis/participants_wild', str(p), 'wrist_ss.csv'), delimiter=',')
        user_data = user_data[:, 1:]
        data = calc_features(user_data)
        # X = preprocessing.scale(data[:, :(user_data.shape[1] - 1) * 9])

        y = data[:, -1]
        print(Counter(y))



if __name__ == '__main__':
    spal_second_experiment()
    # second_experiment()
    # third_experiment()
    # test_size()
    # spal_first_experiment()
    # first_experiment()
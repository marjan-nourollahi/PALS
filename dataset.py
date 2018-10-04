__author__ = 'Marjan Nourollahi'

from scipy import *
from scipy.stats import *
import numpy as np
from os.path import dirname,join
from sklearn import preprocessing


def calc_features(data, step_size = 75, frame_size= 150):
    row, columns = data.shape

    for counter in range(0, row, step_size):
        frame = data[counter:counter + frame_size, :columns]

        m = mean(frame, axis=0)
        v = var(frame, axis=0)
        skew = stats.skew(frame, axis=0)
        kurtosis = stats.kurtosis(frame, axis=0)
        rms = sqrt(mean(frame ** 2, axis=0))
        med = median(frame, axis=0)
        maxvalue = frame.max( axis=0)
        minvalue = frame.min( axis=0)
        p2p = np.ptp(frame, axis=0)


        temp = np.hstack((m, v))
        temp = np.hstack((temp, skew))
        temp = np.hstack((temp, kurtosis))
        temp = np.hstack((temp, rms))
        temp = np.hstack((temp, med))
        temp = np.hstack((temp, maxvalue))
        temp = np.hstack((temp, minvalue))
        temp = np.hstack((temp, p2p))

        result = temp if counter == 0 else np.vstack((result, temp))

    return result


def calc_y(data, step_size=75):

    y = np.array([])
    row_num, column_num = data.shape
    
    for counter in range(0, row_num, step_size):
        l = 1 if sum(data[counter:counter+ step_size, column_num])>= step_size/2 else 0
        y = np.append(y, l)
        
    return y


def make_data():

    NUMBER_OF_TRAIN_FILES = 1
    NUMBER_OF_TEST_FILES = 1
    WINDOW_SIZE = 6
    SAMPLING_RATE = 25

    frame_size = WINDOW_SIZE * SAMPLING_RATE
    step_size =  int(WINDOW_SIZE / 2) * SAMPLING_RATE

    first_iteration = True
    module_path = dirname(__file__)

    for file_id in range(1 , NUMBER_OF_TRAIN_FILES + 1):

        raw_data = np.genfromtxt(join(module_path, 'train/DATA_',str(file_id), '.csv'),
                      delimiter=',')

        train_data = raw_data if first_iteration else np.vstack((train_data, raw_data))
        first_iteration = False

    first_iteration = True
    for file_id in range(1 , NUMBER_OF_TEST_FILES + 1):

        raw_data = np.genfromtxt(join(module_path, 'test/DATA_',str(file_id), '.csv'),
                      delimiter=',')

        test_data = raw_data if first_iteration else np.vstack((test_data, raw_data))
        first_iteration = False

    X = preprocessing.scale(calc_features(train_data, step_size, frame_size ))
    Y = calc_y(train_data, step_size)

    X_test = preprocessing.scale(calc_features(test_data, step_size, frame_size ))
    Y_test= calc_y(test_data, step_size)

    np.savetxt(join(module_path, "data/X_file.csv"), X, delimiter=",", fmt='%10.5f')
    np.savetxt(join(module_path, "data/Y_file.csv"), Y, delimiter=",", fmt= '%10.1f')

    np.savetxt(join(module_path, "data/X_test_file.csv"), X_test, delimiter=",", fmt='%10.5f')
    np.savetxt(join(module_path, "data/Y_test_file.csv"), Y_test, delimiter=",", fmt='%10.1f')

    return utils.Bunch(data=X,target=Y , test_data = X_test , test_target = Y_test)


def load_data():
    module_path = dirname(__file__)

    X = np.genfromtxt (join(module_path, 'data/X_file.csv'), delimiter=",")

    Y = np.genfromtxt (join(module_path, 'data/Y_file.csv'), delimiter=",")

    test_data = np.genfromtxt(join(module_path, 'data/X_test_file.csv'), delimiter=",")

    test_target = np.genfromtxt(join(module_path, 'data/Y_test_file.csv'), delimiter=",")

    return utils.Bunch(data=X,target=Y, test_data=test_data, test_target=test_target)


if __name__ == '__main__':
    make_data()




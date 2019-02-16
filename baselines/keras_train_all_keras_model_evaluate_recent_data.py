from params import read_args
from keras_pickel_data import loading_variable
from keras_model import lstm_cnn_model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import metrics


def auc_score(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    return metrics.auc(fpr, tpr)


if __name__ == '__main__':
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    # commits = loading_variable('train_all_recent_data_unknown_labels')
    commits = loading_variable('train_all_recent_new_data_unknown_labels')
    train_data, test_data, dict_ = commits
    train_labels, train_commits = train_data
    test_labels, test_commits = test_data

    print(train_labels.shape, train_commits.shape, len(dict_))
    print(test_labels.shape, test_commits.shape)

    path_file_model, model_name, model_variation = "./keras_model/", "lstm_cnn_all-", 50
    path_file_save = './keras_model_results/'
    for i in xrange(1, model_variation + 1):
        print(path_file_model + model_name + '{:02d}'.format(int(i)) + ".hdf5")
        model = lstm_cnn_model(dictionary_size=len(dict_), FLAGS=input_option)
        model.load_weights(path_file_model + model_name + '{:02d}'.format(int(i)) + ".hdf5")
        y_pred = model.predict(test_commits, batch_size=input_option.batch_size)
        y_pred = np.ravel(y_pred)
        # np.save(path_file_save + model_name + 'recent_data-' + '{:02d}'.format(int(i)) + '.txt', y_pred)
        np.save(path_file_save + model_name + 'recent_new_data-' + '{:02d}'.format(int(i)) + '.txt', y_pred)

        # y_pred = np.ravel(y_pred)
        # y_pred[y_pred > 0.5] = 1
        # y_pred[y_pred <= 0.5] = 0
        #
        # acc = accuracy_score(y_true=test_labels, y_pred=y_pred)
        # prc = precision_score(y_true=test_labels, y_pred=y_pred)
        # rc = recall_score(y_true=test_labels, y_pred=y_pred)
        # f1 = f1_score(y_true=test_labels, y_pred=y_pred)
        # auc = auc_score(y_true=test_labels, y_pred=y_pred)
        # print ('Acc: %f -- Prc: %f -- Rc: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc))

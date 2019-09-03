import numpy as np
from keras_ultis import reformat_commit_code_keras, extract_commit_recent_sasha_keras_ver1
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras_train_split_keras_model_evaluate import auc_score
from keras_ultis import write_file


def get_false_positive(commits, y_pred, variation):
    dict_false_pos = {}
    for c, p in zip(commits, y_pred):
        id_c, label_c = c['id'], c['stable']
        if label_c != 'true':
            print(id_c, p)
            dict_false_pos[id_c] = p
    false_pos = list()
    for key, value in sorted(dict_false_pos.iteritems(), reverse=True, key=lambda (k, v): (v, k)):
        print(key + '\t' + str(value))
        false_pos.append(key + '\t' + str(value))
    write_file(path_file='./results/recent_new_funcs_false_positive_results_' + variation + '.txt', data=false_pos)


if __name__ == '__main__':
    # # loading recent linux data
    # path_file = '../data/recent_functions_translated.out'
    # commits_recent_data = extract_commit_recent_sasha_keras_ver1(path_file=path_file)
    # commits_recent_data = reformat_commit_code_keras(commits=commits_recent_data, num_file=1)
    # label_commits = [1 if c['stable'] == 'true' else 0 for c in commits_recent_data]
    # id_train_commits = [d['id'] for d in commits_recent_data]
    #
    # path_results = './keras_model_results/'
    # path_model = 'lstm_cnn_all-recent_new_data-'
    # path_variation = '05'
    # # path_variation = '10'
    # path_file = path_results + path_model + path_variation + '.txt.npy'
    # y_pred = np.load(path_file)

    # y_pred = np.ravel(y_pred)
    # y_pred[y_pred > 0.5] = 1
    # y_pred[y_pred <= 0.5] = 0
    #
    # acc = accuracy_score(y_true=label_commits, y_pred=y_pred)
    # prc = precision_score(y_true=label_commits, y_pred=y_pred)
    # rc = recall_score(y_true=label_commits, y_pred=y_pred)
    # f1 = f1_score(y_true=label_commits, y_pred=y_pred)
    # auc = auc_score(y_true=label_commits, y_pred=y_pred)
    # print ('Acc: %f -- Prc: %f -- Rc: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc))

    #########################################################################################################
    # loading recent linux data
    # path_file = '../data/recent_functions_translated.out'
    path_file = '../data/recent_functions_translated_ver1.out'
    commits_recent_data = extract_commit_recent_sasha_keras_ver1(path_file=path_file)
    commits_recent_data = reformat_commit_code_keras(commits=commits_recent_data, num_file=1)
    label_commits = [1 if c['stable'] == 'true' else 0 for c in commits_recent_data]
    id_train_commits = [d['id'] for d in commits_recent_data]

    path_results = './keras_model_results/'
    # path_model = 'lstm_cnn_all-recent_new_data-'
    path_model = 'lstm_cnn_all-recent_new_data_ver1-'
    path_variation = '05'
    # path_variation = '10'
    path_file = path_results + path_model + path_variation + '.txt.npy'
    y_pred = np.load(path_file)

    get_false_positive(commits=commits_recent_data, y_pred=list(y_pred), variation=path_variation)
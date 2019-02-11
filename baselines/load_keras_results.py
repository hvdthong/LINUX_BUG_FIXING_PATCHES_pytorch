import numpy as np
from keras_ultis import extract_commit, reformat_commit_code_keras, extract_commit_recent_sasha, \
    extract_commit_recent_sasha_keras, write_file


def get_id_commits(data):
    return [d['id'] for d in data]


def get_results(commits, results):
    cnt_true, cnt_false = 0, 0
    for c, r in zip(commits, results):
        if float(r) >= 0.5:
            print(c + '\t' + str(r) + '\t' + 'True')
            cnt_true += 1
        else:
            print(c + '\t' + str(r) + '\t' + 'False')
            cnt_false += 1
    print(cnt_true, cnt_false)


def get_recent_results(commits, results, variation):
    dict_ = {}
    for c, r in zip(commits, results):
        dict_[c] = r
    new_results = list()
    for key, value in sorted(dict_.iteritems(), reverse=True, key=lambda (k, v): (v, k)):
        print(key + '\t' + str(value))
        new_results.append(key + '\t' + str(value))
    write_file(path_file='./results/recent_funcs_results_' + variation + '.txt', data=new_results)


if __name__ == '__main__':
    ###########################################################################################
    # path_file = '../data/recent_sashas_functions_translated_nodups.out'
    # commits_recent_sasha = extract_commit_recent_sasha_keras(path_file=path_file)
    # commits_recent_sasha = reformat_commit_code_keras(commits=commits_recent_sasha, num_file=1)
    #
    # # loading training data
    # path_file = '../data/newres_funcalls_jul28.out.sorted'
    # commits = extract_commit(path_file=path_file)
    # commits = reformat_commit_code_keras(commits=commits, num_file=1)
    #
    # # add recent sasha data with training data
    # train_commits = commits + commits_recent_sasha
    # train_commits = train_commits[82403:]
    # id_train_commits = get_id_commits(data=train_commits)
    #
    # path_results = './keras_model_results/'
    # path_model = 'lstm_cnn_all-'
    # path_variation = '04'
    # path_file = path_results + path_model + path_variation + '.txt.npy'
    # y_pred = np.load(path_file)
    # y_pred = y_pred[82403:]
    # get_results(commits=id_train_commits, results=list(y_pred))
    ###########################################################################################

    # loading recent linux data
    path_file = '../data/recent_functions_translated_nodups.out'
    commits_recent_data = extract_commit_recent_sasha_keras(path_file=path_file)
    commits_recent_data = reformat_commit_code_keras(commits=commits_recent_data, num_file=1)
    test_commits = commits_recent_data
    id_train_commits = get_id_commits(data=test_commits)

    path_results = './keras_model_results/'
    path_model = 'lstm_cnn_all-recent_data-'
    # path_variation = '05'
    path_variation = '10'
    path_file = path_results + path_model + path_variation + '.txt.npy'
    y_pred = np.load(path_file)
    get_recent_results(commits=id_train_commits, results=list(y_pred), variation=path_variation)

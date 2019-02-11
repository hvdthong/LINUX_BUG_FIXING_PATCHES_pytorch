from keras_ultis import extract_commit, reformat_commit_code_keras, extract_commit_recent_sasha, \
    extract_commit_recent_sasha_keras
from keras_padding import padding_commit, padding_train_test_commits
from params import read_args
from keras_pickel_data import saving_variable
from data_split import training_testing_split


if __name__ == '__main__':
    # creating pickle data for training and testing
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    # # loading recent sasha data
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
    # # commits_all = commits + commits_recent_sasha
    # # train_all = padding_commit(commits=commits_all, params=input_option)
    # # saving_variable('train_all_recent_keras', variable=train_all)
    #
    # # split training and testing data
    # commits_all = commits + commits_recent_sasha
    # nfolds = 5
    # train_commits, test_commits = training_testing_split(commits=commits_all, nfolds=5, random_state=None)
    # # print(len(train_commits), len(test_commits))
    # train_data, test_data, dict_ = padding_train_test_commits(train=train_commits, test=test_commits, params=input_option)
    # data = (train_data, test_data, dict_)
    # saving_variable('train_split_recent_keras', variable=data)

    ###########################################################################################
    path_file = '../data/recent_sashas_functions_translated_nodups.out'
    commits_recent_sasha = extract_commit_recent_sasha_keras(path_file=path_file)
    commits_recent_sasha = reformat_commit_code_keras(commits=commits_recent_sasha, num_file=1)

    # loading training data
    path_file = '../data/newres_funcalls_jul28.out.sorted'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code_keras(commits=commits, num_file=1)

    # add recent sasha data with training data
    train_commits = commits + commits_recent_sasha

    # loading recent linux data
    path_file = '../data/recent_functions_translated_nodups.out'
    commits_recent_data = extract_commit_recent_sasha_keras(path_file=path_file)
    commits_recent_data = reformat_commit_code_keras(commits=commits_recent_data, num_file=1)
    test_commits = commits_recent_data
    print(len(train_commits), len(test_commits))

    train_data, test_data, dict_ = padding_train_test_commits(train=train_commits, test=test_commits,
                                                              params=input_option)
    data = (train_data, test_data, dict_)
    saving_variable('train_all_recent_data_unknown_labels', variable=data)


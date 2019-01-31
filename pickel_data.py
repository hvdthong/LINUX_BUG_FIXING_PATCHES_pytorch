from ultis import extract_commit, reformat_commit_code
from data_split import training_testing_split
from params import read_args
from padding import padding_train_test_commits, padding_commit
import pickle


def saving_variable(pname, variable):
    f = open('./data/' + pname + '.pkl', 'wb')
    pickle.dump(variable, f, protocol=4)
    f.close()


def loading_variable(pname):
    f = open('./data/' + pname + '.pkl', 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


if __name__ == '__main__':
    # creating pickle data for training and testing
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    # creating training and testing data
    # path_file = './data/newres_funcalls_jul28.out.sorted'
    # commits = extract_commit(path_file=path_file)
    # commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
    #                                num_loc=10, num_leng=120)
    # nfolds, random_state = 5, None
    # train, test = training_testing_split(commits=commits, nfolds=5, random_state=None)
    # train, test, dict_ = padding_train_test_commits(train=train, test=test, params=input_option)
    # saving_variable('train', train)
    # saving_variable('test', test)
    # saving_variable('train_dict', dict_)

    # creating training data
    path_file = './data/newres_funcalls_jul28.out.sorted'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
                                   num_loc=10, num_leng=120)
    train_all = padding_commit(commits=commits, params=input_option)
    saving_variable('train_all', train_all)

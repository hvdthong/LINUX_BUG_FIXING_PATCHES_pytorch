from sklearn.model_selection import KFold


def get_index(data, indexes):
    return [data[i] for i in indexes]


def training_testing_split(commits, nfolds, random_state=None):
    kf = KFold(n_splits=nfolds, random_state=random_state, shuffle=False)
    train_index, test_index = list(kf.split(commits))[nfolds - 1]
    train_commit, test_commit = get_index(data=commits, indexes=train_index), get_index(data=commits,
                                                                                        indexes=test_index)
    return train_commit, test_commit
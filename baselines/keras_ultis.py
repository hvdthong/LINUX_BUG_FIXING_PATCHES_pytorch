#  * This file is part of PatchNet, licensed under the terms of the GPL v2.
#  * See copyright.txt in the PatchNet source code for more information.
#  * The PatchNet source code can be obtained at
#  * https://github.com/hvdthong/PatchNetTool

from keras_extracting import commit_id, commit_stable, commit_msg, commit_date, commit_code
from reformating import reformat_file, reformat_hunk
import numpy as np
import math
import os
from extracting import hunk_code


def load_file(path_file):
    lines = list(open(path_file, "r").readlines())
    lines = [l.strip() for l in lines]
    return lines


def load_dict_file(path_file):
    lines = list(open(path_file, "r").readlines())
    dictionary = dict()
    for line in lines:
        key, value = line.split('\t')[0], line.split('\t')[1]
        dictionary[key] = value
    return dictionary


def write_dict_file(path_file, dictionary):
    split_path = path_file.split("/")
    path_ = split_path[:len(split_path) - 1]
    path_ = "/".join(path_)

    if not os.path.exists(path_):
        os.makedirs(path_)

    with open(path_file, 'w') as out_file:
        for key in dictionary.keys():
            # write line to output file
            out_file.write(str(key) + '\t' + str(dictionary[key]))
            out_file.write("\n")
        out_file.close()


def write_file(path_file, data):
    split_path = path_file.split("/")
    path_ = split_path[:len(split_path) - 1]
    path_ = "/".join(path_)
    if len(split_path) > 1:
        if not os.path.exists(path_):
            os.makedirs(path_)

    with open(path_file, 'w') as out_file:
        for line in data:
            # write line to output file
            out_file.write(str(line))
            out_file.write("\n")
        out_file.close()


def select_commit_based_topwords(words, commits):
    new_commit = list()
    for c in commits:
        msg = c['msg'].split(',')
        for w in msg:
            if w in words:
                new_commit.append(c)
                break
    return new_commit


def load_info_label(data):
    data = list(data)
    pos = [1 for d in data if d == 1]
    neg = [0 for d in data if d == 0]
    return len(pos), len(neg)


def commits_index(commits):
    commits_index = [i for i, c in enumerate(commits) if c.startswith("commit:")]
    return commits_index


def commit_info(commit):
    id = commit_id(commit)
    stable = commit_stable(commit)
    date = commit_date(commit)
    msg = commit_msg(commit)
    code = commit_code(commit)
    return id, stable, date, msg, code


def extract_commit(path_file):
    # extract commit from july data
    commits = load_file(path_file=path_file)
    indexes = commits_index(commits=commits)
    dicts = list()
    for i in range(0, len(indexes)):
        dict = {}
        if i == len(indexes) - 1:
            id, stable, date, msg, code = commit_info(commits[indexes[i]:])
        else:
            id, stable, date, msg, code = commit_info(commits[indexes[i]:indexes[i + 1]])
        dict["id"] = id
        dict["stable"] = stable
        dict["date"] = date
        dict["msg"] = msg
        dict["code"] = code
        dicts.append(dict)
    return dicts


def commit_info_recent(commit):
    id = commit_id(commit)
    msg = commit_msg_recent(commit)
    code = commit_code_recent(commit)
    return id, msg, code


def commit_msg_recent(commit):
    commit_msg = commit[3].strip()  # extracting the simplified commit message
    return commit_msg


def commit_msg_recent_ver1(commit):
    commit_msg = commit[9].strip()
    return commit_msg


def commit_code_recent(commit):
    all_code = commit[6:]  # use for march data
    file_index = [i for i, c in enumerate(all_code) if c.startswith("file:")]
    dicts = list()
    for i in range(0, len(file_index)):
        dict_code = {}
        if i == len(file_index) - 1:
            added_code, removed_code = hunk_code(all_code[file_index[i]:])
        else:
            added_code, removed_code = hunk_code(all_code[file_index[i]:file_index[i + 1]])
        dict_code[i] = all_code[file_index[i]].split(":")[1].strip()
        dict_code["added"] = added_code
        dict_code["removed"] = removed_code
        dicts.append(dict_code)
    return dicts


def extract_commit_recent_sasha(path_file):
    # loading recent_data from Julia (data without labels)
    # the data is getting from sasha so we assume that the data has label stable: true
    commits = load_file(path_file=path_file)
    indexes = commits_index(commits=commits)
    dicts = list()
    for i in range(0, len(indexes)):
        dict = {}
        if i == len(indexes) - 1:
            id, msg, code = commit_info_recent(commits[indexes[i]:])
        else:
            id, msg, code = commit_info_recent(commits[indexes[i]:indexes[i + 1]])
        dict["id"] = id
        dict["stable"] = 'true'
        dict["msg"] = msg
        dict["code"] = code
        dicts.append(dict)
    return dicts


def commit_code_recent_keras(commit):
    all_code = commit[6:]  # use for recent commits
    file_index = [i for i, c in enumerate(all_code) if c.startswith("file:")]
    dicts = list()
    for i in range(0, len(file_index)):
        dict_code = {}
        if i == len(file_index) - 1:
            diff_code = all_code[file_index[i]:]
        else:
            diff_code = all_code[file_index[i]:file_index[i + 1]]
        dict_code["file"] = all_code[file_index[i]].split(":")[1].strip()
        dict_code["diff"] = diff_code[1:]
        dicts.append(dict_code)
    return dicts


def commit_code_recent_keras_ver1(commit):
    all_code = commit[12:]  # use for recent commits
    file_index = [i for i, c in enumerate(all_code) if c.startswith("file:")]
    dicts = list()
    for i in range(0, len(file_index)):
        dict_code = {}
        if i == len(file_index) - 1:
            diff_code = all_code[file_index[i]:]
        else:
            diff_code = all_code[file_index[i]:file_index[i + 1]]
        dict_code["file"] = all_code[file_index[i]].split(":")[1].strip()
        dict_code["diff"] = diff_code[1:]
        dicts.append(dict_code)
    return dicts


def commit_info_recent_keras(commit):
    id = commit_id(commit)
    msg = commit_msg_recent(commit)
    code = commit_code_recent_keras(commit)
    return id, msg, code


def commit_info_recent_keras_ver1(commit):
    id = commit_id(commit)
    stable = commit_stable(commit)
    msg = commit_msg_recent_ver1(commit)
    code = commit_code_recent_keras_ver1(commit)
    return id, stable, msg, code



def extract_commit_recent_sasha_keras(path_file):
    # loading recent_data from Julia (data without labels)
    # the data is getting from sasha so we assume that the data has label stable: true
    commits = load_file(path_file=path_file)
    indexes = commits_index(commits=commits)
    dicts = list()
    for i in range(0, len(indexes)):
        dict = {}
        if i == len(indexes) - 1:
            id, msg, code = commit_info_recent_keras(commits[indexes[i]:])
        else:
            id, msg, code = commit_info_recent_keras(commits[indexes[i]:indexes[i + 1]])
        dict["id"] = id
        dict["stable"] = 'true'
        dict["msg"] = msg
        dict["code"] = code
        dicts.append(dict)
    return dicts


def extract_commit_recent_sasha_keras_ver1(path_file):
    # used to extract commit from Julia's data: Feb 16
    commits = load_file(path_file=path_file)
    indexes = commits_index(commits=commits)
    dicts = list()
    for i in range(0, len(indexes)):
        dict = {}
        if i == len(indexes) - 1:
            id, stable, msg, code = commit_info_recent_keras_ver1(commits[indexes[i]:])
        else:
            id, stable, msg, code = commit_info_recent_keras_ver1(commits[indexes[i]:indexes[i + 1]])
        dict["id"] = id
        dict["stable"] = stable
        dict["msg"] = msg
        dict["code"] = code
        dicts.append(dict)
    return dicts


def reformat_commit_code_keras(commits, num_file):
    commits = reformat_file(commits=commits, num_file=num_file)
    return commits


def reformat_commit_code(commits, num_file, num_hunk, num_loc, num_leng):
    commits = reformat_file(commits=commits, num_file=num_file)
    commits = reformat_hunk(commits=commits, num_hunk=num_hunk, num_loc=num_loc, num_leng=num_leng)
    return commits


def shuffled_mini_batches(X_msg, X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X_msg = X_msg[permutation, :]
    shuffled_X_added = X_added_code[permutation, :, :, :]
    shuffled_X_removed = X_removed_code[permutation, :, :, :]
    if len(Y.shape) == 1:
        shuffled_Y = Y[permutation]
    else:
        shuffled_Y = Y[permutation, :]
    # shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        # mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_added = shuffled_X_added[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[num_complete_minibatches * mini_batch_size: m, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches(X_msg, X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg = X_msg
    shuffled_X_added = X_added_code
    shuffled_X_removed = X_removed_code
    shuffled_Y = Y

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_added = shuffled_X_added[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[num_complete_minibatches * mini_batch_size: m, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_topwords(X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0):
    m = Y.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_added = X_added_code
    shuffled_X_removed = X_removed_code
    shuffled_Y = Y

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_added = shuffled_X_added[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[num_complete_minibatches * mini_batch_size: m, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


if __name__ == "__main__":
    path_data = "./data/data_small.text"
    path_data = "./data/newres_funcalls_jul28.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nloc, nleng = 1, 8, 10, 120
    new_commits = reformat_commit_code(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nloc, num_leng=nleng)

    # total_ids = filtering_commit_union(commits=new_commits, num_file=nfile, num_hunk=nhunk, num_loc=nloc, size_line=nleng)
    # print len(total_ids)

from keras_ultis import extract_commit_recent_sasha_keras, extract_commit_recent_sasha_keras_ver1

if __name__ == '__main__':
    path_file = '../data/recent_functions_translated_nodups.out'
    commits_recent_data = extract_commit_recent_sasha_keras(path_file=path_file)


    path_file = '../data/recent_functions_translated.out'
    commits_new_recent_data = extract_commit_recent_sasha_keras_ver1(path_file=path_file)

    print(len(commits_recent_data), len(commits_new_recent_data))
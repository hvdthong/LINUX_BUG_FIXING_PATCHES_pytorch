from ultis import extract_commit_recent


if __name__ == '__main__':
    # creating training data
    path_file = './data/recent_sashas_functions_translated_nodups.out'
    commits = extract_commit_recent(path_file=path_file)
    print(len(commits))
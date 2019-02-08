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

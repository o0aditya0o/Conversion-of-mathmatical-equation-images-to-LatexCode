from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pickle


def read_data(data):
    ans = []
    for i in data:
        d = i.split(' ')
        new_list = []
        for j in d:
            if len(j) == 0:
                continue
            new_list.append(j)
        ans.append(new_list)
    ret = np.empty([len(ans)-1, len(ans[0])]) 
    i=0
    for l in ans:
        curr = np.array(l, dtype=float)
        if len(curr) == 0:
            continue
        ret[i] = curr
        i += 1
    return ret


def file_read(file):
    f = open(file, 'r')
    f = f.read()
    f = f.split('\n')
    f = read_data(f)
    return f

def run():
    x_train = file_read('final_train_input.txt')
    y_train = file_read('final_train_output.txt')
    x_test = file_read('final_test_input.txt')
    y_test = file_read('final_test_output.txt')

    #normalization of training and text data
    scaler = StandardScaler()
    scaler.fit(x_train)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    clf = MLPClassifier()
    clf.fit(x_train, y_train)

    with open('MLPClassifier.pkl','wb') as f:
        pickle.dump(clf, f)

    g = clf.score(x_test, y_test)
    print(g)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    with open('GuassinaNB.pkl','wb') as f:
        pickle.dump(gnb, f)

    gnbscore = gnb.score(x_test, y_test)
    print("GNB Score : " + str(gnbscore))

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)

    knnscore = neigh.score(x_test, y_test)
    print("KNN score : " + str(knnscore))

run()
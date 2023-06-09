import numpy as np
import csv
import codecs
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

def data_read_csv(file_name):
    file_csv = codecs.open(file_name, 'r', 'gbk')
    reader = csv.reader(file_csv)#,delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    return(reader)
def feature_read(reader,m,n):
    X = np.zeros((m,n))
    i = 0
    for x in reader:
        X[i] = x
        i = i+1
    return(X)


def tree_write (x_train,y_train):
    clf = RandomForestClassifier(max_depth=2)
    clf.fit(x_train,y_train)
    s = pickle.dumps(clf)
    f = open('tree.model', "wb+")
    f.write(s)
    f.close()
    print("Done")

def knn_write(x_train,y_train):
    clf = KNeighborsClassifier(n_neighbors=3) #k取值可以更改
    clf.fit(x_train,y_train)
    s = pickle.dumps(clf)
    f = open('knn.model', "wb+")
    f.write(s)
    f.close()
    print("Done")

def tree_read(x):
    f = open('tree.model', 'rb')
    s = f.read()
    model = pickle.loads(s)
    y_hat = model.predict(x)
    y_hat = y_hat.tolist()
    for i in range(len(y_hat), 11):
        y_hat.append('0')
    return(y_hat)
def knn_read(x):
    f = open('knn.model', 'rb')
    s = f.read()
    model = pickle.loads(s)
    y_hat = model.predict(x)
    y_hat = y_hat.tolist()
    for i in range(len(y_hat), 11):
        y_hat.append('0')
    return(y_hat)
def tree_mfcc_tdoa_read(x):
    f = open('treemfcctdoa.model', 'rb')
    s = f.read()
    model = pickle.loads(s)
    y_hat = model.predict(x)
    y_hat = y_hat.tolist()
    for i in range(len(y_hat),11):
        y_hat.append('0')
    return(y_hat)
def tree_tdoa_read(x):
    f = open('treetdoa.model', 'rb')
    s = f.read()
    model = pickle.loads(s)
    y_hat = model.predict(x)
    y_hat = y_hat.tolist()
    for i in range(len(y_hat), 11):
        y_hat.append('0')
    return(y_hat)
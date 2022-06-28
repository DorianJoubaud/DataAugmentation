import os
from os import listdir
import pandas as pd
from utils.input_data import read_data_sets
import utils.datasets as ds
from utils import *
import numpy as np
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

## import data
dataset = 'ECG5000'
train_data_file = os.path.join("../UCRArchive_2018", dataset, "%s_TRAIN.tsv"%dataset)
test_data_file = os.path.join("../UCRArchive_2018", dataset, "%s_TEST.tsv"%dataset)

x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter="\t")
y_train = ds.class_offset(y_train, dataset)
y_test = ds.class_offset(y_test, dataset)

    
    
nb_classes = ds.nb_classes(dataset)



## Data class distribution
rat = list()
def howmany(my_list, elt):
    tmp = 0
    for x in my_list:
        if (x == elt):
            tmp += 1
    return tmp
for j in range(nb_classes):
    rat.append(howmany(y_train, j))

majority_class = np.argmax(rat)

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
#all metrics
evolutionfm = list()
evolutionfw = list()
evolutiona = list()
evolutionmcc = list()
evolutionrec = list()
evolutionpres = list()
import warnings
from imblearn.over_sampling import RandomOverSampler 
warnings.filterwarnings("ignore")

# x_train, y_train train set
# x_test, y_test test set

#Class Distribution
#rat = [292, 177, 10, 19, 2]

#Oversample class 1 
for i in range(177,292):
    rat_str = {0:292,1:i,2:10,3:19,4:2}
    oversample = RandomOverSampler(rat_str)
    Xo, yo = oversample.fit_resample(x_train, y_train)
    m = TimeSeriesForestClassifier()
    m.fit(Xo, yo)
    y_pred = m.predict(x_test)
    fm = f1_score(y_test, y_pred, average = "macro")
    fw = f1_score(y_test, y_pred, average = "weighted")
    accu = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average=None)
    pres = precision_score(y_test, y_pred, average=None)
    evolutiona.append(accu)
    evolutionfm.append(fm)
    evolutionfw.append(fw)
    evolutionmcc.append(mcc)
    evolutionrec.append(rec)
    evolutionpres.append(pres)
    
print('Class 1 - Ok')    
#Oversample class 2
for j in range(10,292):
    rat_str = {0:292,1:i,2:j,3:19,4:2}
    oversample = RandomOverSampler(rat_str)
    Xo, yo = oversample.fit_resample(x_train, y_train)
    m = TimeSeriesForestClassifier()
    m.fit(Xo, yo)
    y_pred = m.predict(x_test)
    fm = f1_score(y_test, y_pred, average = "macro")
    fw = f1_score(y_test, y_pred, average = "weighted")
    accu = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average=None)
    pres = precision_score(y_test, y_pred, average=None)
    evolutiona.append(accu)
    evolutionfm.append(fm)
    evolutionfw.append(fw)
    evolutionmcc.append(mcc)
    evolutionrec.append(rec)
    evolutionpres.append(pres)


    
print('Class 2 - Ok')    
#Oversample class 3 
for k in range(19,292):
    rat_str = {0:292,1:i,2:j,3:k,4:2}
    oversample = RandomOverSampler(rat_str)
    Xo, yo = oversample.fit_resample(x_train, y_train)
    m = TimeSeriesForestClassifier()
    m.fit(Xo, yo)
    y_pred = m.predict(x_test)
    fm = f1_score(y_test, y_pred, average = "macro")
    fw = f1_score(y_test, y_pred, average = "weighted")
    accu = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average=None)
    pres = precision_score(y_test, y_pred, average=None)
    evolutiona.append(accu)
    evolutionfm.append(fm)
    evolutionfw.append(fw)
    evolutionmcc.append(mcc)
    evolutionrec.append(rec)
    evolutionpres.append(pres)

print('Class 3 - Ok')    
#Oversample class 4
for l in range(2,292):
    rat_str = {0:292,1:i,2:j,3:k,4:l}
    oversample = RandomOverSampler(rat_str)
    Xo, yo = oversample.fit_resample(x_train, y_train)
    m = TimeSeriesForestClassifier()
    m.fit(Xo, yo)
    y_pred = m.predict(x_test)
    fm = f1_score(y_test, y_pred, average = "macro")
    fw = f1_score(y_test, y_pred, average = "weighted")
    accu = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average=None)
    pres = precision_score(y_test, y_pred, average=None)
    evolutiona.append(accu)
    evolutionfm.append(fm)
    evolutionfw.append(fw)
    evolutionmcc.append(mcc)
    evolutionrec.append(rec)
    evolutionpres.append(pres)

    
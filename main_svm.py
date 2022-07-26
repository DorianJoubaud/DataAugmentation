import os
from os import listdir
import pandas as pd
from utils.input_data import read_data_sets
import utils.datasets as ds
from utils import *
from funct import *
import numpy as np
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from tensorflow.keras.utils import to_categorical
import warnings
from sklearn import svm
warnings.filterwarnings("ignore")

## import data
folders = listdir("UCRArchive_2018")
#folders = ['DiatomSizeReduction']

print(folders)
for idx in range(len(folders)):

    dataset = folders[idx]

    print(dataset)
    print(f'{idx}/{len(folders)}')

    #load data
    nb_classes = ds.nb_classes(dataset)
    nb_dims = ds.nb_dims(dataset)
    train_data_file = os.path.join("UCRArchive_2018/", dataset, "%s_TRAIN.tsv"%dataset)
    test_data_file = os.path.join("UCRArchive_2018/", dataset, "%s_TEST.tsv"%dataset)

    x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter="\t")

    y_train = ds.class_offset(y_train, dataset)
    y_test= ds.class_offset(y_test, dataset)
    nb_timesteps = int(x_train.shape[1] / nb_dims)
    input_shape = (nb_timesteps , nb_dims)

    x_train_max = np.max(x_train)
    x_train_min = np.min(x_train)
    #normalise in [-1;1]
    x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
    x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.

    nb_timesteps = int(x_train.shape[1] / nb_dims)
    input_shape = (nb_timesteps , nb_dims)
    #x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
    #x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))
    #y_test = to_categorical(ds.class_offset(y_test, dataset), nb_classes)
    #y_test_num = np.argmax(y_test)


    rat = list()

    for j in range(nb_classes):
        rat.append(howmany(y_train, j))
    majority_class = np.argmax(rat)

    done = rat
    m = svm.SVC()()
    # Oversampling methods
    #ROS, JITTER,TW,SMOTE,SMOTESVM

    Revolutionf = list() #f1 scores
    Revolutiong = list() #g means
    Revolutiona = list() #accuracy
    Revolutionmcc = list() #mcc
    Revolutionrec = list() #precision
    Revolutionpres = list() #recall

    Jevolutionf = list() #f1 scores
    Jevolutiong = list() #g means
    Jevolutiona = list() #accuracy
    Jevolutionmcc = list() #mcc
    Jevolutionrec = list() #precision
    Jevolutionpres = list() #recall

    Tevolutionf = list() #f1 scores
    Tevolutiong = list() #g means
    Tevolutiona = list() #accuracy
    Tevolutionmcc = list() #mcc
    Tevolutionrec = list() #precision
    Tevolutionpres = list() #recall

    Sevolutionf = list() #f1 scores
    Sevolutiong = list() #g means
    Sevolutiona = list() #accuracy
    Sevolutionmcc = list() #mcc
    Sevolutionrec = list() #precision
    Sevolutionpres = list() #recall

    SVevolutionf = list() #f1 scores
    SVevolutiong = list() #g means
    SVevolutiona = list() #accuracy
    SVevolutionmcc = list() #mcc
    SVevolutionrec = list() #precision
    SVevolutionpres = list() #recall
    # for all classes
    for i in range(nb_classes):
      #except the majority
      if i != majority_class:
        #add j samples to the class
        for j in range(rat[i], rat[majority_class]):
          os_strat = {t:done[t] for t in range(nb_classes)}
          os_strat[i] = j

          accu, mcc, f, rec, pres, g = jitterClassif(x_train, y_train, x_test, y_test, i, j,m)
          Jevolutionf.append(f) # f1 scores
          Jevolutiong.append(g) #g means
          Jevolutiona.append(accu) #accuracy
          Jevolutionmcc.append(mcc) #mcc
          Jevolutionrec.append(rec) #precision
          Jevolutionpres.append(pres) #recall

          accu, mcc, f, rec, pres, g = timeWClassif(x_train, y_train, x_test, y_test, i, j,m)
          Tevolutionf.append(f) # f1 scores
          Tevolutiong.append(g) #g means
          Tevolutiona.append(accu) #accuracy
          Tevolutionmcc.append(mcc) #mcc
          Tevolutionrec.append(rec) #precision
          Tevolutionpres.append(pres) #recall

          accu, mcc, f, rec, pres, g = rosClassif(x_train, y_train, x_test, y_test, os_strat,m)
          Revolutionf.append(f) # f1 scores
          Revolutiong.append(g) #g means
          Revolutiona.append(accu) #accuracy
          Revolutionmcc.append(mcc) #mcc
          Revolutionrec.append(rec) #precision
          Revolutionpres.append(pres) #recall

          accu, mcc, f, rec, pres, g = smoteClassif(x_train, y_train, x_test, y_test, os_strat,nb_classes,m)
          Sevolutionf.append(f) # f1 scores
          Sevolutiong.append(g) #g means
          Sevolutiona.append(accu) #accuracy
          Sevolutionmcc.append(mcc) #mcc
          Sevolutionrec.append(rec) #precision
          Sevolutionpres.append(pres) #recall

          accu, mcc, f, rec, pres, g = svmSmoteClassif(x_train, y_train, x_test, y_test, os_strat,nb_classes,m)
          SVevolutionf.append(f) # f1 scores
          SVevolutiong.append(g) #g means
          SVevolutiona.append(accu) #accuracy
          SVevolutionmcc.append(mcc) #mcc
          SVevolutionrec.append(rec) #precision
          SVevolutionpres.append(pres) #recall

        os.makedirs('ResultsSVM/'+ dataset + '/Jittering', exist_ok=True)
        os.makedirs('ResultsSVM/'+ dataset + '/TimeWarping', exist_ok=True)
        os.makedirs('ResultsSVM/'+ dataset + '/ROS', exist_ok=True)
        os.makedirs('ResultsSVM/'+ dataset + '/SMOTE', exist_ok=True)
        os.makedirs('ResultsSVM/'+ dataset + '/SMOTESVM', exist_ok=True)
        #Jittering
        pd.DataFrame(Jevolutionf).to_csv('ResultsSVM/'+ dataset + '/Jittering/' + 'f1_evolution.csv')
        pd.DataFrame(Jevolutiong).to_csv('ResultsSVM/'+ dataset + '/Jittering/' + 'g_evolution.csv')
        pd.DataFrame(Jevolutiona).to_csv('ResultsSVM/'+ dataset + '/Jittering/' + 'accu_evolution.csv')
        pd.DataFrame(Jevolutionmcc).to_csv('ResultsSVM/'+ dataset + '/Jittering/' + 'mcc_evolution.csv')
        pd.DataFrame(Jevolutionrec).to_csv('ResultsSVM/'+ dataset + '/Jittering/' + 'rec_evolution.csv')
        pd.DataFrame(Jevolutionpres).to_csv('ResultsSVM/'+ dataset + '/Jittering/' + 'pres_evolution.csv')
        #Time Warping
        pd.DataFrame(Tevolutionf).to_csv('ResultsSVM/'+ dataset + '/TimeWarping/' + 'f1_evolution.csv')
        pd.DataFrame(Tevolutiong).to_csv('ResultsSVM/'+ dataset + '/TimeWarping/' + 'g_evolution.csv')
        pd.DataFrame(Tevolutiona).to_csv('ResultsSVM/'+ dataset + '/TimeWarping/' + 'accu_evolution.csv')
        pd.DataFrame(Tevolutionmcc).to_csv('ResultsSVM/'+ dataset + '/TimeWarping/' + 'mcc_evolution.csv')
        pd.DataFrame(Tevolutionrec).to_csv('ResultsSVM/'+ dataset + '/TimeWarping/' + 'rec_evolution.csv')
        pd.DataFrame(Tevolutionpres).to_csv('ResultsSVM/'+ dataset + '/TimeWarping/' + 'pres_evolution.csv')
        #ROS
        pd.DataFrame(Revolutionf).to_csv('ResultsSVM/'+ dataset + '/ROS/' + 'f1_evolution.csv')
        pd.DataFrame(Revolutiong).to_csv('ResultsSVM/'+ dataset + '/ROS/' + 'g_evolution.csv')
        pd.DataFrame(Revolutiona).to_csv('ResultsSVM/'+ dataset + '/ROS/' + 'accu_evolution.csv')
        pd.DataFrame(Revolutionmcc).to_csv('ResultsSVM/'+ dataset + '/ROS/' + 'mcc_evolution.csv')
        pd.DataFrame(Revolutionrec).to_csv('ResultsSVM/'+ dataset + '/ROS/' + 'rec_evolution.csv')
        pd.DataFrame(Revolutionpres).to_csv('ResultsSVM/'+ dataset + '/ROS/' + 'pres_evolution.csv')
        #SMOTE

        pd.DataFrame(Sevolutionf).to_csv('ResultsSVM/'+ dataset + '/SMOTE/' + 'f1_evolution.csv')
        pd.DataFrame(Sevolutiong).to_csv('ResultsSVM/'+ dataset + '/SMOTE/' + 'g_evolution.csv')
        pd.DataFrame(Sevolutiona).to_csv('ResultsSVM/'+ dataset + '/SMOTE/' + 'accu_evolution.csv')
        pd.DataFrame(Sevolutionmcc).to_csv('ResultsSVM/'+ dataset + '/SMOTE/' + 'mcc_evolution.csv')
        pd.DataFrame(Sevolutionrec).to_csv('ResultsSVM/'+ dataset + '/SMOTE/' + 'rec_evolution.csv')
        pd.DataFrame(Sevolutionpres).to_csv('ResultsSVM/'+ dataset + '/SMOTE/' + 'pres_evolution.csv')
        #SVMSMOTE
        pd.DataFrame(SVevolutionf).to_csv('ResultsSVM/'+ dataset + '/SMOTESVM/' + 'f1_evolution.csv')
        pd.DataFrame(SVevolutiong).to_csv('ResultsSVM/'+ dataset + '/SMOTESVM/' + 'g_evolution.csv')
        pd.DataFrame(SVevolutiona).to_csv('ResultsSVM/'+ dataset + '/SMOTESVM/' + 'accu_evolution.csv')
        pd.DataFrame(SVevolutionmcc).to_csv('ResultsSVM/'+ dataset + '/SMOTESVM/' + 'mcc_evolution.csv')
        pd.DataFrame(SVevolutionrec).to_csv('ResultsSVM/'+ dataset + '/SMOTESVM/' + 'rec_evolution.csv')
        pd.DataFrame(SVevolutionpres).to_csv('ResultsSVM/'+ dataset + '/SMOTESVM/' + 'pres_evolution.csv')













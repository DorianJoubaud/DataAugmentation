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
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from scipy.interpolate import CubicSpline
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from imblearn.over_sampling import RandomOverSampler


def jitter(x, sigma=0.03):
        # https://arxiv.org/pdf/1706.00527.pdf
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def time_warp(x, sigma=0.2, knot=4):

        orig_steps = np.arange(x.shape[1])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2,
                                                                    x.shape[2]))
        warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
                scale = (x.shape[1]-1)/time_warp[-1]
                ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
        return ret

# Jittering
def jitterClassif(x_train, y_train, x_test, y_test, classe, nb):

  def Augmentation(function, data, label_data, class_under, nb):
      underReprClass = list()
      idxs = np.where((label_data == class_under))
      count = 0
      for i in range(nb):
          if (count >= nb):
            break
          underReprClass.append(function(data[idxs[0][i%len(idxs)]]))# difference of data shape with TW Augmentation
          count +=1
      return (np.array(underReprClass), np.array([class_under for i in range(nb)]))

  Xo, yo = Augmentation(jitter,x_train, y_train,classe ,nb)
  oversamp = np.concatenate((Xo,x_train), axis = 0)
  oversamp_labels = np.concatenate((yo,y_train), axis = 0)
  m = TimeSeriesForestClassifier()
  m.fit(oversamp, oversamp_labels)
  y_pred = m.predict(x_test)
  f = f1_score(y_test, y_pred, average = None)
  accu = accuracy_score(y_test, y_pred)
  mcc = matthews_corrcoef(y_test, y_pred)
  rec = recall_score(y_test, y_pred, average=None)
  pres = precision_score(y_test, y_pred, average=None)
  g = geometric_mean_score(y_test, y_pred, average=None)
  return accu, mcc, f, rec, pres, g
# Time Warping
def timeWClassif(x_train, y_train, x_test, y_test, classe, nb):

  def Augmentation(function, data, label_data, class_under, nb):
      underReprClass = list()
      idxs = np.where((label_data == class_under))[0]
      #print(idxs)
      count = 0
      for i in range(nb):
          if (count >= nb):
            break
          underReprClass.append(function(data)[idxs[i%len(idxs)]])# difference of data shape with Jitter Augmentation
          count +=1
      return (np.array(underReprClass), np.array([class_under for i in range(nb)]))

  Xo, yo = Augmentation(jitter,x_train, y_train,classe ,nb)
  oversamp = np.concatenate((Xo,x_train), axis = 0)
  oversamp_labels = np.concatenate((yo,y_train), axis = 0)
  m = TimeSeriesForestClassifier()
  m.fit(oversamp, oversamp_labels)
  y_pred = m.predict(x_test)
  f = f1_score(y_test, y_pred, average = None)
  accu = accuracy_score(y_test, y_pred)
  mcc = matthews_corrcoef(y_test, y_pred)
  rec = recall_score(y_test, y_pred, average=None)
  pres = precision_score(y_test, y_pred, average=None)
  g = geometric_mean_score(y_test, y_pred, average=None)
  return accu, mcc, f, rec, pres, g




# Random OverSampling


def rosClassif(x_train, y_train, x_test, y_test, os_strat):
  oversample = RandomOverSampler(os_strat)
  Xo, yo = oversample.fit_resample(x_train, y_train)
  m = TimeSeriesForestClassifier()
  m.fit(Xo, yo)
  y_pred = m.predict(x_test)
  f = f1_score(y_test, y_pred, average = None)
  accu = accuracy_score(y_test, y_pred)
  mcc = matthews_corrcoef(y_test, y_pred)
  rec = recall_score(y_test, y_pred, average=None)
  pres = precision_score(y_test, y_pred, average=None)
  g = geometric_mean_score(y_test, y_pred, average=None)
  return accu, mcc, f, rec, pres, g

# Smote
def smoteClassif(x_train, y_train, x_test, y_test, os_strat):

  oversample = SMOTE(os_strat, k_neighbors=1)
  try:
    Xo, yo = oversample.fit_resample(x_train, y_train) # if 1 element in one class => no neighbord => no smote
  except:
      return 0,0,0,0,0,0
  m = TimeSeriesForestClassifier()
  m.fit(Xo, yo)
  y_pred = m.predict(x_test)
  f = f1_score(y_test, y_pred, average = None)
  accu = accuracy_score(y_test, y_pred)
  mcc = matthews_corrcoef(y_test, y_pred)
  rec = recall_score(y_test, y_pred, average=None)
  pres = precision_score(y_test, y_pred, average=None)
  g = geometric_mean_score(y_test, y_pred, average=None)
  return accu, mcc, f, rec, pres, g

#SVM Smote

def svmSmoteClassif(x_train, y_train, x_test, y_test, os_strat):
  oversample = SVMSMOTE(os_strat, k_neighbors=1)
  try:
    Xo, yo = oversample.fit_resample(x_train, y_train)# if 1 element in one class => no neighbord => no smote
  except:
      return 0,0,0,0,0,0
  m = TimeSeriesForestClassifier()
  m.fit(Xo, yo)
  y_pred = m.predict(x_test)
  f = f1_score(y_test, y_pred, average = None)
  accu = accuracy_score(y_test, y_pred)
  mcc = matthews_corrcoef(y_test, y_pred)
  rec = recall_score(y_test, y_pred, average=None)
  pres = precision_score(y_test, y_pred, average=None)
  g = geometric_mean_score(y_test, y_pred, average=None)
  return accu, mcc, f, rec, pres, g


# Occurence of elt in my_list => use to get class distribution
def howmany(my_list, elt):
    tmp = 0
    for x in my_list:
        if (x == elt):
            tmp += 1
    return tmp
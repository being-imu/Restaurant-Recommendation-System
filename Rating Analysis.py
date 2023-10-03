# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 23:50:06 2021

@author: userpc
"""

import numpy as np
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors
import pandas as pd
#referring to https://stats.stackexchange.com/questions/215938/generate-synthetic-data-to-match-sample-data


df = pd.read_csv('Event Feedback.csv').drop(['Timestamp','User Name'],axis=1)
#df = df.iloc[:,:-1] # this gives me df, the final Dataframe which I would like to generate a larger dataset based on. This is the smaller Dataframe with 21000x102 dimensions.
print(df)

def SMOTE(T, N, k):
# """
# Returns (N/100) * n_minority_samples synthetic minority samples.
#
# Parameters
# ----------
# T : array-like, shape = [n_minority_samples, n_features]
#     Holds the minority samples
# N : percetange of new synthetic samples:
#     n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
# k : int. Number of nearest neighbours.
#
# Returns
# -------
# S : array, shape = [(N/100) * n_minority_samples, n_features]
# """
    n_minority_samples, n_features = T.shape

    if N < 100:
       #create synthetic samples only for a subset of T.
       #TODO: select random minortiy samples
       N = 100
       pass

    if (N % 100) != 0:
       raise ValueError("N must be < 100 or multiple of 100")

    N =int( N/100)
   
    n_synthetic_samples = N * n_minority_samples
    n_synthetic_samples = int(n_synthetic_samples)
    n_features = int(n_features)
    S = np.zeros(shape=(n_synthetic_samples, n_features))

    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)

    #Calculate synthetic samples
    for i in range(n_minority_samples):
       #nn = neigh.kneighbors(T[i], return_distance=False)
       nn = neigh.kneighbors(T[i].reshape(1, -1), return_distance=False)
       for n in range(N):
          nn_index = choice(nn[0])
          #NOTE: nn includes T[i], we don't want to select it
          while nn_index == i:
             nn_index = choice(nn[0])

          dif = T[nn_index] - T[i]
          gap = np.random.random()
          S[n + i * N,:] = T[i,:] + gap * dif[:]

    return S

df = df.to_numpy()
print(df)
new_data = SMOTE(df,200,28) # this is where I call the function and expect new_data to be generated with larger number of samples than original df.
print(new_data )
df = pd.DataFrame(new_data)
print(df)
df.to_csv('file1.csv')
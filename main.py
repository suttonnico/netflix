import numpy as np
import csv
import math
import time
#from bm import RBM
import matplotlib.pyplot as plt
from rbm import SRBM
import os
import scipy.io as sio

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def bin_2_score(bin):
    r = 0
    if bin[0] == 1:
        r = 1
    else:
        if bin[1] == 1:
            r = 2
        else:
            if bin[2] == 1:
                r = 3
            else:
                if bin[3] == 1:
                    r = 4
                else:
                    if bin[4] ==1:
                        r = 5
                    else:
                        r = 0
    return r


def generate_test_data(data_user, data_movie, data_bscore, M, test_cases):
    test_data_user = np.zeros(test_cases)
    test_data_movie = np.zeros(test_cases)
    test_data_bscore = np.zeros([test_cases, 5])
    for i in range(0, test_cases):
        ind = np.floor(np.random.rand()*M)
        test_data_user[i] = data_user[int(ind)]
        test_data_movie[i] = data_movie[int(ind)]
        test_data_bscore[i][:] = data_bscore[int(ind)][:]
        data_bscore[int(ind)][:] = [0, 0, 0, 0, 0]
    return data_bscore, test_data_bscore, test_data_user, test_data_movie


def get_user(data_user, data_movie, data_bscore, M, user):
    N = len(data_user)
    L = 0
    #for i in range(0,N):
    #    if data_user[i] == user:
    #        L += 1
    V = np.zeros([int(M),5])
    for i in range(0, int(M)):
        if data_user[i] == user:
            V[int(data_movie[i]), :] = data_bscore[i, :]

    return V


def test_user(data_user,data_movie,data_bscore,M,user,brn):
    V = get_user(data_user,data_movie,data_bscore,M,user)
    RSE = 0
    for i in range(0,int(M)):
        if data_user[i] == user:
            bscore = brn.predict(V, data_movie[i])
            pscore = bin_2_score(bscore)
            dscore = bin_2_score(data_bscore[i][:])
            print 'usuario',user,'pelicula',data_movie[i],'predicho =',pscore,'realidad',dscore,'diferencia=',(math.pow((pscore - dscore), 2))
            RSE += (math.pow((pscore - dscore), 2))
    print 'RES usuario',user,'============== ',RSE


def load_data():
    data_movie = sio.loadmat('data_movie.mat')
    data_movie = data_movie['data_movie']
    data_movie = data_movie[0]

    data_user = sio.loadmat('data_user.mat')
    data_user = data_user['data_user']
    data_user = data_user[0]

    data_bscore = sio.loadmat('data_bscore.mat')
    data_bscore = data_bscore['data_bscore']
    #data_bscore = data_bscore[0]

    test_data_bscore = sio.loadmat('test_data_bscore.mat')
    test_data_bscore = test_data_bscore['test_data_bscore']
    #test_data_bscore = test_data_bscore[0]

    test_data_user = sio.loadmat('test_data_user.mat')
    test_data_user = test_data_user['test_data_user']
    test_data_user = test_data_user[0]

    test_data_movie = sio.loadmat('test_data_movie.mat')
    test_data_movie = test_data_movie['test_data_movie']
    test_data_movie = test_data_movie[0]
    return data_movie, data_user, data_bscore, test_data_bscore, test_data_user, test_data_movie

[data_movie, data_user, data_bscore, test_data_bscore, test_data_user, test_data_movie] = load_data()
#Cantidad de peliculas
M = np.max(data_movie)      #hay 32 peliculas que nadie califico pero no creo que moleste

#cantidad de usuarios
N = np.max(data_user)
V = get_user(data_user, data_movie, data_bscore, M, 1)
test_cases = 100
RSE= 0
for i in range(0,test_cases):
    dscore = bin_2_score(test_data_bscore[i][:])
    RSE += (math.pow((3 - dscore), 2)) / test_cases

print 'RSE for guessing 3 on all movies', RSE

#train for all
print 'Number of users', N


brnt = SRBM(M, 5, 50, 0.001)



V1 = get_user(data_user, data_movie, data_bscore, M, 1)
a = time.time()
brnt.train(V1, 1)
b = time.time()
time_train = b-a

print 'RSE for first user', brnt.predict_user(V1, M)
print 'Prediction of movie 1 for user 1', brnt.predict(V1,1,M)
brn = SRBM(M, 5, 50, 0.0001)
brn.train_bias(data_movie,data_bscore)
print 'biases check'
RSE = 0
for i in range(0,test_cases):
    print 'Guessing average', i, '%done'
    V = get_user(data_user, data_movie, data_bscore, M, test_data_user[i])
    # brn.train(V,1)
    pscore = brn.predict(V, test_data_movie[i], M)
    #pscore = bin_2_score(bscore)
    dscore = bin_2_score(test_data_bscore[i][:])
    RSE += (math.pow((pscore - dscore), 2)) / test_cases

print 'RSE of guessing average for movie', RSE

PRSE = RSE
ARSE = RSE
iteration = 0
while PRSE >= ARSE:
    PRSE = ARSE
    print 'Training'
    for i in range(2, int(N)):
        #print '%.1f' % (i/N*100),'% done', 'Expexted time = ',np.floor(time_train*(N-i)/60),'mins ','%.0f' % ((time_train*(N-i))-np.floor(time_train*(N-i)/60)*60),'secs'
        V = get_user(data_user, data_movie, data_bscore, M, i)
        brn.train(V, 1)

    RSE = 0
    print 'Guessing'
    for i in range(0,test_cases):
        V = get_user(data_user, data_movie, data_bscore, M, test_data_user[i])
        # brn.train(V,1)
        pscore = brn.predict(V, test_data_movie[i], M)
        #pscore = bin_2_score(bscore)
        dscore = bin_2_score(test_data_bscore[i][:])
        RSE += (math.pow((pscore - dscore), 2)) / test_cases

    print 'iteration',iteration,'Atual RSE ===',RSE
    ARSE = RSE
    iteration += 1
brn.save_weights()

import numpy as np
import csv
import math
from bm import RBM
import matplotlib.pyplot as plt
from rbm import SRBM
import os

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
                    r = 5
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


i = 0
data_user = np.zeros(80000)
data_movie = np.zeros(80000)
data_score = np.zeros(80000)
data_time = np.zeros(80000)
data = np.zeros([80000, 3])
with open('data_set.csv', 'rb') as csvfile:
    data_set = csv.reader(csvfile, delimiter=',')
    for row in data_set:
        data_user[i] = row[0]
        data_movie[i] = row[1]
        data_score[i] = row[2]
        data_time[i] = row[3]
        data[i, 0] = row[0]
        data[i, 1] = row[1]
        data[i, 2] = row[3]
        i += 1


#paso los scores a una matriz de 5 columnas
data_bscore = np.zeros([80000, 5])

for i in range(0, 80000):
    if data_score[i] == 1:
        data_bscore[i, :] = [1, 0, 0, 0, 0]
    else:
        if data_score[i] == 2:
            data_bscore[i, :] = [0, 1, 0, 0, 0]
        else:
            if data_score[i] == 3:
                data_bscore[i, :] = [0, 0, 1, 0, 0]
            else:
                if data_score[i] == 4:
                    data_bscore[i, :] = [0, 0, 0, 1, 0]
                else:
                    data_bscore[i, :] = [0, 0, 0, 0, 1]

#Cantidad de peliculas
M = np.max(data_movie)      #hay 32 peliculas que nadie califico pero no creo que moleste

#cantidad de usuarios
N = np.max(data_user)
V = get_user(data_user, data_movie, data_bscore, M, 1)
[data_bscore, test_data_bscore, test_data_user, test_data_movie] = generate_test_data(data_user, data_movie, data_bscore, M, 100)
RSE= 0
for i in range(0,100):
    dscore = bin_2_score(test_data_bscore[i][:])
    RSE += (math.pow((3 - dscore), 2)) / 100

print RSE

#train for all
print N
brn = SRBM(M, 5, 100, 0.05)
#brn.train_bias(data_movie,data_bscore)test_user(data_user,data_movie,data_bscore, M, 2,brn)
#print 'biases check'
for i in range(1, int(N)):
    print i/N*100,'%done'
    V = get_user(data_user, data_movie, data_bscore, M, i)
    brn.train(V, 1)


#brn.train(V, 1000)
#print brn.predict(V,1)
#exit(123)

#RSE
RSE = 0
for i in range(0,100):
    print i, '%done'
    V = get_user(data_user, data_movie, data_bscore, M, test_data_user[i])
    bscore = brn.predict(V, test_data_movie[i])
    pscore = bin_2_score(bscore)
    dscore = bin_2_score(test_data_bscore[i][:])
    RSE += (math.pow((pscore - dscore), 2)) / 100

print RSE


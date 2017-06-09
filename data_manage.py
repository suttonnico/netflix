import numpy as np
import scipy.io as sio
import csv

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


test_cases = 100
M = np.max(data_movie)
[data_bscore, test_data_bscore, test_data_user, test_data_movie] = generate_test_data(data_user, data_movie, data_bscore, M, test_cases)

sio.savemat('data_user.mat', {'data_user': data_user})
sio.savemat('data_movie.mat', {'data_movie': data_movie})
sio.savemat('data_bscore.mat', {'data_bscore': data_bscore})
sio.savemat('test_data_bscore.mat', {'test_data_bscore': test_data_bscore})
sio.savemat('test_data_user.mat', {'test_data_user': test_data_user})
sio.savemat('test_data_movie.mat', {'test_data_movie': test_data_movie})

import numpy as np
#from timeit import default_number as timer
#from numbapro import vectorize




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


#@vectorize(["float32(int32,int32,float32,int32,bool,float32)"])
def get_one_hid_p(m, K, W, j, V, visbiases):
    sum1 = 0
    for i in range(0, m):
        for k in range(0, K):
            sum1 += V[i][k] * W[i][k][j]
    x = visbiases[j] + sum1
    return logistic(x)


def logistic(x):
       return 1.0 / (1 + np.exp(-x))

class SRBM:
    #W [cantidad de peliculas, cantidad de puntajes, cantidad de features]
    #V [pelicula, score]
    def __init__(self, movies_seen, posible_scores, num_hidden, learning_rate=0.1):
        self.F = int(num_hidden)
        self.m = int(movies_seen)
        self.learning_rate = learning_rate
        self.K = int(posible_scores)

        self.W = np.zeros([self.m, self.K, num_hidden])
        self.hidbiases = np.zeros([self.m, self.K])
        self.visbiases = np.zeros(num_hidden)

        self.W = 0.02 * np.random.rand(self.m, self.K, num_hidden ) - 0.01


    def get_hid_p(self,V):
        h = np.zeros(self.F)
        for j in range(0,self.F):
            h[j] = get_one_hid_p(self.m, self.K, self.W, j, V, self.visbiases)
        return h

    def get_vis_p(self,h):
        V = np.zeros([self.m, self.K])
        for i in range(0,self.m):
            for k in range(0,self.K):
                V[i][k] = self.get_one_vis_p(i,k,h)
        return V


    def get_one_vis_p(self, i, k, h):
        num = np.exp(self.hidbiases[i][k] + np.dot(h, self.W[i][k][:]))
        den = 0
        for l in range(0, self.K):
            den += np.exp(self.hidbiases[i][l] + np.dot(h, self.W[i][l][:]))
        return num / den

    def train_bias(self,data_movie,data_bscore):
        N = len(data_movie)
        for j in range(1,int(np.max(data_movie))):
            cant = 0
            for i in range(0,N):
                if data_movie[i] == j:
                    cant += 1
                    self.hidbiases[j][:] += data_bscore[i][:]
            self.hidbiases[j][:] /= cant
            for i in range(0,5):
                self.hidbiases[j][i] = np.log(self.hidbiases[j][i]/(1-self.hidbiases[j][i]))
            #print self.hidbiases[j][:]


    def train(self, data, epochs):
        for epoch in range(0,epochs):
            #print epoch
            pos_hid_p = self.get_hid_p(data)
            pos_hid_states = pos_hid_p > np.random.rand(self.F)
            pos_asso = np.zeros([self.m, self.K, self.F])
            for i in range(0, self.m):
                if np.any(data[i][:]) == 1:
                    for j in range(0,self.F):
                        for k in range(0, self.K):
                            pos_asso[i][k][j] = data[i][k]*pos_hid_p[j]
            neg_asso = np.zeros([self.m, self.K, self.F])
            neg_vis_p = self.get_vis_p(pos_hid_states)
            neg_hid_p = self.get_hid_p(neg_vis_p)
            for i in range(0,self.m):
                if np.any(data[i][:]) == 1:
                    for j in range(0,self.F):
                        for k in range(0, self.K):
                            neg_asso[i][k][j] = neg_vis_p[i][k]*neg_hid_p[j]
            self.W += self.learning_rate*(pos_asso-neg_asso)


    def predict(self, V, q):
        pos_hid_p = self.get_hid_p(V)
        pos_hid_states = pos_hid_p > np.random.rand(self.F)
        neg_vis_p = self.get_vis_p(pos_hid_states)
        movie = neg_vis_p[int(q)][:]
        th = np.max(movie)
        score = movie >= th
        return score

    def predict_user(self, V, M):
        pos_hid_p = self.get_hid_p(V)
        pos_hid_states = pos_hid_p > np.random.rand(self.F)
        neg_vis_p = self.get_vis_p(pos_hid_states)
        movie = neg_vis_p
        score = movie
        for i in range(1, int(M)):
            th = np.max(movie[i][:])
            score[i][:] = movie[i][:] >= th
        RSE = 0
        count = 0
        for i in range(1, int(M)):
            sd = bin_2_score(V[i][:])
            if sd != 0:
                count +=1
                RSE += (sd - bin_2_score(score[i][:]))**2
        RSEN = RSE / count
        return RSE

    def _logistic(self, x):
       return 1.0 / (1 + np.exp(-x))
from pyLocalizedLasso import LocalizedLasso
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import scipy.io as sio


if __name__ == '__main__':

    dataset = 1


    if dataset == 1:
        data = sio.loadmat('toyData.mat')

        X = data['X']
        Y = data['Y']
        R = data['W']

        R = R + R.transpose()

        [d,n] = X.shape

        model = LocalizedLasso(num_iter=50,lam_net=5,lam_exc=0.01,biasflag=1)
        #model = LocalizedLasso(num_iter=20, lam_net=5.0, lam_exc=0.01, biasflag=0)
        model.fit_regression(X,Y,R)

        #Prediction using weber

        rmse = 0
        for ii in range(0,n):
            #yte: predicted score
            #wte: estimated model parameter

            yte,wte = model.prediction(X[:,ii], R[:,ii])

            rmse += (Y[0][ii] - yte)**2

        print np.sqrt(rmse/n)

    else:
        data = sio.loadmat('toyData_cluster.mat')

        X = data['X']
        R = data['R']

        #In clustering, biasflag
        model = LocalizedLasso(num_iter=50, lam_net=5.0, lam_exc=0.1)

        model.fit_clustering(X, R)


    imgplot = plt.imshow(model.W)

    plt.show()
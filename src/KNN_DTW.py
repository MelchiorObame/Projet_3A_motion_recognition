import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

import distance_measure as dm

plt.style.use('bmh')
    
#on ne fait pas le chargement des datas ici. mais  à partir du fichier csv

class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays
    
    Arguments :----------------------------------
        n_neighbors : int, optional (default = 5)
                Number of neighbors to use by default for KNN

        max_warping_window : int, optional (default = infinity) 
                Maximum warping window allowed by the DTW dynamic
                programming function     
                En diagonale. si ==1 alors on prends que ceux de la bande de l'element actuel.
    """
    
    def __init__(self, n_neighbors=5, max_warping_window=1000):
        self.n_neighbors=n_neighbors
        self.max_warping_window=max_warping_window
        
    
    
        
    def fit(self, x_train, y_train):
        """Fit the model with x_train and their labels y_train
        x_train array of action , """
        self.x_train = x_train
        self.y_train = y_train
        
        
        
    def _dtw_distance(self, a1, a2, THO=0.0, normalized=False, confidence=False):  #OK
        """ Returns the DTW similarity distance between 2 timeseries sequences of action
        Arguments:---------
            a1 and a2: action 1 & 2 are array [Nframes,20,4] where Nframes is the number of frames 
            normalized: the parameter normalized is used in case these two
                        action have not same size  """
        #transformer les listes en array dans action
        action1= np.array(a1)
        action2= np.array(a2)
        #lenghts of action : number of frames
        len1= len(action1)
        len2= len(action2)
        costMatrix =  sys.maxsize * np.ones((len1, len2))
        #on initialise la matrice des distances/couts de taille de len(a1)*len(a2) avec des element infinity car on devra prendre le plus petit
        costMatrix[0,0]= dm.distance_frames(action1[0], action2[0],THO, confidence)
        #remplissage de la premiere ligne et de la premiere colonne
        #column
        for j in range(1,len1):
            costMatrix[j, 0]= costMatrix[j-1, 0] + dm.distance_frames(action1[j], action2[0], THO, confidence)
        #line
        for i in range(1,len2):
            costMatrix[0,i]= costMatrix[0, i-1] + dm.distance_frames(action1[0], action2[i], THO, confidence)        
        #on calcul la matrice de distance entre les differentes frames en se servant de la distance calculée dans
        for i in range(1, len1):
            for j in range( max(1,i- self.max_warping_window), min(len2, i+self.max_warping_window)):
                choices=costMatrix[i-1, j-1], costMatrix[i, j-1], costMatrix[i-1, j]
                costMatrix[i, j]= min(choices)+ dm.distance_frames(action1[i], action2[j], THO, confidence)
        #normalisation *1/(len1*len2)
        if not normalized:
            return costMatrix[-1, -1]
        else:
            return costMatrix[-1, -1]/(len1*len2)
            
        
    def _dist_matrix(self,training_dataSet,a2, THO=0.0, normalized=False, confidence=False): #OK
        """Computes the M  distance array between the training DataSet and testing
        action y using the DTW distance measure
        Arguments :-----------
            y :arrays of shape [NFrames, 20, 4]
            training_dataSet : arrays of shape of action [Nactions, Nframes, 20, 4]
        
        """
        training_dataSet=np.array(training_dataSet)
        action=np.array(a2)
        #compute the distance matrix
        dm = np.zeros(np.shape(training_dataSet)[0])
        len_trainindBDD= len(training_dataSet)
        for i in range(len_trainindBDD):
                dm[i]= self._dtw_distance(action, training_dataSet[i], THO, normalized, confidence)
        return dm
                   
        
    def predict(self, x, THO=0.0, normalized=False, confidence=False): #OK
        """ Predict the class labels or probability estimates for the provided data 
        Arguments:  --------------
            x  array ::  action who will be classified of shape [Nframes,20,4] 
        Returns:  2 arrays representing  --------------
           ( 1) the predicted class labels
           ( 2) the Knn label count probability     
        """
        
        dm = self._dist_matrix(self.x_train, x,THO, normalized, confidence)
        # Identify the K nearest neighbors
        knn_idx = dm.argsort()[:self.n_neighbors]
        # Identify the K nearest labels
        knn_labels = self.y_train[knn_idx]
        print("labels voisins :"+str(knn_labels))
        # Model label
        mode_data = mode(knn_labels, axis=0)
        mode_label = mode_data[0]
        #mode_proba = mode_data[1]/self.n_neighbors
        return mode_label.ravel()#, mode_proba.ravel()
    
        
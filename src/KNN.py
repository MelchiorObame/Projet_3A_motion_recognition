import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
plt.style.use('ggplot')

import os
import DataLoader

#print(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.ActionCSV.fileName))
df = pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.ActionCSV.fileName))
print(df.head())
print(df.shape)

#create numpy array for features X and target Y
X= df.drop('subject', axis=1).values
Y= df['subject'].values

def reshape(df):
    for line in df.itertuples():
        print(line)
#reshape(df)
        

#X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42, stratify=Y)

#creating Classifier KNN

#Setup arrays to store training and test accuracies
#neighbors = np.arange(1,9)
#train_accuracy =np.empty(len(neighbors))
#test_accuracy = np.empty(len(neighbors))
#
#for i,k in enumerate(neighbors):
#    #Setup a knn classifier with k neighbors
#    knn = KNeighborsClassifier(n_neighbors=k)
#    
#    #Fit the model
#    knn.fit(X_train, y_train)
#    
#    #Compute accuracy on the training set
#    train_accuracy[i] = knn.score(X_train, y_train)
#    
#    #Compute accuracy on the test set
#    test_accuracy[i] = knn.score(X_test, y_test) 
#    
#plt.title('k-NN Varying number of neighbors')
#plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
#plt.plot(neighbors, train_accuracy, label='Training accuracy')
#plt.legend()
#plt.xlabel('Number of neighbors')
#plt.ylabel('Accuracy')
#plt.show()
import numpy as np

import reshapeDataToMatrix
import visualisation
import DataLoader
import KNN_DTW


#on travail avec 1 comme nombre de voisins
#model= KNN_DTW.KnnDtw(n_neighbors=1, max_warping_window= sys.maxsize)
model= KNN_DTW.KnnDtw(n_neighbors=2, max_warping_window= 5)
X_train, actionLabelsTrain, subjectLabelsTrain, X_test, actionLabelsTest, subjectLabelsTest = reshapeDataToMatrix.train_test_split(DataLoader.DataName.action)

model.fit(np.array(X_train),np.array(actionLabelsTrain))
#model.fit(np.array(X_train),np.array(subjectLabelsTrain))
print("fit OK")

#prediction of other actions
print("Prediction...")

labelsPredicted=[]
GoodPrediction=0
Max= len(X_test)
for i in range(Max):
    print(str(i)+"/"+str(Max))
    action = X_test[i]
    label = model.predict(action, THO=0.2, normalized=False, confidence=False)
    
    labelsPredicted.append(label)
    if(label == actionLabelsTest[i]):
    #if(label == subjectLabelsTest[i]):
        GoodPrediction+=1
        print("Good prediction")
    else:
        print("Bad prediction")
print("percentage of recognition : " +str((GoodPrediction*100)/Max ))

matrix=visualisation.confusionMatrix(subjectLabelsTest, labelsPredicted)
        
      
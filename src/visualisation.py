from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import time

import KNN_DTW
import reshapeDataToMatrix
import DataLoader


def confusionMatrix(labelTruth, labelPredicted, labelTarget,dataName, percentage=0):
    print('----------  Visualisation  ------------')
    if dataName == DataLoader.DataName.action and labelTarget.lower()=="action": #ok
        labels=DataLoader.ActionCSV.actions    
    elif dataName == DataLoader.DataName.action and labelTarget.lower()=="subject": #ok, mais resultat pas ok
        labels = {}
        for i in range(1,DataLoader.ActionCSV.NumberSubject+1):
            labels[i] =str(i)
    elif dataName == DataLoader.DataName.dance and labelTarget.lower()=="action": #ok : for all data
        #inversion key, value items ex 'toto':2    to 2:toto
        labels= {v: k for k, v in DataLoader.DanceCSV.dance.items()}
    elif dataName == DataLoader.DataName.dance and labelTarget.lower()=="subject": #ok
        labels= {v: k for k, v in DataLoader.DanceCSV.subjects.items()} 
    elif dataName == DataLoader.DataName.emotional and labelTarget.lower()=="action":#ok
        labels= {v: k for k, v in DataLoader.EmotionalCSV.emotions.items()} 
    elif dataName == DataLoader.DataName.emotional and labelTarget.lower()=="subject":#ok
        labels= {v: k for k, v in DataLoader.EmotionalCSV.subjects.items()} 
    print(labels)
    print('Report')
    print(len(labelTruth))
    print(len(np.unique(labelTruth)))
    print(len(np.unique(labelPredicted)))
    print(labels.values())
    print(classification_report(labelTruth,labelPredicted , target_names=[l for l in labels.values()] ))
    conf_mat = confusion_matrix(labelTruth, labelPredicted)
    fig = plt.figure(figsize=(9,9))
    print('Figure')
    res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c>0:
                plt.text(j-.2, i+.1, c, fontsize=16)
    fig.colorbar(res)
    plt.title('Confusion Matrix : '+dataName+'DB '+str(labelTarget.lower())+'  '+str(percentage)+' %' )
    _ = plt.xticks(range(len(labels)), [l for l in labels.values()], rotation=70)
    _ = plt.yticks(range(len(labels)), [l for l in labels.values()])
    plt.ylabel('Predicted view')
    plt.xlabel('view')
    #save fig
    name='ConfMat_'+dataName+'_'+str(labelTarget.lower())+'_percent'+str(percentage).replace('.', '_')
    if not os.path.exists(DataLoader.DirPaths.savedPicturesConfusionMatrix):
        os.makedirs(DataLoader.DirPaths.savedPicturesConfusionMatrix) 
    plt.savefig(os.path.join(DataLoader.DirPaths.savedPicturesConfusionMatrix, name))

    
def loadMatrixFromFile(dataName,labelstarget ): #loads data Saved from KNN running
    if(dataName == DataLoader.DataName.action and labelstarget.lower()== "action"):
        with open(os.path.join( DataLoader.DirPaths.Matrix_ActionPredicted, DataLoader.DataMatrixLogFileName.MsrAcion3D_action), "rb") as f:
            dico = pickle.load(f)
        labelTruth=dico['actionLabelsTest']
    elif(dataName == DataLoader.DataName.action and labelstarget.lower()== "subject"):
        with open(os.path.join(DataLoader.DirPaths.Matrix_ActionPredicted,DataLoader.DataMatrixLogFileName.MsrAcion3D_subject), "rb") as f:
            dico = pickle.load(f)
        labelTruth=dico['subjectLabelsTest']
    elif(dataName == DataLoader.DataName.dance and labelstarget.lower()== "action"):
        with open(os.path.join(DataLoader.DirPaths.Matrix_ActionPredicted,DataLoader.DataMatrixLogFileName.DanceDB_action), "rb") as f:
            dico = pickle.load(f)
        labelTruth=dico['actionLabelsTest']
    elif(dataName == DataLoader.DataName.dance and labelstarget.lower()== "subject"):
        with open(os.path.join(DataLoader.DirPaths.Matrix_ActionPredicted,DataLoader.DataMatrixLogFileName.DanceDB_subject), "rb") as f:
            dico = pickle.load(f)
        labelTruth=dico['subjectLabelsTest']
    elif(dataName == DataLoader.DataName.emotional and labelstarget.lower()== "action"):
        with open(os.path.join(DataLoader.DirPaths.Matrix_ActionPredicted,DataLoader.DataMatrixLogFileName.EmotionalDB_action), "rb") as f:
            dico = pickle.load(f)
        labelTruth=dico['actionLabelsTest']
    elif(dataName == DataLoader.DataName.emotional and labelstarget.lower()== "subject"):
        with open(os.path.join(DataLoader.DirPaths.Matrix_ActionPredicted,DataLoader.DataMatrixLogFileName.EmotionalDB_subject), "rb") as f:
            dico = pickle.load(f)  
        labelTruth=dico['subjectLabelsTest']
    labelPredicted=dico['labelsPredicted']
    return labelTruth, list(labelPredicted), labelstarget

def window_Warping_Variation(dataName,labelstarget,index=0, n_neighbors=1, limitDataTrain=-1, THO=0.2, normalized=False, confidence=False): #ok
    """ Performance improvements can be achieved by reducing the max_warping_window parameter. However, these gains will not be sufficient to make KNN & DTW a viable
    classification technique for large or medium sized datasets.
    Arguments : index : position of action wwe want to predict 
                limitDataTrain : to limit Train data set""" 
    time_taken = []
    #windows = [0,1,2,5,10,50,100,500,1000]
    windows = [0,1,2]
    probas=[]
    X_train, actionLabelsTrain, subjectLabelsTrain, X_test, actionLabelsTest, subjectLabelsTest = reshapeDataToMatrix.train_test_split(dataName)
    for w in windows:
        begin = time.time()
        model= KNN_DTW.KnnDtw(n_neighbors, max_warping_window=w)
        if limitDataTrain !=-1:
                X_train=X_train[:limitDataTrain]
                actionLabelsTrain= actionLabelsTrain[:limitDataTrain]
                subjectLabelsTrain = subjectLabelsTrain[:limitDataTrain] 
        if(labelstarget.lower() =="action"):
            model.fit(np.array(X_train),np.array(actionLabelsTrain))
        elif(labelstarget.lower()=="subject"):
            model.fit(np.array(X_train),np.array(subjectLabelsTrain))
        else:
            print("Unknown labels Target")
            return 0
        label , proba = model.predict(X_test[index], dataName, THO, normalized, confidence)
        end = time.time()
        probas.append(proba)
        time_taken.append(end - begin)
    plt.figure(figsize=(12,5))
    _ = plt.plot(windows, [i/400. for i in time_taken], lw=4)
    plt.title('DTW Execution Time with \nvarying Max Warping Window :'+str(dataName)+' '+labelstarget+' for '+labelstarget+'['+str(index)+'], n_neighbors ='+str(n_neighbors))
    plt.ylabel('Execution Time (seconds)')
    plt.xlabel('Max Warping Window')
    plt.xscale('log')
    if not os.path.exists(DataLoader.DirPaths.savedPicturesWindows_Warping):
        os.makedirs(DataLoader.DirPaths.savedPicturesWindows_Warping) 
    name1= 'DTW_execTime'+dataName+'_'+labelstarget+'_neighbors_'+str(n_neighbors)
    plt.savefig(os.path.join(DataLoader.DirPaths.savedPicturesWindows_Warping, name1))
    plt.figure(figsize=(12,5))
    _ = plt.plot(windows, [i for i in probas])
    plt.title('Prediction : '+str(dataName)+' '+labelstarget+' for '+labelstarget+'['+str(index)+'], n_neighbors ='+str(n_neighbors))
    plt.ylabel('Recognition percentage ')
    plt.xlabel('Max Warping Window')
    plt.xscale('log')
    name2= 'Prediction_'+dataName+'_'+labelstarget+'_neighbors_'+str(n_neighbors)
    plt.savefig(os.path.join(DataLoader.DirPaths.savedPicturesWindows_Warping, name2))
   

#window_Warping_VariationMSRACTION = window_Warping_Variation( DataLoader.DataName.action,"action",index=1, n_neighbors=4, limit=33, THO=0.2, normalized=False, confidence=False)
#window_Warping_VariationMSRACTION = window_Warping_Variation( DataLoader.DataName.dance,"subject",index=1, n_neighbors=4, limit=-1, THO=0.2, normalized=False, confidence=False)
#window_Warping_VariationDance = window_Warping_Variation( DataLoader.DataName.dance,"action",limitDataTrain=5, index=1, n_neighbors=2, THO=0.2, normalized=False, confidence=False)


#print('Affichage Matrix MSR action 3D')
#labelTruth, labelPredicted, labelTarget = loadMatrixFromFile(DataLoader.DataName.dance, "action")
#confusionMatrix(labelTruth, labelPredicted, labelTarget,DataLoader.DataName.dance, percentage=8.33)
#print( len(np.unique(labelTruth)) == len(np.unique(labelPredicted)))

#print('Affichage Matrix Dance')
#labelTruth, labelPredicted, labelTarget = loadMatrixFromFile(DataLoader.DataName.dance, "action")
#print(labelTruth)
#print(np.unique(labelPredicted))
#print( len(np.unique(labelTruth))== len(np.unique(labelPredicted)))

#confusionMatrix(labelTruth, labelPredicted, labelTarget,DataLoader.DataName.dance, percentage=5.26)


## emotional
#print('Affichage Matrix Emotional')
#labelTruth, labelPredicted, labelTarget = loadMatrixFromFile(DataLoader.DataName.emotional, "action")
#confusionMatrix(labelTruth, labelPredicted, labelTarget,DataLoader.DataName.emotional, percentage=7.2423)


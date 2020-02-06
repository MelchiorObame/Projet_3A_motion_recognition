import numpy as np
import pickle
import pandas as pd
import os
import sys

import reshapeDataToMatrix
import distance_measure as dm
import DataLoader
import KNN_DTW


def KNN(dataName,labelstarget="action",normalizeData=False , n_neighbors=1, max_warping_window= 1, limitDataTest=-1,limitDataTrain=-1, THO=0.2, normalized=False, confidence=False, viewConfusioMatrix=False, saveLabelsLog=True):
    """Launch KNN model
    Arguments : 
        limit : first n=limit actions  to predict in all action  """
    model= KNN_DTW.KnnDtw(n_neighbors, max_warping_window)
    X_train, actionLabelsTrain, subjectLabelsTrain, X_test, actionLabelsTest, subjectLabelsTest = reshapeDataToMatrix.train_test_split(dataName, normalizeData=normalizeData, labelTarget=labelstarget)
    print('train size '+str(len(actionLabelsTrain)))
    print('test size '+str(len(actionLabelsTest)))
    # divide data to limit
    if limitDataTest != -1: #all the data Set. if limit =-1 , it returns all data Set
        X_test= X_test[:limitDataTest]
        actionLabelsTest=  actionLabelsTest[:limitDataTest]
        subjectLabelsTest = subjectLabelsTest[:limitDataTest]
        print('data segmentation on Test')
    if limitDataTrain != -1:
        X_train=X_train[:limitDataTrain]
        actionLabelsTrain= actionLabelsTrain[:limitDataTrain]
        subjectLabelsTrain = subjectLabelsTrain[:limitDataTrain]
        print('data segmentation on Train ')
    if(labelstarget.lower() =="action"):
        print('\nKNN ------ DB : '+ dataName.upper()+' Based on Action -------')
        model.fit(np.array(X_train),np.array(actionLabelsTrain))
        print('fit OK: action')
    if(labelstarget.lower()=="subject"):
        print('\nKNN ------ DB : '+ dataName.upper()+' Based on Subject -------')
        model.fit(np.array(X_train),np.array(subjectLabelsTrain))
        print('fit OK: subject')
    elif(labelstarget.lower() not in ['action', 'subject']):
        print("Unknown labels Target")
        return 0
    GoodPrediction=0
    print("Prediction...")
    Max=len(X_test)
    labelsPredicted = np.empty(Max, dtype=object)
    for i in range(Max):
        print(str(i+1)+"/"+str(Max))
        label , _= model.predict(X_test[i],dataName ,THO, normalized, confidence)
        print("predicted Label is "+str(label))
        labelsPredicted[i]=label
        if labelstarget.lower()=="action":
            if(label == actionLabelsTest[i]):
                GoodPrediction+=1
                print("--->> Good prediction")
            else:
                print('--->> Bad prediction\ncorrect label was: '+str(actionLabelsTest[i]))            
        else:
            if(label == subjectLabelsTest[i]):
                GoodPrediction+=1
                print("Good prediction")
            else:
                print('Bad prediction\ncorrect label was : '+str(subjectLabelsTest[i]))
    percentage = (GoodPrediction*100)/Max
    print("percentage of recognition : " +str(percentage))
    #File saving 
    if(saveLabelsLog):
        if(labelstarget.lower() == 'action'):
            saveMatrixFile(dataName,labelstarget,actionLabelsTest,labelsPredicted)
        else:
            saveMatrixFile(dataName,labelstarget,subjectLabelsTest,labelsPredicted)
    #visualisation
    if(viewConfusioMatrix and labelstarget.lower()=="action"): 
        import visualisation
        matrix=visualisation.confusionMatrix(actionLabelsTest, labelsPredicted, labelstarget, dataName, percentage)
    elif(viewConfusioMatrix and labelstarget.lower()=="subject"):
        import visualisation
        matrix=visualisation.confusionMatrix(subjectLabelsTest, labelsPredicted, labelstarget, dataName, percentage) #il se peut que se soit le sujet qu'on doit enregistrer. ça depend de c que prend la matrice de confusion
    return percentage,labelsPredicted, actionLabelsTest,subjectLabelsTest
        

def bestNeighbors(dataName,n_neighbors=3, labelstarget="action",max_warping_window= 1, limitDataTest=-1,limitDataTrain=-1, THO=0.2, normalized=False, confidence=False):
    result    = np.zeros((n_neighbors+1)//2)
    neighbors = [i for i in range(1,n_neighbors+1,2)]
    print('------------------- n_neighbors = 1  ')
    KnnResult = KNN( DataLoader.DataName.dance,normalized=normalized, limitDataTest=limitDataTest,limitDataTrain=limitDataTrain, labelstarget=labelstarget ,n_neighbors=1, max_warping_window=max_warping_window, viewConfusioMatrix=False, saveLabelsLog=False)
    result[0] = KnnResult[0]
    print('n_neighbors = 1  ,maxPercentage = '+str(result[0]))
    if(labelstarget.lower() == 'action'):
        saveMatrixFile(dataName,labelstarget,KnnResult[2],KnnResult[1], loopNeighbors=True)
    else:
        saveMatrixFile(dataName,labelstarget,KnnResult[3],KnnResult[1], loopNeighbors=True)            
    for i in neighbors[1:]:
        print('------------------- n_neighbors ='+str(i))
        KnnResult = KNN( DataLoader.DataName.dance,normalized=normalized, limitDataTest=limitDataTest,limitDataTrain=limitDataTrain, labelstarget=labelstarget ,n_neighbors=i, max_warping_window=max_warping_window, viewConfusioMatrix=False, saveLabelsLog=False)
        result[result.index(i)] = KnnResult[0]
        print('n_neighbors = '+str(i)+'  , percentage ='+str(result[result.index(i)]))
        if result[result.index(i)] > result[result.index(i)-1]:
            if(labelstarget.lower() == 'action'):
                saveMatrixFile(dataName,labelstarget,KnnResult[2],KnnResult[1], loopNeighbors=True)
            else:
                saveMatrixFile(dataName,labelstarget,KnnResult[3],KnnResult[1], loopNeighbors=True)        
    maxPercentage = max(result)
    BestNeighbors = neighbors[result.index(maxPercentage)]
    print('BestNeighbors = '+str(BestNeighbors)+'  ,maxPercentage = ',+str(maxPercentage))

        

def saveMatrixFile(dataName,labelstarget,target,labelsPredicted, loopNeighbors=False):
    if not os.path.exists(DataLoader.DirPaths.Matrix_ActionPredicted):
        os.mkdir(DataLoader.DirPaths.Matrix_ActionPredicted)
    path= DataLoader.DirPaths.Matrix_ActionPredicted
    if loopNeighbors:
        if not os.path.exists(DataLoader.DirPaths.Matrix_ActionPredictedLoopNeighbors):
            os.mkdir(DataLoader.DirPaths.Matrix_ActionPredictedLoopNeighbors)
            path= DataLoader.DirPaths.Matrix_ActionPredictedLoopNeighbors
    dico={}
    dico['labelsPredicted']=labelsPredicted
    if(dataName == DataLoader.DataName.action and labelstarget.lower()== "action"):
        dico['actionLabelsTest']=target
        with open(os.path.join(path,DataLoader.DataMatrixLogFileName.MsrAcion3D_action), "wb") as f:
            pickle.dump(dico, f)
    elif(dataName == DataLoader.DataName.action and labelstarget.lower()== "subject"):
        dico['subjectLabelsTest']=target
        with open(os.path.join(path,DataLoader.DataMatrixLogFileName.MsrAcion3D_subject), "wb") as f:
            pickle.dump(dico, f)
    elif(dataName == DataLoader.DataName.dance and labelstarget.lower()== "action"):
        dico['actionLabelsTest']=target
        with open(os.path.join(path,DataLoader.DataMatrixLogFileName.DanceDB_action), "wb") as f:
            pickle.dump(dico, f)
    elif(dataName == DataLoader.DataName.dance and labelstarget.lower()== "subject"):
        dico['subjectLabelsTest']=target
        with open(os.path.join(path,DataLoader.DataMatrixLogFileName.DanceDB_subject), "wb") as f:
            pickle.dump(dico, f)
    elif(dataName == DataLoader.DataName.emotional and labelstarget.lower()== "action"):
        dico['actionLabelsTest']=target
        with open(os.path.join(path,DataLoader.DataMatrixLogFileName.EmotionalDB_action), "wb") as f:
            pickle.dump(dico, f)
    elif(dataName == DataLoader.DataName.emotional and labelstarget.lower()== "subject"):
        dico['subjectLabelsTest']=target
        with open(os.path.join(path,DataLoader.DataMatrixLogFileName.EmotionalDB_subject), "wb") as f:
            pickle.dump(dico, f)    



###********************** TEST PART ******************
            
###____________________________________ D I S T A N C E   M E A S U R E
            
##=============== MSR action 3D 
#j1= np.array([1,2,1,0])
#j2= np.array([1,2,8,0])
#print("distance Euclid : "+str(dm.euclid_distance_joints(j1,j2)))
#print("confidence :"+str(dm.thresholdConfidence(3,3)))  
#frame1= np.random.rand(20,4)
#frame2= np.random.rand(20,4)
#print("distance frames :"+str(dm.distance_frames(frame1,frame2,0.2,confidence=True)))
##================ Dance DB

#j1 = np.array([-1.83298,97.8449,-19.5414,29.4326,-89.9987,29.4985])
#j2 = np.array([8.64259,1.95398,-2.38835,-171.15,3.0966,94.2266])
#distJointRot = dm.distanceJointWithRotationDance(j1,j2)

#f1=np.random.rand(55,6)
#f2=np.random.rand(55,6)
#distFrameRot = dm.distance_framesWithRotationDance(f1, f2)
##============== Emotional DB
    
#j1 = np.array([-1.83298,97.8449,-19.5414,29.4326,-89.9987,29.4985])
#j2 = np.array([8.64259,1.95398,-2.38835,-171.15,3.0966,94.2266])
#distJointRotEmo = dm.distanceJointWithRotationEmotional(j1,j2,1)
#
#f1=np.random.rand(55,6)
#f2=np.random.rand(55,6)
#distFrameRotEmo= dm.distance_framesWithRotationEmotional(f1, f2)
            
###________________________________  D A T A L O A D ER
            
#DataLoader.DataLoader("MSR Action3D")
#DataLoader.DataLoader("Dance")
#DataLoader.DataLoader("Emotion")
            
##________________________________  R E S H A P E   M A T R I X
            
##==================== Action
#df = pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.ActionCSV.fileName))
#print("action")
#actionMSR= reshapeDataToMatrix.getAction(1,1,1, df)
#dataSetAction = reshapeDataToMatrix.getDataSet(DataLoader.DataName.action, True)
#dataset,actionLabels,subjectLabels= reshapeDataToMatrix.loadDataSet(DataLoader.DataName.action)

#X_train, actionLabelsTrain, subjectLabelsTrain, X_test, actionLabelsTest, subjectLabelsTest = reshapeDataToMatrix.train_test_split(DataLoader.DataName.action) 

#print('loading OK')
#a1=reshapeDataToMatrix.getAction(1,1,1, df)
#a2=reshapeDataToMatrix.getAction(1,1,2, df)
#a3=reshapeDataToMatrix.getAction(3,5,2, df)
#model= KNN_DTW.KnnDtw(n_neighbors=5, max_warping_window=3)
#distance, cost= model._dtw_distance(a3,a1,DataLoader.DataName.action, THO=0.0,normalized=False)
#print("distance= "+str(distance))

## ================== Dance

#df = pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.DanceCSV.fileName))
##print("dance")
#actionDance=reshapeDataToMatrix.getActionDance(17,0,0, df)
#dataSetDance = reshapeDataToMatrix.getDataSet(DataLoader.DataName.dance, False)
#data, dance, subject= reshapeDataToMatrix.loadDataSet(DataLoader.DataName.dance)
    
########## test distance dance a1, a2 case Normalized
#df = reshapeDataToMatrix.NormalizeData(pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.DanceCSV.fileName)))
#print('loading OK')
#a1=reshapeDataToMatrix.getActionDance(17,0,0, df)
#a2=reshapeDataToMatrix.getActionDance(17,0,1, df)
#a3=reshapeDataToMatrix.getActionDance(3,0,0, df)
#a4=reshapeDataToMatrix.getActionDance(1,0,0, df)
#
#####
#model= KNN_DTW.KnnDtw(n_neighbors=2, max_warping_window=3)
#####
##begin = time.time()
#distance= model._dtw_distance(a1,a2,DataLoader.DataName.dance, THO=0.0,normalized=True)
##end = time.time()
#print("distance= "+str(distance))
##print('executed in : '+str(end-begin))
###___________________
#### probleme sur le nombre de labels dans le cas de 'action , pour subject c'est OK '
#X_train, actionLabelsTrain, subjectLabelsTrain, X_test, actionLabelsTest, subjectLabelsTest = reshapeDataToMatrix.train_test_split(DataLoader.DataName.dance, normalizeData=True,  labelTarget='subject')
#print(len(np.unique(actionLabelsTest)))
#print(len(np.unique(actionLabelsTrain)))
#print(len(DataLoader.DanceCSV.dance.items()))
#print(len(np.unique(actionLabelsTest)) == len(DataLoader.DanceCSV.dance.items()))
#print(len(np.unique(subjectLabelsTest)) == len(DataLoader.DanceCSV.subjects.items()))

#X_trainNorm, actionLabelsTrainNorm, subjectLabelsTrainNorm, X_testNorm, actionLabelsTestNorm, subjectLabelsTestNorm = reshapeDataToMatrix.train_test_split(DataLoader.DataName.dance, normalizeData=True)

##=================== Emotional  : OK
#import time
#df = pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.EmotionalCSV.fileName))
#print("emotion test")
#e1=reshapeDataToMatrix.getActionEmotional(8,0,1,df)
#e2=reshapeDataToMatrix.getActionEmotional(6,0,0,df)
#e3=reshapeDataToMatrix.getActionEmotional(9,0,0,df)
#dataSetEmotional = reshapeDataToMatrix.getDataSet(DataLoader.DataName.emotional, True,normalizeData=True)

#model= KNN_DTW.KnnDtw(n_neighbors=3, max_warping_window=1)
#begin = time.time()
#distance= model._dtw_distance(e1,e2,DataLoader.DataName.emotional, THO=0.0,normalized=False)
#end = time.time()
#print("distance= "+str(distance))
#print('executed in : '+str(end-begin))
###___________________
###### les labels sont OK pour emotional 
#X_train, actionLabelsTrain, subjectLabelsTrain, X_test, actionLabelsTest, subjectLabelsTest = reshapeDataToMatrix.train_test_split(DataLoader.DataName.emotional, normalizeData=True, labelTarget='subject')
#print(len(np.unique(subjectLabelsTest)) == len(DataLoader.EmotionalCSV.subjects.items()))



##_________________________________  K N N        
##========= Action 3D :  OK
#Knn_MsrAction3D_action  = KNN(DataLoader.DataName.action, labelstarget="action",n_neighbors=2, limit=-1, max_warping_window= 5, THO=0.2, normalized=False, confidence=False, viewConfusioMatrix=False, saveLabelsLog=True)
#Knn_MsrAction3D_subject = KNN(DataLoader.DataName.action, labelstarget="subject",n_neighbors=2,limit=-1, max_warping_window= 5, THO=0.2, normalized=False, confidence=False, viewConfusioMatrix=False, saveLabelsLog=True)
       
##========= Dance: 1/2   #penser à effacer le .dat si on change normalizedd
#Knn_DanceDB_subject = KNN( DataLoader.DataName.dance,normalizeData=True, limitDataTest=-1,limitDataTrain=-1, labelstarget="subject",n_neighbors=3, max_warping_window= 4,viewConfusioMatrix=True, saveLabelsLog=True)
#Knn_DanceDB_action =  KNN( DataLoader.DataName.dance,normalized=True, limitDataTest=-1,limitDataTrain=-1,labelstarget="action",n_neighbors=5, max_warping_window=3,viewConfusioMatrix=True, saveLabelsLog=True)

##========= Emotion   :: prendre max_win=1
#Knn_EmotionDB_action =  KNN( DataLoader.DataName.emotional, labelstarget="action",n_neighbors=3,limitDataTest=-1,limitDataTrain=-1, max_warping_window= 1,viewConfusioMatrix=False, saveLabelsLog=True)
#Knn_EmotionDB_subject = KNN( DataLoader.DataName.emotional, labelstarget="subject",n_neighbors=3, max_warping_window= 1,viewConfusioMatrix=False, saveLabelsLog=False)

#### ____________ General test : Find best N
            
#actionMSRAction = bestNeighbors(DataLoader.DataName.action,n_neighbors=10, labelstarget="action",max_warping_window= 5, limitDataTest=-1,limitDataTrain=-1, THO=0.2, normalized=False, confidence=False)
#dance = bestNeighbors(dataName,n_neighbors=3, labelstarget="action",max_warping_window= 1, limitDataTest=-1,limitDataTrain=-1, THO=0.2, normalized=False, confidence=False)
#emotional = bestNeighbors(DataLoader.DataName.emotional,n_neighbors=1, labelstarget="action",max_warping_window= 1, limitDataTest=-1,limitDataTrain=-1, THO=0.2, normalized=False, confidence=False)
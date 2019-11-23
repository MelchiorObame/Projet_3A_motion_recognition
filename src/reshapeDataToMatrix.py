import pandas as pd
import os
import pickle
import numpy as np

import DataLoader
import distance_measure


        
def getAction(a,s,e,df):
    """return an action in the shape[Nframes, 20, 4]
    Arguments : a action number, s subject number, e essaie number, df : dataFrame of previously csv action file"""
    action = []
    for row in df.itertuples():
        if(getattr(row, 'action')==a and getattr(row, 'subject')==s and getattr(row, 'essai')==e):       
            #append joint
            action.append(np.array([getattr(row, 'x'),getattr(row, 'y'),getattr(row, 'z'),getattr(row, 'confidence')]))
    action= np.array(action)
    NFrames=len(action)//distance_measure.Data.NJoints
    return np.reshape(action,[NFrames,distance_measure.Data.NJoints,distance_measure.Data.NcoordinatesOfJoint])

    

def getDataSet(dataName,save):
    """cree la matrice contenant toutes les actions 
    return :: dataSet, a numpy Array of shape [Naction, Nframes, 20,4]"""
    #on cree la np.array content 567 actions
    dataSet=[]
    ActionLabels=[]
    SubjectLabels=[]
    if dataName==DataLoader.DataName.action:
        df = pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.ActionCSV.fileName))
        actionArray=df.drop(['y','x', 'time', 'joint', 'z', 'confidence'], axis=1).values
        distinctActionArray=np.unique(actionArray, axis=0)
        #lecture des actions,
        for i in range(len(distinctActionArray)):
            print(str(i)+"/"+str(len(distinctActionArray)))
            tupleAction=distinctActionArray[i]
            action= getAction(tupleAction[0],tupleAction[1],tupleAction[2],df) 
            # add a and s  labels
            ActionLabels.append(tupleAction[0])
            SubjectLabels.append(tupleAction[1])
            print("action = "+str(tupleAction[0]) +" subject = "+str(tupleAction[1])+" essaie= "+str(tupleAction[2]))
            dataSet.append(action)
    #saving by pickle
    if(save):
        dico={}
        if dataName==DataLoader.DataName.action:
            dico['dataSet']=dataSet
            dico['ActionLabels']=ActionLabels
            dico['SubjectLabels']=SubjectLabels
            print("Saving progress...")
            with open(DataLoader.DataSetFileName.action, "wb") as f:
                pickle.dump(dico, f)
            print("Save actions successful.")
    return dataSet


def loadDataSet(dataName):
    """load most easly data from a .dat file
    retourne : triplet (dataSet, ActionLabels, SubjectLabels)"""
    
    print("Loading progress...")
    if dataName==DataLoader.DataName.action:
        with open(DataLoader.DataSetFileName.action, "rb") as f:
            dico = pickle.load(f)
        dataset=dico['dataSet']
        actionLabels=dico['ActionLabels']
        subjectLabels=dico['SubjectLabels']
        print("Loadinf DataSet Action file : success.")
        return dataset,actionLabels,subjectLabels
    elif dataName==DataLoader.DataName.emotional:
        with open(DataLoader.DataSetFileName.emotional, "rb") as f:
            dico = pickle.load(f)
        dataset=dico['dataSet']
        actionLabels=dico['ActionLabels']
        subjectLabels=dico['SubjectLabels']
        print("Load emotional dataSet file : success")
        return dataset,actionLabels,subjectLabels
    elif dataName==DataLoader.DataName.dance:
        with open(DataLoader.DataSetFileName.dance, "rb") as f:
            dico = pickle.load(f)
        dataset=dico['dataSet']
        actionLabels=dico['ActionLabels']
        subjectLabels=dico['SubjectLabels']
        print("Load dance dataSet file : success")
        return dataset,actionLabels,subjectLabels
    
    

def train_test_split(dataName):
    """divise le jeu de données en train test selon les deux labels d'actions et de sujets"""
    data, action, subject =loadDataSet(dataName)

    dataActionSubject= np.array([data,action, subject]).transpose() #numpy array
    df = pd.DataFrame({'data': dataActionSubject[:, 0], 'action': dataActionSubject[:, 1], 'subject': dataActionSubject[:, 2]})
    #dataFrame de test
    testDataSet = df.drop_duplicates(['action','subject'])

    #list de test
    X_test= testDataSet['data'].values.tolist()
    actionLabelsTest= testDataSet['action'].values.tolist()
    subjectLabelsTest= testDataSet['subject'].values.tolist()
    #list de train : Dataframe
    trainDataSet = df[~df.index.isin(testDataSet.index)]    
    X_train= trainDataSet['data'].values.tolist()
    actionLabelsTrain= trainDataSet['action'].values.tolist()
    subjectLabelsTrain= trainDataSet['subject'].values.tolist()
    return  X_train, actionLabelsTrain, subjectLabelsTrain, X_test, actionLabelsTest, subjectLabelsTest

#X_train, actionLabelsTrain, subjectLabelsTrain, X_test, actionLabelsTest, subjectLabelsTest = train_test_split(DataLoader.DataName.action)

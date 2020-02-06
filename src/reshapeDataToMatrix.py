import pandas as pd
import os
import pickle
import numpy as np

import DataLoader
import distance_measure

        
def getAction(a,s,t,df): #ok
    """return an action in the shape[Nframes, 20, 4]    x,y,z,c
    Arguments : a action number, s subject number, e essaie number, df : dataFrame of previously csv action file"""
    requete= 'action=='+str(a)+' and subject=='+ str(s) +' and try_=='+str(t)
    action=df.query(requete).drop(['action','subject','try_','time','joint'], axis=1).values.tolist()
    action= np.array(action)
    NFrames=len(action)//distance_measure.Data.NJoints
    return np.reshape(action,[NFrames,distance_measure.Data.NJoints,distance_measure.Data.NcoordinatesOfJoint])

def getActionDance(d, s, v, df): # ok 
    """return an action in the shape[Nframes, 55, 6]   
    Arguments : dance number,  subject number, version  number(try), df : dataFrame of previously csv action file"""
    requete= 'DanceType=='+str(d)+' and Subject=='+ str(s) +' and Version=='+str(v)
    dance=df.query(requete).drop(['DanceType','Subject','Version','Time','Joint'], axis=1).values.tolist()
    dance= np.array(dance)
    NFrames=len(dance)//distance_measure.DataDanceInfo.NJoints
    return np.reshape(dance, [NFrames,distance_measure.DataDanceInfo.NJoints,distance_measure.DataDanceInfo.NcoordinatesOfJoint])

def getActionEmotional(e, s, v, df): # ok
    """ return an action in the shape[Nframes, 23, 6]
     Arguments : emotion number,  subject number, version  number(try), df : dataFrame of previously csv action file """
    requete= 'IntendedEmotion=='+str(e)+' and Subject=='+ str(s) +' and Version=='+str(v)
    emotion=df.query(requete).drop(['IntendedEmotion','Subject','Version','Time','Joint'], axis=1).values.tolist()
    emotion=np.array(emotion)
    print(emotion.shape)
    NFrames=len(emotion)//distance_measure.DataEmotionalInfo.NJoints
    print(NFrames)
    print(requete)
    return np.reshape(emotion, [NFrames,distance_measure.DataEmotionalInfo.NJoints ,distance_measure.DataEmotionalInfo.NcoordinatesOfJoint])
    

def NormalizeData(df, oneCompletion=False, dataName=None):
    """Normalize data just for Dance and Emotional DB"""
    maxXpos, maxYpos, maxZpos= df.loc[df['Xposition'].idxmax()]['Xposition'], df.loc[df['Yposition'].idxmax()]['Yposition'], df.loc[df['Zposition'].idxmax()]['Zposition']
    minXpos, minYpos, minZpos= df.loc[df['Xposition'].idxmin()]['Xposition'], df.loc[df['Yposition'].idxmin()]['Yposition'], df.loc[df['Zposition'].idxmin()]['Zposition']
    #maxXrot, maxYrot, maxZrot= df.loc[df['Xrotation'].idxmax()]['Xrotation'], df.loc[df['Yrotation'].idxmax()]['Yrotation'], df.loc[df['Zrotation'].idxmax()]['Zrotation']
    #minXrot, minYrot, minZrot= df.loc[df['Xrotation'].idxmin()]['Xrotation'], df.loc[df['Yrotation'].idxmin()]['Yrotation'], df.loc[df['Zrotation'].idxmin()]['Zrotation']
    if oneCompletion:
        df['Xposition'] = df['Xposition'].apply(lambda x: 1 if x == 0 else x)
        df['Yposition'] = df['Yposition'].apply(lambda x: 1 if x == 0 else x)
        df['Zposition'] = df['Zposition'].apply(lambda x: 1 if x == 0 else x)
    #normalize x, y and z coordinates
    df['Xposition'] = df['Xposition'].apply(lambda x: (x - minXpos)/(maxXpos-minXpos))
    df['Yposition'] = df['Yposition'].apply(lambda x: (x - minYpos)/(maxYpos-minYpos))
    df['Zposition'] = df['Zposition'].apply(lambda x: (x - minZpos)/(maxZpos-minZpos))
    print('Data Normalization...')
    return df



def getDataSet(dataName,save, normalizeData=False): #ok
    """ Cree la matrice contenant toutes les actions 
    return :: dataSet, a numpy Array of shape [Naction, Nframes, 20,4] for MSR Action 3D"""
    dataSet=[] #contient toutes les actions
    ActionLabels=[] 
    SubjectLabels=[]
    DanceLabels=[]
    EmotionLabels=[]
    if dataName==DataLoader.DataName.action: #ok
        #creates csv file if not exists
        if not os.path.exists(os.path.join(DataLoader.DirPaths.CSVdir ,DataLoader.ActionCSV.fileName)):
            DataLoader.DataLoader(DataLoader.DataName.action)
        print('Loading CSV File...')
        df = pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.ActionCSV.fileName))
        actionArray=df.drop(['y','x', 'time', 'joint', 'z', 'confidence'], axis=1).values #contains 3 columns
        distinctActionArray=np.unique(actionArray, axis=0)
        dataSet = np.empty(len(distinctActionArray), dtype=object)
        #lecture des actions,
        for i in range(len(distinctActionArray)):
            print(str(i+1)+"/"+str(len(distinctActionArray)))
            tupleAction=distinctActionArray[i]
            action= getAction(tupleAction[0],tupleAction[1],tupleAction[2],df) 
            # add a and s  labels
            ActionLabels.append(tupleAction[0])
            SubjectLabels.append(tupleAction[1])
            dataSet[i]=action   
    elif dataName==DataLoader.DataName.dance:  #ok
        #creates csv file if not exists
        if not os.path.exists(os.path.join(DataLoader.DirPaths.CSVdir ,DataLoader.DanceCSV.fileName)):
            DataLoader.DataLoader(DataLoader.DataName.dance)
        print('Loading CSV File...')
        if normalizeData:
            df= NormalizeData( pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.DanceCSV.fileName)), dataName=dataName )
        else:
            df=pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.DanceCSV.fileName))
        #distinction between action based on  'Dance Type','Subject','Version'
        actionArray=df.drop(['Time','Xposition','Yposition','Zposition','Joint'], axis=1).values  #contains 3 
        distinctActionArray=np.unique(actionArray, axis=0)
        dataSet = np.empty(len(distinctActionArray), dtype=object)
        #lecture des actions de danceDB
        for i in range(len(distinctActionArray)):#must be =134
            print(str(i+1)+"/"+str(len(distinctActionArray)))
            tupleAction=distinctActionArray[i]
            dance= getActionDance(tupleAction[0],tupleAction[1],tupleAction[2],df)
            # add a and s  labels
            DanceLabels.append(tupleAction[0])  #dance type
            SubjectLabels.append(tupleAction[1]) #subject
            dataSet[i]=dance
    elif dataName==DataLoader.DataName.emotional:#ok
        #creates csv file if not exists
        if not os.path.exists(os.path.join(DataLoader.DirPaths.CSVdir ,DataLoader.EmotionalCSV.fileName)):
            DataLoader.DataLoader(DataLoader.DataName.emotional)
        print('Loading CSV File...')
        if normalizeData:
            df=NormalizeData(pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.EmotionalCSV.fileName)) , oneCompletion=True, dataName=dataName)
        else:
            df= pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.EmotionalCSV.fileName))
        actionArray=df.drop(['Time','Xposition','Yposition','Zposition','Yrotation','Xrotation','Zrotation','Joint'], axis=1).values  #contains 3 
        distinctActionArray=np.unique(actionArray, axis=0)
        dataSet = np.empty(len(distinctActionArray), dtype=object)
        #lecture des actions de danceDB
        for i in range(len(distinctActionArray)):#must be =1447
            print(str(i+1)+"/"+str(len(distinctActionArray)))
            tupleAction=distinctActionArray[i]
            emotion= getActionEmotional( tupleAction[0], tupleAction[1],tupleAction[2], df)
            EmotionLabels.append(tupleAction[0]) #emotion type
            SubjectLabels.append(tupleAction[1]) #subject
            dataSet[i]=emotion    
    #saving by pickle
    if(save):
        if not os.path.exists(DataLoader.DirPaths.SavedData):
                os.mkdir(DataLoader.DirPaths.SavedData)
        dico={}
        if dataName==DataLoader.DataName.action:
            dico['dataSet']=dataSet
            dico['ActionLabels']=ActionLabels
            dico['SubjectLabels']=SubjectLabels
            print("Saving progress...")
            with open(os.path.join(DataLoader.DirPaths.SavedData,DataLoader.DataSetFileName.action), "wb") as f:
                pickle.dump(dico, f)
        elif dataName==DataLoader.DataName.dance: #ok 
            dico['dataSet']=dataSet
            dico['DanceLabels']=DanceLabels
            dico['SubjectLabels']=SubjectLabels
            print("Saving progress...")
            with open(os.path.join(DataLoader.DirPaths.SavedData, DataLoader.DataSetFileName.dance), "wb") as f:
                pickle.dump(dico, f)
        elif dataName==DataLoader.DataName.emotional: #ok 
            dico['dataSet']=dataSet
            dico['EmotionLabels']=EmotionLabels
            dico['SubjectLabels']=SubjectLabels
            print("Saving progress...")
            with open(os.path.join(DataLoader.DirPaths.SavedData, DataLoader.DataSetFileName.emotional), "wb") as f:
                pickle.dump(dico, f)
        print("backup properly made")
    return dataSet



def loadDataSet(dataName, normalizeData=False):
    """load most easly data from a .dat file
    retourne : triplet (dataSet, ActionLabels, SubjectLabels) """
    if dataName == DataLoader.DataName.action:
        if not os.path.exists(os.path.join(DataLoader.DirPaths.SavedData,DataLoader.DataSetFileName.action)):
            getDataSet(DataLoader.DataName.action, True, normalizeData=normalizeData)
    elif dataName == DataLoader.DataName.dance:
        if not os.path.exists(os.path.join(DataLoader.DirPaths.SavedData,DataLoader.DataSetFileName.dance)):
            getDataSet(DataLoader.DataName.dance, True ,normalizeData=normalizeData)
    elif dataName == DataLoader.DataName.emotional:
        if not os.path.exists(os.path.join(DataLoader.DirPaths.SavedData,DataLoader.DataSetFileName.emotional)):
            getDataSet(DataLoader.DataName.emotional, True, normalizeData=normalizeData)
    if dataName==DataLoader.DataName.action: #ok
        with open(os.path.join(DataLoader.DirPaths.SavedData,DataLoader.DataSetFileName.action), "rb") as f:
            dico = pickle.load(f)
        dataset=dico['dataSet']
        actionLabels=dico['ActionLabels']
        subjectLabels=dico['SubjectLabels']
        print("Loading "+DataLoader.DataSetFileName.action+" file : success ")
        return dataset,actionLabels,subjectLabels
    elif dataName==DataLoader.DataName.dance: #ok 
        with open(os.path.join(DataLoader.DirPaths.SavedData,DataLoader.DataSetFileName.dance), "rb") as f:
            dico = pickle.load(f)
        dataset=dico['dataSet']
        DanceLabels=dico['DanceLabels']
        subjectLabels=dico['SubjectLabels']
        print("Loading "+DataLoader.DataSetFileName.dance+" file : success ")
        return dataset,DanceLabels,subjectLabels
    elif dataName==DataLoader.DataName.emotional: 
        with open(os.path.join(DataLoader.DirPaths.SavedData, DataLoader.DataSetFileName.emotional), "rb") as f:
            dico = pickle.load(f)
        dataset=dico['dataSet']
        EmotionLabels=dico['EmotionLabels']
        subjectLabels=dico['SubjectLabels']
        print("Loading "+DataLoader.DataSetFileName.emotional+" file : success ")
        return dataset,EmotionLabels,subjectLabels


def train_test_split(dataName, normalizeData=False, labelTarget=None):#pas OK pour dance et emotional
    """divise le jeu de données en train test selon les deux labels d'actions et de sujets"""
    data, action, subject =loadDataSet(dataName, normalizeData=normalizeData)
    dataActionSubject= np.array([data,action, subject]).transpose() #numpy array
    df = pd.DataFrame({'data': dataActionSubject[:, 0], 'action': dataActionSubject[:, 1], 'subject': dataActionSubject[:, 2]})
    if dataName == DataLoader.DataName.action:
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
    if dataName == DataLoader.DataName.dance or dataName == DataLoader.DataName.emotional:
        X_train, actionLabelsTrain, subjectLabelsTrain=[], [], []
        X_test, actionLabelsTest, subjectLabelsTest =[], [], []
        ratio=1
        if dataName == DataLoader.DataName.dance:ratio=3 #on prend le tier environs dans le cas dance DB et le quart pour emotional
        else:ratio=4
        if labelTarget.lower() =="action":
            groupByDF = df.groupby('action')
        elif labelTarget.lower() =="subject":
            groupByDF = df.groupby('subject')
        for name, group in groupByDF:
            #add the half of every group
            mid = len(group)//ratio
            labelsForGroup = group.iloc[:,2].values.tolist() # subject number
            actionsForGroup = [name for k in range(len(group))]  #numero des actions 
            X_test += group.iloc[:mid,0].values.tolist()
            actionLabelsTest += actionsForGroup[:mid]
            subjectLabelsTest +=labelsForGroup[:mid]
            X_train += group.iloc[mid:,0].values.tolist()
            subjectLabelsTrain += labelsForGroup[mid:]
            actionLabelsTrain += actionsForGroup[mid:]
        return  X_train, actionLabelsTrain, subjectLabelsTrain, X_test, actionLabelsTest, subjectLabelsTest
        

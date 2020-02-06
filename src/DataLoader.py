"""
Emotional DB : http://ebmdb.tuebingen.mpg.de
Dance DB : http://dancedb.eu/main/performances
MSR Action3d : https://documents.uow.edu.au/~wanqing/#Datasets

"""
import os
import numpy as np
import glob
from BVH import bvh_emotionalFormat,bvh_preprocessing
from BVH.bvh_parsers import BVHParser

class DataSetFileName:  
    #for .dat file
    action = "dataSetActionMsr3D.dat"
    dance = 'dataSetDance.dat'
    emotional = 'dataSetEmotional.dat'
    danceNormalized = 'dataSetDance.dat'
    emotionalNormalized = 'dataSetEmotional.dat'
    
class DataMatrixLogFileName:
    MsrAcion3D_action = 'MsrAction3D_action.dat'
    MsrAcion3D_subject = 'MsrAction3D_subject.dat'
    DanceDB_action = 'DanceDB_action.dat'
    DanceDB_subject = 'DanceDB_subject.dat'
    EmotionalDB_action = 'EmotionalDB_action.dat'
    EmotionalDB_subject = 'EmotionalDB_subject.dat'
    
    
class DataName:
    action = "MSR Action3D"
    dance = "Dance"
    emotional = "Emotion"
    
class DirPaths:
    action = '../data/MSRAction3D/'
    dance = "../data/DanceDB/"
    emotional = "../data/EmotionalBody/"
    CSVdir='../data/csv/'
    Matrix_ActionPredicted='../data/Matrix_ActionPredicted/'
    Matrix_ActionPredictedLoopNeighbors=Matrix_ActionPredicted+'loopNeigbors/'
    SavedData='../data/snapshotData/'
    savedPicturesWindows_Warping = '../doc/images/windows_warping/'
    savedPicturesConfusionMatrix = '../doc/images/conf_Matrix/'
    savedCNNpictures ='../doc/images/CNN/'
    weightLocation='../model/'
    PictureDataSet='../data/PictureDataSet/'


class ActionCSV:
    fileName="Action3dData.csv"
    header= ['time','x','y','z','confidence','joint','action','subject','try_']
    actions={1:'high wave', 2:'horizontal wave', 3:'hammer', 4:'hand catch',5:'Foward punch',
    6:'high throw',7:'draw X',8:'draw tick',9:'draw circle',
    10:'hand clap',11:'two hand clap',12:'side boxing',13:'bend',
    14:'forward kick',15:'side kick',16:'jogging',17:'tennis swing',18:'tennis serve',19:'golf swing', 20:'pick-up throw'}
    sep=','
    NumberSubject=10
    pictureActionDBDirName='Action3DPictures'
    pictureActionDBDirNameSPMF='Action3DPicturesSPMF'

class DanceCSV:
    fileName="DanceData.csv"
    header= ['Time','Xposition','Yposition','Zposition','Joint','DanceType','Subject','Version']
    subjects={'Andria':0,'Anna':1,'Clio':2,'Elena':3,'Olivia':4,'Sophie':5,'StefanosKoullapis':6,'Stefanos':7,'Theodora':8,'Theodoros':9,'Vasso':10}
    dance={'Angry':0,'Annoyed':1,'Excited':2,'Happy':3,'Miserable':4,'Mix':5,'Pleased':6,'Relaxed':7,'Sad':8,'Satisfied':9,'Tired':10,'Bachata':11,'Zeibekiko-Fast':12,'Zeibekiko-Slow':13,'1os-antrikos-karsilamas':14,'2os-antrikos-karsilamas':15,'3os-antrikos-karsilamas':16,'Zeibekiko-CY':16,'Afraid':17,'Bored':18,'Capoeira':19,'Hasapiko':20,'Neutral':21,'Zeibekiko':22,'Active':23,'Curiosity':24,'Nervous':25,'Scary':26,'Haniotikos':27,'Maleviziotikos':28,'Flamenco':29}
    sep=','
    pictureDanceDBDirName='DanceDBPictures'
    pictureDanceDBDirNameSPMF='DanceDBPicturesSPMF'


class EmotionalCSV:
    fileName="EmotionalData.csv"
    header= ['Time','Xposition','Yposition','Zposition','Yrotation','Xrotation','Zrotation','Joint','IntendedEmotion','Subject','Version']
    emotions={'amusement':0,'anger':1,'disgust':2,'fear':3,'joy':4,'neutral':5,'pride':6,'relief':7,'sadness':8,'shame':9,'surprise':10}
    subjects={'AnBh':0,'DiMi':1,'HeGa':2,'LeSt':3,'MaMa':4,'NoVo':5,'PaPi':6,'SiGl':7,'Sinead':8}
    sep=','
    pictureEmotionalDBPicture='emotionalDBPictures'
    pictureEmotionalDBPictureSPMF='emotionalDBPicturesSPMF'
    incorrectData=['SiGl_nv_al_shame_21275-21992.bvh', 'PaPi_0509_nv_al_t2_sadness_15737-16254.bvh']


class DataLoader:
    "loads data"
    def __init__(self, DBName):
        self.filePath='/'
        self.dicEmotion={}
        self.dicDance={}
        self.jointArrayEmotional=[]
        self.jointArrayDance=[]
        self.jointArrayAction=['Right Shoulder','Left Shoulder','Neck','Spine','Right Hip','Left Hip','Middle Hip','Right Elbow','Left Elbow','Right Wrist','Left Wrist','Right Hand','Left Hand','Right Knee','Left Knee','Right Ankle','Left Ankle','Right Foot','Left Foot','Head']
        if not os.path.exists(DirPaths.CSVdir):
                os.mkdir(DirPaths.CSVdir)
 
        #getJoint=False
        if (DBName==DataName.action):
            "loads data from MSR Action3D"
            print("------------- MSR Action 3D")
            self.filePath= DirPaths.action
            assert self.filePath[-1]=='/'
            print("creating "+ActionCSV.fileName+" ...")
            myFile=open(os.path.join(DirPaths.CSVdir,ActionCSV.fileName), "w")
            headerLine = ActionCSV.sep.join(ActionCSV.header) + "\n"
            n=1
            myFile.write(headerLine)
            FilesTXT=glob.glob(self.filePath+'*.txt')
            for action in FilesTXT:
                fileNameSplit = os.path.basename(action).strip().split('.')[0].split('_')[:3]   #['a01', 's01', 'e01']
                print(os.path.basename(action)+'   '+str(n)+'/'+str(len(FilesTXT)))
                with open(action,'r', encoding='utf-8') as file:
                    line = file.readline()
                    time=0
                    while line:
                        for i in range(1,len(self.jointArrayAction)+1):
                            lineSplit=line.strip().split('  ')   #[1:].lstrip('0') pour tranformer 'a01' -> 1
                            row= [str(time), str(lineSplit[0]), str(lineSplit[1]), str(lineSplit[2]),str(lineSplit[3]),str(i),str(fileNameSplit[0][1:].lstrip('0')), str(fileNameSplit[1][1:].lstrip('0')),str(fileNameSplit[2][1:].lstrip('0'))]
                            ligne = ActionCSV.sep.join(row) + "\n"
                            myFile.write(ligne)
                            line = file.readline()
                        time+=1
                n+=1
            myFile.close()
        elif DBName==DataName.emotional: #ok
            "loads data from emotional"
            print('------------- Emotional DB')
            self.filePath= DirPaths.emotional
            assert self.filePath[-1]=='/'
            n=1
            FilesBVH=glob.glob(self.filePath+'*.bvh')
            myFile=open(os.path.join(DirPaths.CSVdir,EmotionalCSV.fileName), "w")
            headerLine = EmotionalCSV.sep.join(EmotionalCSV.header) + "\n"
            myFile.write(headerLine)
            #names, subject ,version and emations extraction
            for action in FilesBVH:
                if os.path.basename(action) not in EmotionalCSV.incorrectData:
                    fileNameSplit = os.path.basename(action).strip().split('.')[0].split('_')
                    print('data : '+str(n)+'/'+str(len(FilesBVH)))
                    if(not fileNameSplit[0]=='Session2'):
                        actor=fileNameSplit[0]
                        if actor not in EmotionalCSV.subjects.keys():
                            actor=actor[:4]
                        if(len(fileNameSplit)==10):
                            emotion=fileNameSplit[-1]
                        elif(len(fileNameSplit) in [5,7]):
                            emotion=fileNameSplit[-2]
                            if emotion not in EmotionalCSV.emotions.keys() and actor=='SiGl':
                                emotion= os.path.basename(action).strip().split('.')[1].split('_')[-1]
                        elif(len(fileNameSplit)==6):
                            emotion=fileNameSplit[-3]
                            if emotion not in EmotionalCSV.emotions.keys():
                                emotion=fileNameSplit[-2]
                                if emotion not in EmotionalCSV.emotions.keys():
                                    emotion=fileNameSplit[-1]
                    elif fileNameSplit[0]=='Session2':
                        actor=fileNameSplit[1].strip().split('-')[0]
                        emotion=fileNameSplit[-1]
                    emotion = emotion.lower()
                    if ((actor,emotion) in self.dicEmotion):
                        self.dicEmotion[(actor,emotion)]+=1
                    else :
                        self.dicEmotion[(actor,emotion)]=0
                    version=self.dicEmotion[(actor,emotion)]
                    
                    #create Bvh parser
                    anim = bvh_emotionalFormat.Bvh()
                    # parser file
                    anim.parse_file(action)
                    all_p, all_r = anim.all_frame_poses()
                    #print('Reading data from :',os.path.basename(action)+'   '+str(n)+'/'+str(len(FilesBVH)))
                    n+=1    
                    for i in range(all_p.shape[0]):
                        for j in range(all_p.shape[1]):
                            jointPos= np.round( all_p[i][j],6)
                            jointRot=np.round(all_r[i][j],7)
                            row=[i, jointPos[0],jointPos[1],jointPos[2],jointRot[1],jointRot[0],jointRot[2], j+1,EmotionalCSV.emotions[emotion], EmotionalCSV.subjects[actor], version ]
                            w = EmotionalCSV.sep.join(str(e) for e in row)+"\n"
                            myFile.write(w)
                else:
                    print('incorrect Data :',os.path.basename(action))
            myFile.close()

            
        elif (DBName==DataName.dance):  #ok
            "loads data from dance"
            print("------------- Dance DB")
            self.filePath= DirPaths.dance
            assert self.filePath[-1]=='/'
            n=1
            FilesBVH=glob.glob(self.filePath+'*.bvh')
            myFile=open(os.path.join(DirPaths.CSVdir,DanceCSV.fileName), "w")#création du csv
            headerLine = DanceCSV.sep.join(DanceCSV.header) + "\n" #remplissage de l'entete
            myFile.write(headerLine)
            for action in FilesBVH:
                fileNameSplit = os.path.basename(action).strip().split('.')[0].split('_') #s_emotion ou s_emotion_version pour dance contemporaines
                subject=fileNameSplit[0]
                if len(fileNameSplit)==2:
                    dance=fileNameSplit[-1]
                elif len(fileNameSplit)==3:
                    dance=fileNameSplit[-2]
                    
                if ((subject,dance) in self.dicDance):
                    self.dicDance[(subject,dance)]+=1
                else :
                    self.dicDance[(subject,dance)]=0
                version=self.dicDance[(subject,dance)]                
                print('Reading data from :',os.path.basename(action)+'   '+str(n)+'/'+str(len(FilesBVH)))
                n+=1
                parser = BVHParser()
                parsed_data = parser.parse(action)
                mp = bvh_preprocessing.MocapParameterizer('position')
                positions = mp.fit_transform([parsed_data])
                
                dataDF = positions[0].values
                #remove fingers : joint from 67 to 27
                dataDF = dataDF[dataDF.columns.drop(list(dataDF.filter(regex='Ring')))]
                dataDF = dataDF[dataDF.columns.drop(list(dataDF.filter(regex='Thumb')))]
                dataDF = dataDF[dataDF.columns.drop(list(dataDF.filter(regex='Middle')))]
                dataDF = dataDF[dataDF.columns.drop(list(dataDF.filter(regex='Pinky')))]
                dataDF = dataDF[dataDF.columns.drop(list(dataDF.filter(regex='Index')))]

                Nframes = dataDF.shape[0]
                Njoints = dataDF.shape[1]//3
                for i in range(Nframes):
                    #jointPos= np.round( all_p[i][j],6)
                    line= dataDF.iloc[i] 
                    for j in range(Njoints):
                        #header= ['Time','Xposition','Yposition','Zposition','Joint','DanceType','Subject','Version']
                        row=[i,line[j:j+3][0],line[j:j+3][1],line[j:j+3][2],j+1,DanceCSV.dance[dance],DanceCSV.subjects[subject],version]
                        w = DanceCSV.sep.join(str(e) for e in row)+"\n"
                        myFile.write(w)              
            myFile.close()


#a=DataLoader(DataName.emotional)
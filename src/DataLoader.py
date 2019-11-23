import os
import glob
import numpy


class DataSetFileName:  #for .dat file
    action="dataSetActionMsr3D.dat"
    dance="dataSetDance.dat"
    emotional="dataSetEmotional.dat"

class DataName:
    action="MSR Action3D"
    dance="Dance"
    emotional="Emotion"
    
class DirPaths:
    action='../data/MSRAction3D/'
    dance="../data/DanceDB/"
    emotional="../data/EmotionalBody/"
    CSVdir='../data/csv/'

class ActionCSV:
    fileName="Action3dData.csv"
    header= ['time','x','y','z','confidence','joint','action','subject','essai']
    sep=','
    
########################################
class EmotionalCSV:
    fileName="EmotionalData.csv"
    header= ['time','Xposition','Yposition','Zposition','Zrotation','Xrotation','Yrotation','Joint','Intended emotion','Subject']
    sep=';'
    
class DanceCSV:
    fileName="EmotionalData.csv"
    header= ['time','Xposition','Yposition','Zposition','Yrotation','Xrotation','Zrotation','Joint','Dance Type','Subject']
    sep=';'    
########################################

class DataLoader:
    "loads data"
    ## fixer la taille de la dataArray finale , pour chaque fichier , calcler au prealable la taille et enfin fusionner avec le grand Array
    def __init__(self, DBName,isCSV=False):
        self.filePath='/'
        self.arrayData=[]
        self.withCSV=isCSV
        
        if DBName==DataName.action:
            "loads data from MSR Action3D"    
            self.filePath= DirPaths.action
            self.arrayData = numpy.empty(shape=[1, len(ActionCSV.header)])
            assert self.filePath[-1]=='/'
            n=1
            FilesTXT=glob.glob(self.filePath+'*.txt')
            for action in FilesTXT:
                fileNameSplit = os.path.basename(action).strip().split('.')[0].split('_')[:3]    #['a01', 's01', 'e01']
                print(os.path.basename(action)+'   '+str(n)+'/'+str(len(FilesTXT)))
                with open(action,'r', encoding='utf-8') as file:
                    line = file.readline()
                    time=0
                    while line:
                        for i in range(1,21):                           
                            lineSplit=line.strip().split('  ')   #[1:].lstrip('0') pour tranformer 'a01' -> 1
                            self.arrayData = numpy.append(self.arrayData, [[time, lineSplit[0], lineSplit[1], lineSplit[2],lineSplit[3],i,fileNameSplit[0][1:].lstrip('0'), fileNameSplit[1][1:].lstrip('0'),fileNameSplit[2][1:].lstrip('0')]], axis=0)
                            line = file.readline()
                        time+=1
                n+=1
            self.arrayData = numpy.delete(self.arrayData, (0), axis=0)
            
        elif (DBName==DataName.emotional):
            "loads data from emotional"
            self.filePath= DirPaths.emotional
            self.arrayData = numpy.empty(shape=[1, len(EmotionalCSV.header)])
            assert self.filePath[-1]=='/'
            for action in glob.glob(self.filePath+'*.bvh'):
                fileNameSplit = os.path.basename(action).strip().split('.')[0].split('_')
                print(os.path.basename(action))
                with open(action,'r', encoding='utf-8') as file:
                    line=file.readline()
                    while (not line.startswith('Frame Time')):
                            if line.find('JOINT')!=-1:
                                joint=line.strip().split(' ')
                                self.jointArrayEmotional.append(joint[-1])
                            line=file.readline()
                    line = file.readline()
                    time=0
                    while line:
                        if (len(line.strip())!=0):
                            start=3
                            lineSplit=line.strip().split(' ')
                            for joint in range(0,len(self.jointArrayEmotional)):
                                rotationArray=[lineSplit[i] for i in range(start,start+3)]
                                if (self.jointArrayEmotional[joint]=='Hips'):
                                        Xpos=lineSplit[0]
                                        Ypos=lineSplit[1]
                                        Zpos=lineSplit[2]
                                else :
                                        Xpos=0
                                        Ypos=0
                                        Zpos=0
                                self.arrayData = numpy.append(self.arrayData,[[time,Xpos,Ypos,Zpos,rotationArray[0],rotationArray[1],rotationArray[2], self.jointArrayEmotional[joint],fileNameSplit[-1], fileNameSplit[0] ]], axis=0)
                                start+=3
                            time+=1
                            line = file.readline()
                            
        elif (DBName==DataName.dance):
            "loads data from dance"
            self.filePath= DirPaths.dance
            self.arrayData = numpy.empty(shape=[1, len(DanceCSV.header)])
            assert self.filePath[-1]=='/'
            for action in glob.glob(self.filePath+'*.bvh'):
                fileNameSplit = os.path.basename(action).strip().split('.')[0].split('_')
                print(os.path.basename(action))
                with open(action,'r', encoding='utf-8') as file:
                    line=file.readline()
                    while (not line.startswith('Frame Time')):
                            if line.find('JOINT')!=-1:
                                joint=line.strip().split(' ')
                                self.jointArrayDance.append(joint[-1])
                            line=file.readline()
                    line = file.readline()
                    time=0
                    while line:
                        if (len(line.strip())!=0):
                            start=0
                            lineSplit=line.strip().split(' ')
                            for joint in range(0,len(self.jointArrayDance)):
                                rotationArray=[lineSplit[i] for i in range(start,start+6)]
                                self.arrayData = numpy.append(self.arrayData,[[time,rotationArray[0],rotationArray[1],rotationArray[2],rotationArray[3],rotationArray[4],rotationArray[5], self.jointArrayDance[joint],fileNameSplit[1], fileNameSplit[0] ]], axis=0)
                                start+=3
                            time+=1
                            line = file.readline()
                            
        
        if(self.withCSV):
            if not os.path.exists(DirPaths.CSVdir):
                os.mkdir(DirPaths.CSVdir)
            print("creating "+ActionCSV.fileName+" ...")
            fichierCSV = open(os.path.join(DirPaths.CSVdir,ActionCSV.fileName), "w")
            headerLine = ActionCSV.sep.join(ActionCSV.header) + "\n"
            fichierCSV.write(headerLine)
            for line in self.arrayData:
                 ligne = ActionCSV.sep.join(line) + "\n"
                 fichierCSV.write(ligne)
            fichierCSV.close()
            print("creation OK")


            
#a=DataLoader("MSR Action3D", isCSV=True).arrayData

               
           

        
        
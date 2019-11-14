import os
import glob
import numpy
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
    #try plutot que essai
    header= ['time','x','y','z','joint','action','subject','try']
    sep=';'
class EmotionalCSV:
    fileName="EmotionalData.csv"
    #try plutot que essai
    header= ['time','Xposition','Yposition','Zposition','Zrotation','Xrotation','Yrotation','Joint','Intended emotion','Subject']
    sep=';'
class DanceCSV:
    fileName="EmotionalData.csv"
    #try plutot que essai
    header= ['time','Xposition','Yposition','Zposition','Yrotation','Xrotation','Zrotation','Joint','Dance Type','Subject']
    sep=';'


class DataLoader:
    "loads data"
    def __init__(self, DBName,isCSV=False):
        self.filePath='/'
        self.arrayData=[]
        self.withCSV=isCSV
        self.jointArrayEmotional=[]
        self.jointArrayDance=[]
        self.jointArrayAction=['Right Shoulder','Left Shoulder','Neck','Spine','Right Hip','Left Hip','Middle Hip','Right Elbow','Left Elbow','Right Wrist','Left Wrist','Right Hand','Left Hand','Right Knee','Left Knee','Right Ankle','Left Ankle','Right Foot','Left Foot','Head']

        #rajout de parenthèses
        if (DBName==DataName.action):
            "loads data from MSR Action3D"
            self.filePath= DirPaths.action
            self.arrayData = numpy.empty(shape=[1, len(ActionCSV.header)])
            assert self.filePath[-1]=='/'
            for action in glob.glob(self.filePath+'*.txt'):
                fileNameSplit = os.path.basename(action).strip().split('.')[0].split('_')[:3]    #['a01', 's01', 'e01']
                print(os.path.basename(action))
                with open(action,'r', encoding='utf-8') as file:
                    line = file.readline()
                    time=0
                    while line:
                        #joint plutot que i
                        for joint in range(0,len(self.jointArrayAction)):
                            lineSplit=line.strip().split('  ')
                            self.arrayData = numpy.append(self.arrayData, [[time, lineSplit[0], lineSplit[1], lineSplit[2],self.jointArrayAction[joint],fileNameSplit[0], fileNameSplit[1],fileNameSplit[2]]], axis=0)
                            line = file.readline()
                        time+=1

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

            self.arrayData = numpy.delete(self.arrayData, (0), axis=0)#delete first row
            if(self.withCSV):
                #rajout de commentaire
                "Create CSV"
                if not os.path.exists(DirPaths.CSVdir):
                    os.mkdir(DirPaths.CSVdir)
                print("creating "+ActionCSV.fileName+" ...")
                #file csv
                fileCSV = open(os.path.join(DirPaths.CSVdir,ActionCSV.fileName), "w")
                headerLine = ";".join(ActionCSV.header) + "\n"
                fileCSV.write(headerLine)
                for line in self.arrayData:
                     #line plutot que ligne
                     line = ";".join(line) + "\n"
                     fileCSV.write(line)
                fileCSV.close()

a=DataLoader("Dance", isCSV=False).arrayData






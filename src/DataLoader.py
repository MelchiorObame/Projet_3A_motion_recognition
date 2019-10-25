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
    header= ['time','x','y','z','joint','action','subject','essai']
    sep=';'

class DataLoader:
    "loads data"
    
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
            for action in glob.glob(self.filePath+'*.txt'):
                fileNameSplit = os.path.basename(action).strip().split('.')[0].split('_')[:3]    #['a01', 's01', 'e01']
                print(os.path.basename(action)+'   '+str(n)+'/567')
                with open(action,'r', encoding='utf-8') as file:
                    line = file.readline()
                    time=0
                    while line:
                        for i in range(1,21):                           
                            lineSplit=line.strip().split('  ')
                            self.arrayData = numpy.append(self.arrayData, [[time, lineSplit[0], lineSplit[1], lineSplit[2],i,fileNameSplit[0], fileNameSplit[1],fileNameSplit[2]]], axis=0)
                            line = file.readline()
                        time+=1
                n+=1
            self.arrayData = numpy.delete(self.arrayData, (0), axis=0)
            
            if(self.withCSV):
                if not os.path.exists(DirPaths.CSVdir):
                    os.mkdir(DirPaths.CSVdir)
                print("creating "+ActionCSV.fileName+" ...")
                fichierCSV = open(os.path.join(DirPaths.CSVdir,ActionCSV.fileName), "w")
                headerLine = ";".join(ActionCSV.header) + "\n"
                fichierCSV.write(headerLine)
                for line in self.arrayData:
                     ligne = ";".join(line) + "\n"
                     fichierCSV.write(ligne)
                fichierCSV.close()



            
            
a=DataLoader("MSR Action3D", isCSV=True).arrayData

               
           

        
        
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
    header= ['time','x','y','z','confidence','joint','action','subject','essai']
    sep=','

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

               
           

        
        
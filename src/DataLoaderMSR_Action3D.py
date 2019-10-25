import os
import numpy

class DataName:
    action="MSR Action3D"
    dance="Dance"
    emotional="Emotion"
    
class DirPaths:
    action='../data/MSRAction3D/'
    dance="../data/DanceDB/"
    emotional="../data/EmotionalBody/"


class DataLoader:
    "loads data from MSR Action3D"
    
    def __init__(self, DBName):
        self.filePath='/'
        
        if DBName==DataName.action:
           self.filePath= DirPaths.action
           assert self.filePath[-1]=='/'
           arrayData = numpy.empty(shape=[1, 8])
           for action in os.listdir(DirPaths.action):#self.filePath
               #get name without extension
               basenameSplit = action.strip().split('.')#get (.txt)
               #prendre queles .txt
               assert basenameSplit[1]=='txt'
               #get in this order action, subject, essay
               fileNameSplit = basenameSplit[0].split('_')
               print(os.path.basename(action))
               #open file
               with open(DirPaths.action+action,'r', encoding='utf-8') as file:
                   time=-1
                   line = file.readline()                   
                   while line:
                       time=time+1
                       for i in range(1,21):                           
                           lineSplit=line.strip().split('  ')
                           arrayData = numpy.append(arrayData, [[time, i, lineSplit[0], lineSplit[1], lineSplit[2], fileNameSplit[0], fileNameSplit[2], fileNameSplit[1] ]], axis=0)
                           line = file.readline()
               arrayData = numpy.delete(arrayData, (0), axis=0)#delete first row
DataLoader("MSR Action3D")


               
           

        
        
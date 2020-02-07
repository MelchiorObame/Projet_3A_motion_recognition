import numpy as np
from PIL import Image, ImageFilter
import os, pandas as pd, sys, shutil
import matplotlib as plt


import distance_measure as dm
import DataLoader
import reshapeDataToMatrix


#--------- encoding to RGB :

def getPictureDataSet(dataName,minNFrames=30, labelTarget=None,normalizeData=False, size=(70,70)): #OK
    """ creates resized pictures 
    Arguments : size = size of our output picture """
    #create picture folder if not exists
    dirPath= DataLoader.DirPaths.PictureDataSet
    if dataName == DataLoader.DataName.action:
        DataSetpath= os.path.join(dirPath, DataLoader.ActionCSV.pictureActionDBDirName)
        df = pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.ActionCSV.fileName))
        maxX, maxY, maxZ= df.loc[df['x'].idxmax()]['x'], df.loc[df['y'].idxmax()]['y'], df.loc[df['z'].idxmax()]['z']
        minX, minY, minZ= df.loc[df['x'].idxmin()]['x'], df.loc[df['y'].idxmin()]['y'], df.loc[df['z'].idxmin()]['z']
    if dataName == DataLoader.DataName.dance:
        DataSetpath= os.path.join(dirPath, DataLoader.DanceCSV.pictureDanceDBDirName)
        df=pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.DanceCSV.fileName))
    if dataName == DataLoader.DataName.emotional:
        DataSetpath= os.path.join(dirPath, DataLoader.EmotionalCSV.pictureEmotionalDBPicture)   
        df= pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.EmotionalCSV.fileName))
    dirValidation = os.path.join(DataSetpath,'Validation')
    dirTrain = os.path.join(DataSetpath,'Train')
    if  os.path.exists(DataSetpath):
        shutil.rmtree(DataSetpath)
    os.makedirs(dirValidation)
    os.makedirs(dirTrain)
    print('loading Datas...')
    X_train, actionLabelsTrain, subjectLabelsTrain, X_test, actionLabelsTest, subjectLabelsTest = reshapeDataToMatrix.train_test_split(dataName, normalizeData=normalizeData, labelTarget=labelTarget)
    X_train = X_train
    X_test = X_test
    print('Train Pictures... ')
    for i in range(len(X_train)):
        if dataName == DataLoader.DataName.action:
            action1 =np.delete(X_train[i],3, axis=2)[:minNFrames]
            action = normalizeDataRGB(dataName,action1 ,minX, maxX, minY, maxY,minZ, maxZ)
            action = action.reshape(action1.shape)            
        elif dataName == DataLoader.DataName.dance:
            action = X_train[i][:minNFrames]   
        elif dataName == DataLoader.DataName.emotional:
            action = X_train[i][:minNFrames]
        name= 'a'+str(actionLabelsTrain[i])+'_s'+str(subjectLabelsTrain[i])+'_'+str(i)
        subDir=str(actionLabelsTrain[i])
        if labelTarget.lower()=='subject':
            subDir=str(subjectLabelsTrain[i])
        savePicture(action, name,dirTrain,size, subDir)
    print('Validation Pictures... ')
    for i in range(len(X_test)):
        if dataName == DataLoader.DataName.action:
            action1 =np.delete(X_test[i],3, axis=2)[:minNFrames]
            action = normalizeDataRGB(dataName,action1 ,minX, maxX, minY, maxY,minZ, maxZ)
            action = action.reshape(action1.shape)            
        elif dataName == DataLoader.DataName.dance:
            action = X_test[i][:minNFrames]   
        elif dataName == DataLoader.DataName.emotional:
            action = X_test[i][:minNFrames]
        name= 'a'+str(actionLabelsTest[i])+'_s'+str(subjectLabelsTest[i])+'_'+str(i)
        subDir=str(actionLabelsTest[i])
        if labelTarget.lower()=='subject':
            subDir=str(subjectLabelsTest[i])
        savePicture(action, name,dirValidation,size,subDir)
        
        
def savePicture(action, name,directory,size, subDir):
    path = os.path.join(directory ,subDir)
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        img = Image.fromarray(action, 'RGB').resize(size, Image.ANTIALIAS)
        img=img.rotate(90)
        img.save(os.path.join(path ,name)+'.png')
    except IOError:
        print('cannot create thumbnail for')


def normalizeDataRGB(dataName, action ,minX, maxX, minY, maxY,minZ, maxZ):
    """If normalisation is not used before.  return a numpy array reshaped where every coordinates have been normalized.
    And reshape to be used to image conversion ( transpose )"""
    if dataName==DataLoader.DataName.action:
        NJoints=  dm.Data.NJoints
        Ncoord = dm.Data.NcoordinatesOfJoint-1
    elif dataName==DataLoader.DataName.dance:
        NJoints=  dm.DataDanceInfo.NJoints
        Ncoord = dm.DataDanceInfo.NcoordinatesOfJoint
    elif dataName==DataLoader.DataName.emotional:
        NJoints=  dm.DataEmotionalInfo.NJoints
        Ncoord = dm.DataEmotionalInfo.NcoordinatesOfJoint
    result = np.zeros((action.shape[0],NJoints, Ncoord))
    for j in range(len(action)):
        for joint in range(NJoints):
            result[j][joint][0] = 255*( (action[j][joint][0]-minX)/(maxX-minX))
            result[j][joint][1] = 255*( (action[j][joint][1]-minY)/(maxY-minY))
            result[j][joint][2] = 255*( (action[j][joint][2]-minZ)/(maxZ-minZ))
    return result.transpose(1,0,2)

### -------------- TEST : 1  : All tests work. 
#getPictureDataSet(DataLoader.DataName.action,labelTarget='subject',minNFrames=30) #ok
getPictureDataSet(DataLoader.DataName.dance,  labelTarget='subject',minNFrames=800) #ok
#getPictureDataSet(DataLoader.DataName.emotional,labelTarget='subject',minNFrames=600) #ok

#--------- SPMF : Skeleton Pose-Motion Feature
#determination de PF : In the same Frame

def JJD(dataName, j1,j2): #ok
    """ euclidian distance between two vectors/joint of the same frame (j1 != j2)"""
    if dataName == DataLoader.DataName.action:
        return dm.euclid_distance_joints(j1,j2)
    elif dataName == DataLoader.DataName.dance:
        #return dm.distanceJointWithRotationDance(j1, j2)
        #return dm.distanceJointWithRotationDance3(j1, j2)
        return dm.euclid_distance_joints(j1,j2)
    elif dataName == DataLoader.DataName.emotional:
        return dm.distanceJointWithRotationEmotional(j1, j2)
    
    
def JJO(dataName, j1,j2, minX, maxX, minY, maxY,minZ, maxZ): #OK   juste pour action en ce ce moment, peuvent etre negatives c'est pas un pb -1=255
    """Joint-Joint Orientation between two position vectors j1 and j2 
        Returns an 3D unit vector of same shape as joints (x,y,z), normalized in [0, 255] JJO_jk_t"""
    #distance = JJD(dataName, j1,j2)
    if dataName == DataLoader.DataName.action: #ok , #les actions n'avaient pas déja été normalisées 
        result = (np.array(j1)-np.array(j2) )#/distance
        #normalize coordinates
        result[0]=int(255*( (result[0]-minX)/(maxX-minX)))   #x
        result[1]=int(255*( (result[1]-minY)/(maxY-minY)))   #y
        result[2]=int(255*( (result[2]-minZ)/(maxZ-minZ)))   #z
        return result[:3].tolist()     #take just x,y and z
    elif dataName == DataLoader.DataName.dance:
        result = (np.array(j1)-np.array(j2) )
        a,b,c =result[0],result[1], result[2]
        result[0]=int(255*( (result[0]-minX)/(maxX-minX)))   #x
        result[1]=int(255*( (result[1]-minY)/(maxY-minY)))   #y
        result[2]=int(255*( (result[2]-minZ)/(maxZ-minZ)))   #z
        if(result[0] ==sys.maxsize or result[0] ==sys.maxsize or result[0] ==sys.maxsize ):
            print(a,b,c)
        return result.tolist()     #take just x,y and z
    elif dataName == DataLoader.DataName.emotional:
        #Xposition','Yposition','Zposition','Yrotation','Xrotation','Zrotation'
        X1, X2 = j1[0]+j1[4], j2[0]+j2[4]
        Y1, Y2 = j1[1]+j1[3], j2[1]+j2[3]
        Z1, Z2 = j1[2]+j1[5], j2[2]+j2[5]
        j1, j2 =np.array([X1, Y1, Z1]), np.array([X2, Y2, Z2])
        result = j1-j2
        result[0]=int(255*( (result[0]-minX)/(maxX-minX)))   #x
        result[1]=int(255*( (result[1]-minY)/(maxY-minY)))   #y
        result[2]=int(255*( (result[2]-minZ)/(maxZ-minZ)))   #z
        return result.tolist()

        



def JJD_RGB(dataName, frame): #ok
    """ 3D array of shape [NJoint, NJoint-1,3] that is formed by all JJD values .between two distinct joints
    simple matrix of distance """
    result = np.empty(0)
    if dataName == DataLoader.DataName.action: #ok , shape (20, 19, 3)
        njoints = dm.Data.NJoints
    elif dataName == DataLoader.DataName.dance:
        njoints = dm.DataDanceInfo.NJoints
    elif dataName == DataLoader.DataName.emotional:
        njoints = dm.DataEmotionalInfo.NJoints
    for j in range(njoints): #joints
        joint=frame[j]
        for i in range(njoints):#computes jjo between two distinct joints and normalize in [0, 255]
            if i!=j :
                result=np.append(result,JJD(dataName, joint, frame[i]))
    result = result.reshape( njoints, njoints-1 )
    ### convert to [0,1] 
    maxX = np.max(result)
    minX = np.min(result)
    result = (result -minX) /(maxX - minX)
    #convert to RGB [0, 255] using JET scale. color palette called JET color map
    resultRGB = np.delete(plt.cm.jet(result),3, axis=2)*255 # remove transparancy column and convert in [0 255]
    return resultRGB.astype(np.int64)
    
    

def JJO_RGB(dataName, frame, minX, maxX, minY, maxY,minZ, maxZ): #ok ppr action
    """ 3D array of shape [NJoint, NJoint-1, 3] that is formed by all JJO values .between two distinct joints"""   
    result = np.empty(0) # we will reshape
    ncoordinates = dm.Data.NcoordinatesOfJoint
    if dataName == DataLoader.DataName.action: #ok , shape (20, 19, 3)
        njoints = dm.Data.NJoints
    elif dataName == DataLoader.DataName.dance:
        njoints = dm.DataDanceInfo.NJoints
    elif dataName == DataLoader.DataName.emotional:
        njoints = dm.DataEmotionalInfo.NJoints
    for j in range(njoints): #joints
        joint=frame[j]
        for i in range(njoints):#computes jjo between two distinct joints and normalize in [0, 255]
            if j!=i :
                result= np.append(result, JJO(dataName, joint, frame[i],minX, maxX, minY, maxY,minZ, maxZ))
    return result.reshape(njoints, njoints-1, ncoordinates-1).astype(np.int64)
    
    
#determination de MF Motion feature : In succesive frame
        
def MF_JJD_RGB(dataName, f1, f2): #ok pour action
    """ Computes JJD_RGB between two consecutives frames f1 compared to f2 
        Returns an array of shape [NJoint, NJoint-1,3] """
    result = np.empty(0)
    if dataName == DataLoader.DataName.action: #ok , shape (20, 19, 3)
        njoints = dm.Data.NJoints
    elif dataName == DataLoader.DataName.dance:
        njoints = dm.DataDanceInfo.NJoints
    elif dataName == DataLoader.DataName.emotional:
        njoints = dm.DataEmotionalInfo.NJoints
    for j in range(njoints): #joints
        for i in range(njoints):
            if j!=i :
                result=np.append(result, JJD(dataName, f1[j], f2[i]))
    result = result.reshape( njoints, njoints-1 )
    ### convert to [0,1] 
    maxX = np.max(result)
    minX = np.min(result)
    result = (result -minX) /(maxX - minX)
    #convert to RGB [0, 255] using JET scale. color palette called JET color map
    resultRGB = np.delete(plt.cm.jet(result),3, axis=2)*255 # remove transparancy column and convert in [0 255]
    return resultRGB.astype(np.int64)
    

def MF_JJO_RGB(dataName, f1,f2, minX, maxX, minY, maxY,minZ, maxZ): #ok pour action
    """ Computes JJO_RGB between two consecutives frames f1 compared to f2 
        Returns an array of shape [NJoint, NJoint-1,3]"""
    result = np.empty(0)
    ncoordinates = dm.Data.NcoordinatesOfJoint
    if dataName == DataLoader.DataName.action: #ok , shape (20, 19, 3)
        njoints = dm.Data.NJoints
    elif dataName == DataLoader.DataName.dance:
        njoints = dm.DataDanceInfo.NJoints
    elif dataName == DataLoader.DataName.emotional:
        njoints = dm.DataEmotionalInfo.NJoints
    for j in range(njoints): #joints
        for i in range(njoints):#computes jjo between two distinct joints and normalize in [0, 255]
            if j!=i :
                result= np.append(result, JJO(dataName, f1[j], f2[i],minX, maxX, minY, maxY,minZ, maxZ) )
    return result.reshape(njoints, njoints-1, ncoordinates-1).astype(np.int64)


def PF(dataName, frame, minX, maxX, minY, maxY,minZ, maxZ ): #ok pr action
    """" Position feature of skeleton: makes concatenation of the tensors
        Return 3D array of shape (NJoint,(Njoint-1)*2,3)"""
    jjd_rgb = JJD_RGB(dataName, frame)
    jjo_rgb = JJO_RGB(dataName, frame, minX, maxX, minY, maxY,minZ, maxZ)
    return np.concatenate((jjd_rgb, jjo_rgb), axis= 1)


def MF(dataName, f, fnext, minX, maxX, minY, maxY,minZ, maxZ): #ok pour action
    """" motion feature of skeleton: makes concatenation of the tensors between consecutives frames .p=k+1 
        Returns : of shape (NJoint,(Njoint-1)*2,3)
        Arguments : f : current frame, fnext : next frame """
    mf_jjd_rgb = MF_JJD_RGB(dataName, f, fnext)
    mf_jjo_rgb = MF_JJO_RGB(dataName, f, fnext, minX, maxX, minY, maxY,minZ, maxZ)
    return np.concatenate((mf_jjd_rgb, mf_jjo_rgb), axis= 1)


def SPMF(dataName, action,minX, maxX, minY, maxY,minZ, maxZ): #ok for action
    '''Builds Global Action Map from PFs and MFs, concatenate  all PFs and MFs
        from the skeleton sequence  '''
    sumSPMF = np.empty(len(action)).tolist()
    for i in range(len(action)):
        #print('frame '+str(i+1)+'/'+str(len(action)))
        pf = PF(dataName, action[i] , minX, maxX, minY, maxY,minZ, maxZ )
        if i!=len(action)-1:
            frameNext=action[i+1]
            mf = MF(dataName, action[i] , frameNext, minX, maxX, minY, maxY,minZ, maxZ)
            sumSPMF[i]= np.concatenate((pf, mf), axis= 1)
        else:
            sumSPMF[i]=pf
    spmf=sumSPMF[0]
    for i in range(1,len(action)):
        spmf=np.concatenate((spmf,sumSPMF[i]), axis= 1)
    return spmf



def filteringSavitzkyGolay(frame, t):#ok
    """ smoothing filter used to reduce the effect of noise on skeletal data
        t : frame number [0,N], it's a low-pass filter  """
    frame = np.array(frame)
    result = ( -3*(frame**(t-2)) + 12*(frame**(t-1))+ 17*(frame**t)+ 12*(frame**(t+1)) - 3*(frame**(t+2)) )/35
    return result

def compareMin(PrevA,NewA):#ok
    if PrevA>NewA:
        return NewA
    return PrevA

def compareMax(PrevA,NewA):#ok
    if PrevA<NewA:
        return NewA
    return PrevA
    

def applySGtoDataSet(dataName,datas, minX, maxX, minY, maxY,minZ, maxZ): #  #ok
    """ apply SG filter to all data Set
        Returns : datas : all datas and also minX, maxX, minY, maxY,minZ, maxZ after the applying of filter """
    minXResult, maxXResult, minYResult, maxYResult,minZResult, maxZResult = sys.maxsize,0,sys.maxsize,0,sys.maxsize,0
    if dataName == DataLoader.DataName.action: #ok , shape (20, 19, 3)
        njoints = dm.Data.NJoints
        ncoordinates = dm.Data.NcoordinatesOfJoint
    elif dataName == DataLoader.DataName.dance:
        njoints = dm.DataDanceInfo.NJoints
        ncoordinates = dm.DataDanceInfo.NcoordinatesOfJoint
    elif dataName == DataLoader.DataName.emotional:
        njoints = dm.DataEmotionalInfo.NJoints
        ncoordinates = dm.DataEmotionalInfo.NcoordinatesOfJoint
    for i in range(len(datas)):
        for j in range(len(datas[i])): #est déja entre 0,1
            for k in range(njoints): #first, normalize in [0,1]
                datas[i][j][k][0] = ( datas[i][j][k][0] - minX)/(maxX-minX) #x
                datas[i][j][k][1] = ( datas[i][j][k][1] - minY)/(maxY-minY) #y
                datas[i][j][k][2] = ( datas[i][j][k][2] - minZ)/(maxZ-minZ) #z
            datas[i][j] = filteringSavitzkyGolay(datas[i][j],j)
        action = datas[i].reshape((datas[i].shape[0]*njoints, ncoordinates))
        _maxX, _maxY, _maxZ = np.amax(action, axis=0)[:3]
        _minX, _minY, _minZ = np.amin(action, axis=0)[:3]
        minXResult = compareMin(minXResult,_minX)
        minYResult = compareMin(minYResult,_minY)
        minZResult = compareMin(minZResult,_minZ)
        maxXResult = compareMax(maxXResult,_maxX)
        maxYResult = compareMax(maxYResult,_maxY)
        maxZResult = compareMax(maxZResult,_maxZ)     
    return datas, minXResult, maxXResult, minYResult, maxYResult,minZResult, maxZResult
            
    

def dataSetSPMF_Pictures(dataName, normalizeData=False, size=(320,320), SGfilter=False):
    """ Create data for SMPF methodes """
    dirPath= DataLoader.DirPaths.PictureDataSet
    print('CSV loading...')
    if dataName == DataLoader.DataName.action:
        
        DataSetpath= os.path.join(dirPath, DataLoader.ActionCSV.pictureActionDBDirNameSPMF)
        df = pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.ActionCSV.fileName))
        #appliquer le filtre sur toute la df et retirer les max et min 
        maxX, maxY, maxZ = df.loc[df['x'].idxmax()]['x'], df.loc[df['y'].idxmax()]['y'], df.loc[df['z'].idxmax()]['z']
        minX, minY, minZ = df.loc[df['x'].idxmin()]['x'], df.loc[df['y'].idxmin()]['y'], df.loc[df['z'].idxmin()]['z']
    if dataName == DataLoader.DataName.dance:
        DataSetpath= os.path.join(dirPath, DataLoader.DanceCSV.pictureDanceDBDirNameSPMF)
        df=pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.DanceCSV.fileName))
        maxX, maxY, maxZ = df.loc[df['Xposition'].idxmax()]['Xposition'], df.loc[df['Yposition'].idxmax()]['Yposition'], df.loc[df['Zposition'].idxmax()]['Zposition']
        minX, minY, minZ = df.loc[df['Xposition'].idxmin()]['Xposition'], df.loc[df['Yposition'].idxmin()]['Yposition'], df.loc[df['Zposition'].idxmin()]['Zposition']
    if dataName == DataLoader.DataName.emotional:
        DataSetpath= os.path.join(dirPath, DataLoader.EmotionalCSV.pictureEmotionalDBPictureSPMF)
        df= pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.EmotionalCSV.fileName))
        maxX, maxY, maxZ = df.loc[df['Xposition'].idxmax()]['Xposition'], df.loc[df['Yposition'].idxmax()]['Yposition'], df.loc[df['Zposition'].idxmax()]['Zposition']
        minX, minY, minZ = df.loc[df['Xposition'].idxmin()]['Xposition'], df.loc[df['Yposition'].idxmin()]['Yposition'], df.loc[df['Zposition'].idxmin()]['Zposition']
    if not os.path.exists(DataSetpath):
        os.makedirs(DataSetpath)
    datas, actions, subjects = reshapeDataToMatrix.loadDataSet(dataName, normalizeData=normalizeData)
    datas = datas.tolist()
    print('Creating picture...')
    if SGfilter: 
        datas, minX, maxX, minY, maxY,minZ, maxZ = applySGtoDataSet(dataName,datas,minX, maxX, minY, maxY,minZ, maxZ)
        print(' Savitzky-Golay smoothing filter applied ')
    print(minX, maxX, minY, maxY,minZ, maxZ) 
    for i in range(len(actions)): #actions
        action = SPMF(dataName, np.array(datas[i]), minX, maxX, minY, maxY,minZ, maxZ)
        #return action[0]
        name= 'a'+str(actions[i])+'_s'+str(subjects[i])+'_'+str(i)
        print('picture : '+str(i+1)+'/'+str(len(actions)) +' : OK ')
        try: #resizing and saving
            img = Image.fromarray(action, 'RGB').resize(size, Image.ANTIALIAS)
            if not SGfilter:
                img = img.filter(ImageFilter.GaussianBlur(radius=1))
            img.save(os.path.join(DataSetpath ,name)+'.png')
        except IOError:
            print('cannot create thumbnail for')
    print('Create picture : OK')
    
    

    
    
    
    
##---- test SMPF 
#j1=[100, 12, 25, 4, 3, 2]
#j2=[10, 102, 3, 5, 34, 78]
#normalizeJointCoordinate=normalizeJointCoordinate(j1, 5,190, 2, 100, 23, 50000)
#jjd=JJD(DataLoader.DataName.dance, j1, j2)
#jjo=JJO(DataLoader.DataName.dance, j1, j2)

# ok
#minX, maxX, minY, maxY,minZ, maxZ= 0.0, 76.0, 0.0, 2147480000.0, 0.0, 970.0
#j1=[1.6200000e+002,1.4800000e+002,6.5300000e+002,7.4519200e-001]
#j2=[1.3900000e+002,7.5000000e+001,6.4600000e+002,0.0000000e+000]
#jjo = JJO(DataLoader.DataName.action, j1,j2, minX, maxX, minY, maxY,minZ, maxZ)

#f1=np.random.uniform(0,2140,(550,4))
#f2=np.random.uniform(0,2140,(550,4))

#jjd_rgb = JJD_RGB(DataLoader.DataName.action, f1)
#mf_jjd_rgb = MF_JJD_RGB(DataLoader.DataName.action, f1, f2)

#maxX, maxY, maxZ,_ = np.amax(f1, axis=0)
#minX, minY, minZ,_ = np.amin(f1, axis=0)
#jjo_rgb = JJO_RGB(DataLoader.DataName.action, f1, minX, maxX, minY, maxY,minZ, maxZ)
#mf_jjo_rgb= MF_JJO_RGB(DataLoader.DataName.action, f1,f2, minX, maxX, minY, maxY,minZ, maxZ)
#pf= PF(DataLoader.DataName.action, f1, minX, maxX, minY, maxY,minZ, maxZ)
#mf= MF(DataLoader.DataName.action, f1, f2, minX, maxX, minY, maxY,minZ, maxZ)


#df = pd.read_csv(os.path.join(DataLoader.DirPaths.CSVdir, DataLoader.ActionCSV.fileName))
#a=reshapeDataToMatrix.getAction(7,4,1, df)
#a2=reshapeDataToMatrix.getAction(1,1,1, df)
#print('-------- prog -------')
#action = np.random.uniform(0,2140,(546,20,4))
#maxX, maxY, maxZ,_ = np.amax(f1, axis=0)
#minX, minY, minZ,_ = np.amin(f1, axis=0)
#spmf =SPMF(DataLoader.DataName.action, a1)

#action 168 a un probleme car il y'a NaN dedans 
#a=dataSetSPMF_Pictures(DataLoader.DataName.dance , SGfilter=True)
    
#datas, actions, subjects = reshapeDataToMatrix.loadDataSet(DataLoader.DataName.action, normalizeData=False)
#datas = datas.tolist()
#applySGtoDataSet = applySGtoDataSet(DataLoader.DataName.action,datas)
import numpy as np
import math


########################### CLASSES ##########   
class Data:
    NJoints=20
    NcoordinatesOfJoint=4
    Naction=567

class DataDanceInfo:
    NJoints=27
    NcoordinatesOfJoint=3
    Naction=134
    
class DataEmotionalInfo:
    NJoints=28
    NcoordinatesOfJoint=6
    Naction=1447
    
############################# For MSR ACTION 3D        

def euclid_distance_joints(j1, j2): #OK  en supposant que l'origine est la mm pour les 2
    """ euclidian distance between two vectors/joint 
        A joint has 3 cordinates J(x,y,z), array [x,y,z,c] pf shape [4] """
    joint1, joint2=np.array(j1), np.array(j2)
    return math.sqrt( (joint1[0]-joint2[0])**2 + (joint1[1]-joint2[1])**2 + (joint1[2]-joint2[2])**2)


def thresholdConfidence(confidence, tho): #OK
    "computes Wjf the weight of confidence c of a joint : 1 if c<tho 0 otherwise"
    if confidence >= tho:
        return 1
    return 0
    
def distance_frames(f1,f2,  tho,confidence=False):  #OK
    """ distance between Frames list of joints
    f1,f2 array of shape [NJoint=20,nCollumn=4]  matrix
    J_prime : nombre de joints communs non nul dans les deux frames """
    frame1, frame2 =np.array(f1), np.array(f2)
    somme=0
    if not confidence:
        for j in range(Data.NJoints): 
            somme+= euclid_distance_joints(frame1[j], frame2[j] )
        return somme
    else:
        #calcul de J_prime et des poids de confiance
        J_prime=0
        for i in range(Data.NJoints):
            if(thresholdConfidence( frame1[i,Data.NcoordinatesOfJoint-1], tho)==1 and thresholdConfidence(frame2[i,Data.NcoordinatesOfJoint-1], tho)==1):
                J_prime+=1
        #calcul de la distance
        for j in range(Data.NJoints):
            somme+= euclid_distance_joints(frame1[j], frame2[j]) *( thresholdConfidence(frame1[j,Data.NcoordinatesOfJoint-1],tho)*thresholdConfidence(frame2[j,Data.NcoordinatesOfJoint-1], tho))    
        if J_prime==0 :
            return somme/(J_prime+0.000001)
        else:
            return somme/J_prime
        

def distance_framesWithRotationDance(f1, f2): #ok
    """ distance between Frames list of joints
    f1,f2 array of shape [NJoint=55,nCollumn = (NcoordinatesOfJoint=6)]  matrix """
    frame1, frame2 =np.array(f1), np.array(f2)
    sumDist=0
    for j in range(DataDanceInfo.NJoints):
        sumDist+= euclid_distance_joints(frame1[j],frame2[j])
    return sumDist


################## Emotional    
def distanceJointWithRotationEmotional(j1,j2):#test 2
    """ distance between two vectors/joint 
        A joint has 6 cordinates J([x,y,z,RotY, RotX, RotZ), array [x,y,z,RotY, RotX, RotZ] pf shape [6] """
    return euclid_distance_joints(j1[:3], j2[:3])

def distance_framesWithRotationEmotional(f1, f2): #probleme  
    """ distance between Frames list of joints
    f1,f2 array of shape [NJoint=23,nCollumn= (NcoordinatesOfJoint=6)]  matrix """
    frame1, frame2 =np.array(f1), np.array(f2)
    sumDist=0
    for j in range(DataEmotionalInfo.NJoints):
        sumDist+=  euclid_distance_joints(frame1[j][:3],frame2[j][:3])
    return sumDist

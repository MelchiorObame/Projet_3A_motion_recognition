'''
Preprocessing Tranformers Based on sci-kit's API
'''

import pandas as pd
import math
import numpy as np
from sklearn.base import TransformerMixin
import matplotlib.pyplot as plt


class MocapParameterizer( TransformerMixin):
    def __init__(self, param_type = 'euler'):
        '''
        param_type = {'euler'}
        '''
        self.param_type = param_type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):       
        return self._to_pos(X)


    def _to_pos(self, X):
        '''Converts joints rotations in Euler angles to joint positions'''
        Q = []
        for track in X:
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            pos_df = pd.DataFrame(index=euler_df.index)

            # List the columns that contain rotation channels
            rot_cols = [c for c in euler_df.columns if ('rotation' in c)]

            # List the columns that contain position channels
            pos_cols = [c for c in euler_df.columns if ('position' in c)]

            # List the joints that are not end sites, i.e., have channels
            #joints = (joint for joint in track.skeleton)
            
            tree_data = {}

            for joint in track.traverse():
                parent = track.skeleton[joint]['parent']
                # Get the rotation columns that belong to this joint
                rc = euler_df[[c for c in rot_cols if joint in c]]

                # Get the position columns that belong to this joint
                pc = euler_df[[c for c in pos_cols if joint in c]]

                # Make sure the columns are organized in xyz order
                if rc.shape[1] < 3:
                    euler_values = [[0,0,0] for f in rc.iterrows()]
                else:
                    euler_values = [[f[1]['%s_Xrotation'%joint], 
                                     f[1]['%s_Yrotation'%joint], 
                                     f[1]['%s_Zrotation'%joint]] for f in rc.iterrows()]

                if pc.shape[1] < 3:
                    pos_values = [[0,0,0] for f in pc.iterrows()]
                else:
                    pos_values =[[f[1]['%s_Xposition'%joint], 
                                  f[1]['%s_Yposition'%joint], 
                                  f[1]['%s_Zposition'%joint]] for f in pc.iterrows()]
                
                # Convert the eulers to rotation matrices
                rotmats = np.asarray([Rotation([f[0], f[1], f[2]], from_deg=True).rotmat for f in euler_values])                  
                tree_data[joint]=[
                                    [], # to store the rotation matrix
                                    []  # to store the calculated position
                                 ] 

                if track.root_name == joint:
                    tree_data[joint][0] = rotmats
                    # tree_data[joint][1] = np.add(pos_values, track.skeleton[joint]['offsets'])
                    tree_data[joint][1] = pos_values
                else:
                    # for every frame i, multiply this joint's rotmat to the rotmat of its parent
                    tree_data[joint][0] = np.asarray([np.matmul(rotmats[i], tree_data[parent][0][i]) 
                                                      for i in range(len(tree_data[parent][0]))])

                    # add the position channel to the offset and store it in k, for every frame i
                    k = np.asarray([np.add(pos_values[i],  track.skeleton[joint]['offsets'])
                                    for i in range(len(tree_data[parent][0]))])

                    # multiply k to the rotmat of the parent for every frame i
                    q = np.asarray([np.matmul(k[i], tree_data[parent][0][i]) 
                                    for i in range(len(tree_data[parent][0]))])

                    # add q to the position of the parent, for every frame i
                    tree_data[joint][1] = np.asarray([np.add(q[i], tree_data[parent][1][i])
                                                      for i in range(len(tree_data[parent][1]))])


                # Create the corresponding columns in the new DataFrame
                pos_df['%s_Xposition'%joint] = pd.Series(data=[e[0] for e in tree_data[joint][1]], index=pos_df.index)
                pos_df['%s_Yposition'%joint] = pd.Series(data=[e[1] for e in tree_data[joint][1]], index=pos_df.index)
                pos_df['%s_Zposition'%joint] = pd.Series(data=[e[2] for e in tree_data[joint][1]], index=pos_df.index)

            new_track = track.clone()
            new_track.values = pos_df
            Q.append(new_track)
        return Q

def deg2rad(x):
    return x/180*math.pi


def rad2deg(x):
    return x/math.pi*180

class Rotation():
    def __init__(self,rot, **params):
        self.rotmat = []
        self._from_euler(rot[0],rot[1],rot[2], params)

    def _from_euler(self, alpha, beta, gamma, params):
        '''Expecting degress'''
        if params['from_deg']==True:
            alpha = deg2rad(alpha)
            beta = deg2rad(beta)
            gamma = deg2rad(gamma)        
        ca = math.cos(alpha)
        cb = math.cos(beta)
        cg = math.cos(gamma)
        sa = math.sin(alpha)
        sb = math.sin(beta)
        sg = math.sin(gamma)        
        Rx = np.asarray([[1, 0, 0], 
              [0, ca, sa], 
              [0, -sa, ca]
              ])
        Ry = np.asarray([[cb, 0, -sb], 
              [0, 1, 0],
              [sb, 0, cb]])
        Rz = np.asarray([[cg, sg, 0],
              [-sg, cg, 0],
              [0, 0, 1]])
        self.rotmat = np.eye(3)
        self.rotmat = np.matmul(Rx, self.rotmat)
        self.rotmat = np.matmul(Ry, self.rotmat)
        self.rotmat = np.matmul(Rz, self.rotmat)

    def __str__(self):
        return "Rotation Matrix: \n " + self.rotmat.__str__()
    


def print_skel(X):
    stack = [X.root_name]
    tab=0
    while stack:
        joint = stack.pop()
        tab = len(stack)
        print('%s- %s (%s)'%('| '*tab, joint, X.skeleton[joint]['parent']))
        for c in X.skeleton[joint]['children']:
            stack.append(c)
            
def draw_stickfigure(mocap_track, frame, data=None, joints=None, draw_names=False, ax=None, figsize=(8,8)):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    if joints is None:
        joints_to_draw = mocap_track.skeleton.keys()
    else:
        joints_to_draw = joints
    
    if data is None:
        df = mocap_track.values
    else:
        df = data
        
    for joint in joints_to_draw:
        ax.scatter(x=df['%s_Xposition'%joint][frame], 
                   y=df['%s_Yposition'%joint][frame],  
                   alpha=0.6, c='b', marker='o')

        parent_x = df['%s_Xposition'%joint][frame]
        parent_y = df['%s_Yposition'%joint][frame]
        
        children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]
        
        for c in children_to_draw:
            child_x = df['%s_Xposition'%c][frame]
            child_y = df['%s_Yposition'%c][frame]
            ax.plot([parent_x, child_x], [parent_y, child_y], 'k-', lw=2)
            
        if draw_names:
            ax.annotate(joint, 
                    (df['%s_Xposition'%joint][frame] + 0.1, 
                     df['%s_Yposition'%joint][frame] + 0.1))

    return ax


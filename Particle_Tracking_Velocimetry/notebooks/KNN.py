# import the essential modules

# import modules for the detection of the particles 
import glob
import numpy as np 
import pandas as pd
from sklearn.neighbors import NearestNeighbors as nnbrs


# function to perform the Particle tracking via using KNN
def vel_spins(frame1, frame2):
    '''
    This function uses KNN to match the particles with there image ain two consequetive frames
    Implemented KNN via the 'ball tree algo'.
    K =1 for the nearest neighbor
    Params:
        frame1: preceding snapshot
        frame2: succeeding snapsot
    '''
    
    # convert the positional data from the preceding snap into correct fromat of  [xi, yi] 
    counter_frame1 = [];
    for i in range(len(frame1.values)):
        counter_frame1.append(frame1.values[i][1:])
    frame1_train_data = np.array(counter_frame1) 
    
    # convert the positional data from the succeeding snap into correct fromat of  [xi, yi]
    counter_frame2 =[];
    for i in range(len(frame2.values)):
        counter_frame2.append(frame2.values[i][1:])
    frame2_train_data = np.array(counter_frame2)
    
    # ...... Implementation of the KNN to find the nearest neighbour
    
    # instatiate the KNN model setting K =1 and the algorithm of ball tree for more efficient search
    train_data = nnbrs(n_neighbors=1, algorithm='ball_tree')
    
    # train the model using the data from the preceding SnapShot
    train_data.fit(frame1_train_data)
    
    # Use this aforementioned trained model to predict the nearest neighbour of 
    # Here we implement the logic .... the nearest neighbour will be the image of itself!!!
    
    _,indices = train_data.kneighbors(frame2_train_data) 
    # indices above provide the tags (refernce) to connect the images of the same particles between two frames 
    
    # Use the above tags to re-arrange the particle positions from the preceding frame 
    add_X = []
    add_Y = []
    for x in indices:
        add_X.append(frame1.iloc[x]['X'])
        add_Y.append(frame1.iloc[x]['Y'])
    
    # attach the rearanged particle poisitons to the positions in the final frame 
    # to create a single master coordinate table by adding to frame 2
    frame2['XX'] = np.array(add_X)
    frame2['YY'] = np.array(add_Y)  
    
    frame2['dX'] = frame2['XX'] - frame2['X']
    frame2['dY'] = frame2['YY'] - frame2['Y']
    
    # compute  ''' Displacement''' for gnerating the velocity vectors
    frame2['dR'] = np.sqrt(frame2['dX']**2 + frame2['dY']**2)
    
    # return the master table (updated frame 2) with the linking data 
    return frame2 



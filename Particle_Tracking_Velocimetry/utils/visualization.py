import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def visualize(path, arrowdata_F, arrowdata_NF):
    '''
    Function to visualize the generated velocity vector directions from the tagged data. 
    
    Params: 
        path        : path to the base image (snaphot) of the particles
        arrowdata_F : tagged data for the fluo-particles
        arrowdata_NF: tagged data for the non-fluo particles  
    '''
    
    # # Plot the PTV ~~~~~~~~~~~~~
   
    plt.figure(figsize = (12,12))
    ax = plt.axes()
    ax.axis('off')
    base_img = plt.imread(path)
    
    plt.imshow(base_img, cmap = 'Greys_r')

    
    # plot the tagged data for the fluorescent particles
    for i in range(0,len(arrowdata_F)):
        plt.arrow(arrowdata_F.iloc[i]['X'],arrowdata_F.iloc[i]['Y'],-arrowdata_F.iloc[i]['dX']*4,
                  -arrowdata_F.iloc[i]['dY']*4, head_width=12, head_length=9, color = 'b')
    
    # plot the tagged data for the non-fliuorescent particles
    for i in range(0,len(arrowdata_NF)):
        plt.arrow(arrowdata_NF.iloc[i]['X'],arrowdata_NF.iloc[i]['Y'],-arrowdata_NF.iloc[i]['dX']*5,
                  -arrowdata_NF.iloc[i]['dY']*5, head_width=12, head_length=9, color = 'r', width= 0.3)
    
    plt.show()

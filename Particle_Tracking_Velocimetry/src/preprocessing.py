# import the necessary modules
import numpy as np
import pandas as pd



class Preprocess:
    '''
    Define a class to preprocess the trajectory data and the tagged data.
    '''
    def __init__(self, paths,center, radius):
        '''
        Constructor for initialisation
        '''
        self.F_data_2  = pd.read_csv(paths[0], sep = '\t')
        self.NF_data_2  = pd.read_csv(paths[1], sep = '\t')

        # 2nd frame
        self.F_data = pd.read_csv(paths[2], sep='\t')
        self.NF_data = pd.read_csv(paths[3], sep='\t')
        
        self.center = center
        self.radius = radius
        
        
    def Cat2Pol_coord(self):
        '''
        Cartesian to polar coodinate system transformation for both the particle systems
        returns: 
            The cleaned trajectory data without stray points in the following sequence from left to right
            fluo-frame-1 
            non-fluo-frame-1
            fluo-frame-2
            non-fluo-frame-2
            
        '''
        self.NF_data['R'] = np.sqrt((self.NF_data['X'] - self.center[0])**2 + (self.NF_data['Y'] - self.center[1])**2)
        self.NF_Data_cropped = self.NF_data[self.NF_data['R']<self.radius]
        self.F_data['R'] = np.sqrt((self.F_data['X'] - self.center[0])**2 + (self.F_data['Y'] - self.center[1])**2)
        self.F_Data_cropped = self.F_data[self.F_data['R']<self.radius]

        self.NF_data_2['R'] = np.sqrt((self.NF_data_2['X'] - self.center[0])**2 + (self.NF_data_2['Y'] - self.center[1])**2)
        self.NF_Data_cropped_2 = self.NF_data_2[self.NF_data_2['R']<self.radius]
        self.F_data_2['R'] = np.sqrt((self.F_data_2['X'] - self.center[0])**2 + (self.F_data_2['Y'] - self.center[1])**2)
        self.F_Data_cropped_2 = self.F_data_2[self.F_data_2['R']<self.radius]
        
        return self.F_Data_cropped, self.NF_Data_cropped, self.F_Data_cropped_2, self.NF_Data_cropped_2
        
        
    def particle_image_mismatch_removal(self,arrowdata_F, arrowdata_NF, cutoff): 
        '''
        Removes the particles trajectories that suffer from wrong mismatch
        
        Params:
            arrowdata_F: Tagged dataframe for the fluorescent particles
            arrowdata_NF: Tagged dateframe for the non-fluorescent particles.
            
            cutoff: distances (in px) higher than cutoff implies a clear mismatch.
            We know this fromm the theoretical limit set by max frame-rate of the camera
        
        Returns: 
            dataframe with the particles tagged to their images
        '''
        # drop the ones without neighbours
        arrowdata_F.dropna()
        
        # consider only the ones within the cutoff for the image
        arrowdata_F = arrowdata_F[arrowdata_F['dR']<cutoff]
        
        # dr cannot be less than 0 for a correct tagging
        arrowdata_F = arrowdata_F[arrowdata_F['dR']>0]
        
        # do the same for the non-florescent particles
        arrowdata_NF.dropna()
        arrowdata_NF = arrowdata_NF[arrowdata_NF['dR']<cutoff]
        arrowdata_NF = arrowdata_NF[arrowdata_NF['dR']>0]
        arrowdata_NF['R'] = np.sqrt((arrowdata_NF['X']-self.center[0])**2 + (arrowdata_NF['Y']-self.center[1])**2)
        arrowdata_NF = arrowdata_NF[arrowdata_NF['R']<self.radius]
        
        return arrowdata_F, arrowdata_NF
        
    
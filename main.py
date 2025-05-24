# import the modules and my scripts

# import the main modules 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from glob import glob
from pathlib import Path

# import user defined packages 
from src.preprocessing import Preprocess
from src.KNN import vel_spins 
from utils.visualization import visualize

def main()

   # define some of the necessary constants
   radius = 630 # px
   center  =  [700, 700] # px of x and y
   cutoff = 8
   
   # Set up the file subsystem architecture.
   # find the current directory
   root = Path().resolve()
   print(root)
   # set the path to the trajectory data 
   trajectory_paths = glob(str(root /'data') + '/**/*')
   
   
   # create an object of the preprocessing class
   traj_set1 = Preprocess(trajectory_paths,center, radius)
   coordinate_set = traj_set1.Cat2Pol_coord()
   
   # implement the KNN to tag the images
   arrowdata_F = vel_spins(coordinate_set[2][[' ', 'X', 'Y']],coordinate_set[0][[' ', 'X', 'Y']])
   arrowdata_NF = vel_spins(coordinate_set[3][[' ', 'X', 'Y']],coordinate_set[1][[' ', 'X', 'Y']])
   
   
   # remove the untagged or wrongly tagged particles
   arrowdata_F, arrowdata_NF = traj_set1.particle_image_mismatch_removal(arrowdata_F, arrowdata_NF, cutoff)
   
   # visualise the velocity vectors of the sucessfully tagged particles
   visualize(glob(str(root/'data')+'/*')[0], arrowdata_F, arrowdata_NF)



if __name__ == "__main__":
   main()

# Imports for clarity
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.optimize import curve_fit
import pandas as pd
import sys
# Import of the module I wrote for MCMC Simulation
import IsingModel as Alloy
plt.style.use('FigureStyle/ConfigStyle.mplstyle')

def PlotMagsProfile(filepath,fitparams,use_fit = True):
    # Read generated data into DataFrame
    dataset = pd.read_csv(filepath)
    # Generate Fit curve
    if use_fit:
        x = np.linspace(0.05,2.5,num=300)
        y = Alloy.MagsFit(x,*fitparams)
    # Plot dataset
    plt.scatter(dataset['Temps'],dataset['Mags'])
    if use_fit:
        plt.plot(x,y)
    plt.xlabel(r'$k_BT/Jz$')
    plt.ylabel(r'$\langle |M| \rangle$')
    plt.show()

if __name__ == '__main__':
    # Define keys for plotting
    Lkey = [55,8,16,32,11]
    Mukey = [0.0,-1.7]
    hkey = [0.0,0.1,0.3,0.5]
    # Read Muidx, hidx and T from command line
    Muidx = sys.argv[1]
    hidx = sys.argv[2]
    T_ = float(sys.argv[3])
    save_ = sys.argv[4] == 'True'
    # Create Alloy Object for simulation
    myAlloy = Alloy.IsingLattice2D\
    (LatticeSize=16,mu=Mukey[int(Muidx)],h=hkey[int(hidx)])
    # Initialise Lattice
    myAlloy.FillNeighbours()
    # Create animation of thermalisation
    filename_ = 'Animations/Anime'+Muidx+'_'+hidx
    myAlloy.ThermalAnime(T=T_,Filename=filename_,save=save_)

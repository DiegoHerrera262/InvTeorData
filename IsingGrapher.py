# Imports for clarity
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.optimize import curve_fit
import pandas as pd
# Import of the module I wrote for MCMC Simulation
import IsingModel as Alloy
plt.style.use('FigureStyle/PaperStyle.mplstyle')

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
    fitparams = []
    PlotMagsProfile('Alt_Mags_0_0.csv',fitparams,use_fit=False)

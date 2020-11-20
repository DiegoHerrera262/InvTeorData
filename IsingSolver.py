# Imports for clarity
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.optimize import curve_fit
import pandas as pd
# Import twilio client for messaging
from twilio.rest import Client
import os
# Import of the module I wrote for MCMC Simulation
import IsingModel as Alloy
# Set matplotlib style for plotting
# plt.style.use('FigureStyle/PaperStyle.mplstyle')

client = Client()
myphone = os.environ['MYPHONE']

def SaveMagsProfile(\
    s = 8, miu = -10.0, hi = 0.0, points = 30, filename = 'DemoFile',\
    to_num = '1234568090'):

    '''
    Function for data generation and stage control reporting
    using Twilio API
    '''

    # Define lattice object
    MyAlloy = Alloy.IsingLattice2D(LatticeSize = s, mu = miu, h = hi)
    # Initialise Lattice
    MyAlloy.FillNeighbours()
    # Initialise Server notification params
    server_num = 'whatsapp:+14155238886'
    mynum = 'whatsapp:+57'+to_num
    # Send starting message
    init_message = 'Started Profile Cooking...\n' +\
                   'mu = ' + str(miu) + '\n' +\
                   'h = ' + str(hi)
    client.messages.create(body=init_message,
                           from_ = server_num,
                           to = mynum)
    # Compute Magnetisation profile and best interpolation
    start_time = time.time()
    Temps, Mags, fitparams = \
    MyAlloy.MagsProfile(Tmin=0.05,Tmax=3.5,points=points)
    end_time = time.time()
    # Send finish cooking message
    dt = end_time - start_time
    init_message = 'Finished Profile Cooking...\n' +\
                   'Ellapsed time = ' + str(dt) + '\n' +\
                   'Begin saving data to .csv file...'
    client.messages.create(body=init_message,
                           from_ = server_num,
                           to = mynum)
    # Convert np.arrays to dataframe
    dataset = {
        'Temps':Temps,
        'Mags':Mags,
    }
    data = pd.DataFrame.from_dict(dataset)
    filepath = filename+'.csv'
    data.to_csv(filepath,index=False,header=True)
    # Send finish writting message
    dt = end_time - start_time
    init_message = 'Finished saving data...\n' +\
                   'A = ' + str(fitparams[0]) + '\n' +\
                   'B = ' + str(fitparams[1]) + '\n'+\
                   'C = ' + str(fitparams[2]) + '\n'+\
                   'D = ' + str(fitparams[3]) + '\n'+\
                   'Hope You Have an excellent day ;).'
    client.messages.create(body=init_message,
                           from_ = server_num,
                           to = mynum)
    return fitparams

def GenData(mus, hs, si=4):

    '''
    Function for generating and saving all magnetisation data
    and fitting parameters for critical temperature determination
    '''

    # Create dictionary to save fitting parameters
    fits = {}
    for i in range(len(mus)):
        for j in range(len(hs)):
            myk = str(i)+'_'+str(j)
            aux = {myk:[]}
            fits.update(aux)

    # Start generating and saving data
    for i in range(len(mus)):
        for j in range(len(hs)):
            myk = str(i)+'_'+str(j)
            fname = 'Mags_'+myk
            # Compute magnetisation profile, save and returns
            # interpolation data
            fitdata = SaveMagsProfile(\
            s=si,miu=mus[i],hi=hs[j],points=30,filename=fname,\
            to_num=myphone)
            # Save interpolation data to file for reference later
            fits[myk] = fitdata

    # Create dataframe for saving fit parameters
    fitsframe = pd.DataFrame.from_dict(fits)
    fitsframe.to_csv('FitParams.csv',index=False,header=True)
    # Send finishing message
    server_num = 'whatsapp:+14155238886'
    init_message = 'Finished Cooking Batch...\n' +\
                   'Start pushing results'
    client.messages.create(body=init_message,
                           from_ = server_num,
                           to = myphone)


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
    plt.ylabel(r'$\rangle |M| \langle$')
    plt.show()

if __name__ == '__main__':

    #fitparams = SaveMagsProfile(\
    #s=8,miu=-10.0,hi=0.0,points=26,filename='TestFile',to_num='3158009152')

    #fitparams = []
    #PlotMagsProfile('Mags_1_1.csv',fitparams,use_fit=False)

    mu = [0.0,-10.0]
    hs = [0.0,0.1]

    GenData(mu,hs,si=2)

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.optimize import curve_fit

def num2bin(num):
    return [int(x) for x in '{0:02b}'.format(int(num))]

def MagsFit(x,A,B,C,D,E):
    return D*(np.tanh(A-B*x)+E)**C

class IsingLattice2D:

    '''Class For modelling an Ising Hamiltonian using Monte Carlo'''

    # Attributes of the class
    LatticeSize = 1
    NumSpins = 1
    Lattice = []
    Neighbours = None
    SpinflipKey = {
        -1:[0,1],
        0:[-1,1],
        1:[-1,0]
    }
    mu = 0.0
    h = 0.0

    # Functions of the class

    def __init__(self,LatticeSize = 16,mu = -1.0,h = 0.0):

        '''
        Initialise Lattice with Random Integers -1,1
        '''

        # Random initialization
        self.mu = mu
        self.h = h
        self.LatticeSize = LatticeSize
        self.NumSpins = 2*LatticeSize**2
        self.Lattice = [rd.choice([1, 0,-1]) for x in range(self.NumSpins)]

    def lidx2ij(self,lidx):

        '''
        Function that converts linear array idx to 2D idx
        with Periodic Boundary Conditions
        '''

        return np.array([(lidx//self.LatticeSize)%self.LatticeSize,\
         lidx%self.LatticeSize])

    def ij2lidx(self, idx_list):

        '''
        Function that converts 2D idx to linear array idx
        with Periodic Boundary Conditions
        '''

        return self.LatticeSize*(idx_list[0]%self.LatticeSize) +  \
        idx_list[1]%self.LatticeSize

    def cell(self, lidx):

        '''
        Function that returns linear idx of numerated cells
        '''

        idx2d = np.array(self.lidx2ij(lidx))
        return np.array([self.ij2lidx(np.add(idx2d,np.array(num2bin(k))))\
         for k in range(4)])

    def FillNeighbours(self):

        '''
        Function that fills array of neighbours for each S sublattice
        '''

        auxcont = np.array([[self.cell(\
        self.ij2lidx(self.lidx2ij(idx) - k*np.array([1,1]))) \
        for idx in range(self.LatticeSize**2)] \
        for k in range(2)])

        dictLattice0 = {k:(auxcont[0,k]+self.NumSpins//2)\
         for k in range(self.NumSpins//2)}
        dictLattice1 = {(k+self.NumSpins//2):auxcont[1,k]\
         for k in range(self.NumSpins//2)}
        dictLattice0.update(dictLattice1)
        self.Neighbours = dictLattice0

    def DeltaEnergy(self,next_spin,idx):

        '''
        Compute energy difference for flipping.
        '''

        dS = next_spin - self.Lattice[idx]
        s = next_spin + self.Lattice[idx]
        sig = -1 if idx >= self.LatticeSize else 1
        return dS*(-0.5*sum(self.Lattice[k] for k in self.Neighbours[idx])+\
                self.h*sig + self.mu*s)

    def MCStep(self,beta):

        '''
        Implementation of a Monte Carlo Step.
        '''

        # Choose random lattice site
        k = rd.randint(0,self.NumSpins-1)
        # Choose random new spin value
        flip_idx = 1
        if rd.uniform(0.0,1.0) < 0.5:
            flip_idx = 0
        newspin = self.SpinflipKey[self.Lattice[k]][flip_idx]
        # Accept according to Monte Carlo
        if rd.uniform(0.0,1.0) < np.exp(-beta*self.DeltaEnergy(newspin,k)):
            self.Lattice[k] = newspin

    # Compute statistically important quantities
    def ConfigMagnetisation(self):

        '''
        Compute Magnetisation of system
        '''

        return sum(self.Lattice[k] for k in range(self.NumSpins))

    def ConfigEnergy(self):

        '''
        Compute Energy of system
        '''

        return sum(self.Lattice[k]*(\
                        -1.0*sum(self.Lattice[j] for j in self.Neighbours[k]) \
                        + self.h\
                        + self.mu * self.Lattice[k])
                        for k in range(self.NumSpins//2)) + \
               sum(self.Lattice[k]*(-self.h + self.mu*self.Lattice[k])\
                   for k in range(self.NumSpins//2,self.NumSpins))

    def ConfigStatQuantities(self):

        '''
        Compute Magnetisation & Energy of system
        '''

        Eint = self.ConfigEnergy()
        M = np.abs(self.ConfigMagnetisation())
        return Eint, M

    # Implement Lattice Monte Carlo Step
    def LatticeMCStep(self,beta):

        '''
        Implementation of a Monte Carlo Step on the lattice. This normalises
        execution time.
        '''

        for step in range(0,self.NumSpins):
            self.MCStep(beta)

    # Perform simulation over temperature range
    def MCSimulation(\
        self,Tmin=0.1,Tmax=4.0,points=10,MCSTEPS=15000,SPARE_MCSTEPS=2000):

        '''
        Simulation over a temperature range. This function computes:
        - Magnetisation
        - Internal Energy
        - Specific heat
        - Mag. Susceptibility
        '''

        # Create array of temperatures
        Temps = [0.0]*int(points)
        deltaT = (Tmax-Tmin)/(points-1)
        for i in range(points):
            Temps[i] = i*deltaT + Tmin
        # Create array of magnetisations
        Mags = [0.0]*int(points)
        # Create array of sqrd magnetisations
        Suscep = [0.0]*int(points)
        # Create array of energies
        Eints = [0.0]*int(points)
        # Create array of sqrd energies
        SpecHeat = [0.0]*int(points)
        # Perform MCMC Steps
        MC_Mag = np.array([0.0]*(MCSTEPS-SPARE_MCSTEPS))
        MC_Eint = np.array([0.0]*(MCSTEPS-SPARE_MCSTEPS))
        for i in range(0,points):
            beta = 1.0/Temps[i]
            # Perform MCMC Steps
            # Spare MC steps while thermalising
            for j in range(SPARE_MCSTEPS):
                self.LatticeMCStep(beta)
            # Compute observables at every MC step
            for j in range(MCSTEPS-SPARE_MCSTEPS):
                self.LatticeMCStep(beta)
                MC_Eint[j] = self.ConfigEnergy()/self.NumSpins
                MC_Mag[j] = abs(self.ConfigMagnetisation())/self.NumSpins
            Mags[i] = np.mean(MC_Mag)
            Eints[i] = np.mean(MC_Eint)
            SpecHeat[i] = (np.std(MC_Eint)**2)*(beta**2)*self.NumSpins
            Suscep[i] = (np.std(MC_Mag)**2)*beta*self.NumSpins
        return Temps,Mags,Suscep,Eints,SpecHeat

    def SingleTempMCMC(self,T = 0.1,MCSTEPS=15000,SPARE_MCSTEPS=2000):

        '''
        Computes Important thermodynamic quantities for a single
        temperature. This function might be used with multiprocessing
        module to determine if it speeds up calculations
        '''

        MC_Mag = np.array([0.0]*(MCSTEPS-SPARE_MCSTEPS))
        MC_Eint = np.array([0.0]*(MCSTEPS-SPARE_MCSTEPS))
        beta = 1.0/T
        # Perform MCMC Steps
        # Spare MC steps while thermalising
        for j in range(SPARE_MCSTEPS):
            self.LatticeMCStep(beta)
        # Compute observables at every MC step
        for j in range(MCSTEPS-SPARE_MCSTEPS):
            self.LatticeMCStep(beta)
            MC_Eint[j] = self.ConfigEnergy()/self.NumSpins
            MC_Mag[j] = abs(self.ConfigMagnetisation())/self.NumSpins
        Mags = np.mean(MC_Mag)
        Eints = np.mean(MC_Eint)
        SpecHeat = (np.std(MC_Eint)**2)*(beta**2)*self.NumSpins
        Suscep = (np.std(MC_Mag)**2)*beta*self.NumSpins
        return Mags, Eints, SpecHeat, Suscep

    def SingleTempMag(self,T = 0.1,MCSTEPS=15000,SPARE_MCSTEPS=2000):

        '''
        Computes magnetisation for a single temperature. This Function
        might be used with multiprocessing module to determine critical
        tempereture from critical point of curve.
        '''

        MC_Mag = np.array([0.0]*(MCSTEPS-SPARE_MCSTEPS))
        beta = 1.0/T
        # Perform MCMC Steps
        # Spare MC steps while thermalising
        for j in range(SPARE_MCSTEPS):
            self.LatticeMCStep(beta)
        # Compute observables at every MC step
        for j in range(MCSTEPS-SPARE_MCSTEPS):
            self.LatticeMCStep(beta)
            MC_Mag[j] = abs(self.ConfigMagnetisation())/self.NumSpins
        return np.mean(MC_Mag)

    def MagsProfile(\
        self,Tmin=0.1,Tmax=4.0,points=10,MCSTEPS=15000,SPARE_MCSTEPS=5000):

        '''
        Simulation over a temperature range. This function computes:
        - Magnetisation
        '''

        # Create array of temperatures
        Temps = np.array([0.0]*int(points))
        deltaT = (Tmax-Tmin)/(points-1)
        for i in range(points):
            Temps[i] = i*deltaT + Tmin
        # Create array of magnetisations
        Mags = np.array([0.0]*int(points))
        MC_Mag = np.array([0.0]*(MCSTEPS-SPARE_MCSTEPS))
        for i in range(0,points):
            beta = 1.0/Temps[i]
            # Perform MCMC Steps
            # Spare MC steps while thermalising
            for j in range(SPARE_MCSTEPS):
                self.LatticeMCStep(beta)
            # Compute observables at every MC step
            for j in range(MCSTEPS-SPARE_MCSTEPS):
                self.LatticeMCStep(beta)
                MC_Mag[j] = abs(self.ConfigMagnetisation())/self.NumSpins
            Mags[i] = np.mean(MC_Mag)
        # Compute best fit to MagsFit
        fitparams, covmat = curve_fit(\
        MagsFit,Temps,Mags,bounds=(0,[50.0,50.0,1,0.5,10.0]))
        return Temps,Mags,fitparams

    def Lattice2img(self):
        img = np.array(127*np.array(self.Lattice)+128, dtype=np.uint8)
        return np.transpose(np.reshape(img, (-1,self.LatticeSize)))

    def ShowLattice(self):
        img = np.array(127*np.array(self.Lattice)+128, dtype=np.uint8)
        img = np.reshape(img, (-1,self.LatticeSize))
        plt.title(r"Spin Configuration")
        plt.imshow(img,cmap='gray')
        plt.show()

    def ThermalAnime(self, T = 0.1, MCSTEPS = 15000, Filename = 'Demo', \
    save = False):
        fig = plt.figure()
        ims = [None]*(MCSTEPS//300)
        beta = 1.0/T
        for i in range(MCSTEPS):
            self.LatticeMCStep(beta)
            if i%300 == 0:
                im = plt.imshow(self.Lattice2img(), cmap = 'gray', \
                animated = True)
                ims[i//300] = [im]
        ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, \
        repeat_delay=100)
        plt.title(r'Thermalisation of System'+'\n'+r'$T = $ '+str(T))
        plt.show()
        if save:
            ani.save(Filename+'.gif', writer='imagemagick')
            plt.close()

    def SimulationTest(self, T = 0.1, MCSTEPS=15000):

        '''
        Function for performing thermalisation analysis
        '''

        beta = 1.0/T
        MCEints = [0.0]*MCSTEPS
        MCMags = [0.0]*MCSTEPS
        start_time = time.time()
        for i in range(MCSTEPS):
            self.LatticeMCStep(beta)
            MCMags[i] = abs(self.ConfigMagnetisation())/\
                        self.NumSpins
            MCEints[i] = self.ConfigEnergy()/\
                        self.NumSpins
        end_time = time.time()
        print('Ellapsed Time: ',end_time-start_time)
        # Plot thermalisation
        results, params = plt.subplots(2,1, constrained_layout = True)
        # Plot magnetisation results
        params[0].plot(range(MCSTEPS),MCMags,'ko')
        params[0].set_xlabel('Time (MCs)')
        params[0].set_ylabel('Magnetisation')
        # Plot Energy results
        params[1].plot(range(MCSTEPS),MCEints,'bo')
        params[1].set_xlabel('Time (MCs)')
        params[1].set_ylabel('Energy')
        results.suptitle('Thermalisation of System \n' + '$T = $ ' + str(T))
        plt.show()

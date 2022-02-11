import numpy as np
import pandas as pd
import scipy.special as special
from numba import jit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib.transforms import (
    Bbox, TransformedBbox, blended_transform_factory)

# import scipy.special as special

# Import math Library
import math 

from sympy import *

# PARA USAR A LA HORA DE GUARDAR DOS COLUMNAS EN UN ARCHIVO
import csv


densz= pd.read_csv("densz_histogram.txt",header=None,sep='\s+', names=['nz','z'])

densz_2= pd.read_csv("dens_hist2.txt",header=None,sep='\s+', names=['nz2','z2'])

densz_teo= pd.read_csv("densz_teo.txt",header=None,sep='\s+',names=['nz_teo','z_teo'])

stdev= pd.read_csv("stdev_nz.txt",header=None,sep='\s+', names=['stdev','z3'])








    # return -4.0*epsilon**3.0*rho*np.sqrt(tx)*( ty -tx )/(3.0*np.sqrt(np.pi))
# tiemp=np.linspace(0.0,30050.0,len(temp))
plt.plot(densz_teo['z_teo'],densz_teo['nz_teo'],color='C0',label="$n_z$ ")
# plt.plot(zmedia,profile_density_2(h,z,a,norma),color='C2',label="$n_z$ ")
plt.plot(densz['z'],densz['nz'],color='C1',marker="o",linestyle="",label="$n_z$ (MD)")   
plt.errorbar(densz_2['z2'],densz_2['nz2'], yerr=stdev['stdev'],color='C2',marker="o",linestyle="",label="$n_z$ (MD)")    
plt.grid(color='k', linestyle='--', linewidth=0.5,alpha=0.2)
plt.xlabel ( r' $\overline{z}$ ', fontsize=30)
plt.ylabel ( r' $n_2$ ',rotation=0.0,fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.title ( r' \textbf {Perfil de densidad ($h=1.5\sigma$)}  ',fontsize=40)



# plt.legend(loc=0,fontsize=30)
plt.show()

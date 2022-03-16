#instability_dens.py
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

import cmath


# density= pd.read_csv("densidad_horizontal_0.95_0.50.txt",header=None,sep='\s+', names=['re_n','im_n'])

# t= pd.read_csv("tiemposdecol_0.95.txt",header=None,sep='\s+', names=['t'])
kapa= Symbol("kapa",positive=True)
mu=Symbol("mu",positive=True)

h=1.50
# h=1.90
# rho=0.2111
rho=0.015
n=500

# rho=0.03
l=n/(rho*(h-1.0))
k=np.pi/l

densy=319.60441969489989/l
# print(densy)
# densy_teo=-0.6*(k**4*(0.3*x-y))**(1/3)

# Estos son los desarrollos en serie de los los autovalores obtenidos  en orden 2, o sea, con un error O(k^3)

eigen1=-0.333333 *1.50* (kapa - mu) *k**2


eigen2=( k*1j)-0.333333 *(0.75*(mu+kapa))*k**2


eigen3=-( k*1j)-0.333333 *(0.75*(mu+kapa))*k**2



print(eigen1)

print(eigen2)

print(eigen3)

coef=np.arccos(densy)/(-1.10329e-9)

# print(coef)




    # return -4.0*epsilon**3.0*rho*np.sqrt(tx)*( ty -tx )/(3.0*np.sqrt(np.pi))
# tiemp=np.linspace(0.0,30050.0,len(temp))

# plt.plot(zmedia,profile_density_2(h,z,a,norma),color='C2',label="$n_z$ ")
# plt.plot(t['t'],density['re_n'],color='C1',label="$n_{\frac{\pi}{L}}$ (MD)")   
 
# plt.grid(color='k', linestyle='--', linewidth=0.5,alpha=0.2)
# plt.xlabel ( r' $\overline{z}$ ', fontsize=30)
# plt.ylabel ( r' $n_2$ ',rotation=0.0,fontsize=30)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)

# plt.title ( r' \textbf {Perfil de densidad ($h=1.5\sigma$)}  ',fontsize=40)



# plt.legend(loc=0,fontsize=30)
# plt.show()

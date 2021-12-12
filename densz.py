import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib.transforms import (
    Bbox, TransformedBbox, blended_transform_factory)

import scipy.special as special
#from scipy import special
# Import math Library
import math 

densz= pd.read_csv("densz_1.00.txt",names=['nz'])


partz=10
n=500
h=1.50
rho=0.06
l=n/(rho*(h-1.0))

z=np.linspace(0.5,(h-0.5),100)
zmd=np.linspace(0.5,(h-0.5),len(densz))

densz= densz/(550000*(h-1.0))


def elimina_ceros(original):
    nueva = []
    for dato in original:
        if dato != 0.0:
            nueva.append(dato)
    return nueva

densz=elimina_ceros(densz['nz'])

print(densz)

# print(temp)
# print(tiempo)
def profile_density(h,n,z,l):
     """
     h: height of the profile
     n: number of particles
     z: position of the particles
     l: length of the box
     """
     a= math.pi*n/l
     b=np.sqrt(math.pi/a)*special.erfi(np.sqrt(a)*(h/2.0-0.50))
     return (n/(l*b))*np.exp(a*(z-h*0.50)**2) 

    
    # return -4.0*epsilon**3.0*rho*np.sqrt(tx)*( ty -tx )/(3.0*np.sqrt(np.pi))
# tiemp=np.linspace(0.0,30050.0,len(temp))
plt.plot(z,l*profile_density(h,n,z,l),color='C0',label="$n_z$ ")
# plt.plot(zmd,densz,color='C1',marker="o",label="$n_z$ (MD)")      
plt.grid(color='k', linestyle='--', linewidth=0.5,alpha=0.2)
plt.xlabel ( r' $z$ ', fontsize=30)
plt.ylabel ( r' $n_z$ ',rotation=0.0,fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.title ( r' \textbf {Perfil de densidad}  ',fontsize=40)



plt.legend(loc=0,fontsize=30)
plt.show()

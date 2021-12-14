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

densz= pd.read_csv("densz_1.00.txt",names=['nz'])

particion=8
sigma=1.00
n=500
# h=1.90
h=1.30
rho=0.2111
# rho=0.03
l=n/(rho*(h-1.0))
a= n/l
b=np.sqrt(math.pi/a)*special.erfi(np.sqrt(a)*(h*0.50-0.50))




z=np.linspace(0.5,(h-0.5),100)
zmd=np.linspace(0.5,(h-0.5),len(densz))

zmedia=(2*z-sigma)/(2*sigma)

zmedia2=(2*zmd-sigma)/(2*sigma)


# densz=particion*densz/(h-1.0)


def elimina_ceros(original):
    nueva = []
    for dato in original:
        if dato != 0.0:
            nueva.append(dato)
    return nueva
def elimina_elemento(original,elemento):
    original.pop(elemento)
    return original

densz=elimina_ceros(densz['nz'])
# densz= elimina_elemento(densz,0)

print(densz)






# print(temp)
# print(tiempo)
def profile_density(h,n,z,l,a,b):
     """
     h: height of the profile
     n: number of particles
     z: position of the particles
     l: length of the box
     """
    
     return (n/(l*b))*np.exp(a*(z-h*0.50)**2) 

    
    # return -4.0*epsilon**3.0*rho*np.sqrt(tx)*( ty -tx )/(3.0*np.sqrt(np.pi))
# tiemp=np.linspace(0.0,30050.0,len(temp))
plt.plot(zmedia,l*profile_density(h,n,z,l,a,b),color='C0',label="$n_z$ ")
# plt.plot(zmedia2,densz,color='C1',marker="o",linestyle="",label="$n_z$ (MD)")      
plt.grid(color='k', linestyle='--', linewidth=0.5,alpha=0.2)
plt.xlabel ( r' $\overline{z}$ ', fontsize=30)
plt.ylabel ( r' $n_2$ ',rotation=0.0,fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.title ( r' \textbf {Perfil de densidad ($h=1.9\sigma$)}  ',fontsize=40)



# plt.legend(loc=0,fontsize=30)
plt.show()

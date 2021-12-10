import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib.transforms import (
    Bbox, TransformedBbox, blended_transform_factory)

# import scipy.special as special
#from scipy import special
# Import math Library
# import math 

densz= pd.read_csv("densz_1.00.txt",names=['nz'])



def elimina_ceros(original):
    nueva = []
    for dato in original:
        print(dato)
        if dato != 0.0:
            nueva.append(dato)
    return nueva

densz=elimina_ceros(densz['nz'])

print(densz)

partz=10
n=500
rho=0.06
h=1.90
l=n/(rho*(h-1.0))
# z=np.linspace(0.0,h,100)
z=np.linspace(0.0,(2.0*(h-1.0)-1.0)/2.0,len(densz))
# print(temp)
# print(tiempo)
# def profile_density(h,n,z,l):
#      """
#      h: height of the profile
#      n: number of particles
#      z: position of the particles
#      l: length of the box
#      """
#      a= math.pi*n/l
#      b=np.sqrt(math.pi/a)*special.erfi(np.sqrt(a)*(h/2-0.50))
#      return (n/l*b)*np.exp(a*(z-h*0.50)**2) 

    
    # return -4.0*epsilon**3.0*rho*np.sqrt(tx)*( ty -tx )/(3.0*np.sqrt(np.pi))
# tiemp=np.linspace(0.0,30050.0,len(temp))
# plt.plot(z,profile_density(h,n,z,l),color='C0',label="$n_z$ ")
plt.plot(z,densz,color='C1',marker="o",label="$n_z$ (MD)")      
plt.grid(color='k', linestyle='--', linewidth=0.5,alpha=0.2)
plt.xlabel ( r' $z$ ', fontsize=30)
plt.ylabel ( r' $n_z$ ',rotation=0.0,fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.title ( r' \textbf {Perfil de densidad}  ',fontsize=40)



plt.legend(loc=0,fontsize=30)
plt.show()

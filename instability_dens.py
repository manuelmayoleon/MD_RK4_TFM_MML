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

import functions

from sklearn.linear_model import LinearRegression

density= pd.read_csv("densitypromy_0.95.txt",header=None,sep='\s+', names=['re','im'])

t= pd.read_csv("tiemposdecol_0.95.txt",header=None,sep='\s+', names=['t'])

kapa= Symbol("kapa",positive=True)
mu=Symbol("mu",positive=True)

h=1.50
sigma=1.0
# h=1.90
# rho=0.2111
# rho=0.015
rho=0.03
n=500

# rho=0.03
l=n/(rho*(h-1.0))
k=2*np.pi/l
alfa=0.95
epsilon=0.5

dens = n/(l*(h-sigma))
print(k)
print(dens)

# densy=319.60441969489989/l
# print(densy)
# densy_teo=-0.6*(k**4*(0.3*x-y))**(1/3)

# Estos son los desarrollos en serie de los los autovalores obtenidos  en orden 2, o sea, con un error O(k^3)

eigen1=-0.333333 *1.50* (kapa - mu) *k**2


eigen2=( k*1j)-0.333333 *(0.75*(mu+kapa))*k**2


eigen3=-( k*1j)-0.333333 *(0.75*(mu+kapa))*k**2





# print(coef)
tt=np.linspace(0,max(t['t']),1000)

kk=np.linspace(0.0,0.6,1000)
# x=Symbol("x")

# print( np,sqrt((-4.5 *k**4 *functions.Panel(1).kapa(alfa) - 2.* k**6* functions.Panel(1).kapa(alfa)**3 + 13.5* k**4* functions.Panel(1).mu(alfa) +22.5* k**2*functions.lamda1(alfa,epsilon) - 6.* k**4* functions.Panel(1).kapa(alfa)**2*functions.lamda1(alfa,epsilon)- 6.* k**2 *functions.Panel(1).kapa(alfa)* functions.lamda1(alfa,epsilon)**2 - 2. *functions.lamda1(alfa,epsilon)**3)**2 + 
#                 4* (3.* k**2 - (1.* k**2* functions.Panel(1).kapa(alfa) + 1.* functions.lamda1(alfa,epsilon))**2)**3+0j))


# print(1j*np.sqrt(abs((-4.5 *k**4* functions.Panel(1).kapa(alfa) - 2.* k**6* functions.Panel(1).kapa(alfa)**3 + 13.5* k**4* functions.Panel(1).mu(alfa) + 
#                 22.5* k**2* functions.lamda1(alfa,epsilon) - 6. *k**4* functions.Panel(1).kapa(alfa)**2* functions.lamda1(alfa,epsilon) - 6. *k**2* functions.Panel(1).kapa(alfa)* functions.lamda1(alfa,epsilon)**2 - 2. *functions.lamda1(alfa,epsilon)**3)**2 + 
#                 4 *(3.* k**2 - (1.* k**2* functions.Panel(1).kapa(alfa) + 1.* functions.lamda1(alfa,epsilon))**2)**3)))

# print(functions.eigen3_taylor(functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),0.0001))

# print(functions.Panel(1).mu(alfa))
# print(functions.Panel(1).kapa(alfa))
# print(functions.lamda1(alfa,epsilon))
min_index = np.argmin(density['re'])
max_index = np.argmax(density['re']) 
print(min_index) 
print(max_index) 


density=density['re']+1j*density['im']
denslog=np.log(density)
x=t['t']

denslog=denslog[min_index:max_index]
x=x[min_index :max_index]
density=density[min_index :max_index]
# print(len(x))
# print(len(density))
# print(len(x.values.reshape((-1, 1))))
reg = LinearRegression().fit(x.values.reshape((-1, 1)),np.real(denslog))
r_sq = reg.score(x.values.reshape((-1, 1)), np.real( denslog))
print('coefficient of determination:', r_sq)
print('intercept:', reg.intercept_)
print('slope:', reg.coef_)
linear_reg=reg.coef_*x+reg.intercept_
    # return -4.0*epsilon**3.0*rho*np.sqrt(tx)*( ty -tx )/(3.0*np.sqrt(np.pi))
# tiemp=np.linspace(0.0,100.0,len(t['t']))
densteo=np.exp(functions.eigenvalue1(functions.Panel(1).mu(alfa),functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),k)*tt*dens/20)
print('Theoretical slope:', functions.eigenvalue1(functions.Panel(1).mu(alfa),functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),k)*dens/20)

# expdens= np.exp(0.16*tt)
# print(functions.eigenvalue2(functions.Panel(1).mu(alfa),functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),k))
# plt.plot(tt,densteo,color='C2',label="$n_y$ ")
# plt.plot(x,density,color='C1',label="$n_{\frac{\pi}{L}}$ (MD)")   
plt.plot(x,denslog,color='C1',label="$n_{\frac{\pi}{L}}$ (MD)")   
plt.plot(x,linear_reg,color='C2',label="$n_{\frac{\pi}{L}}$ (MD)")   
# print((functions.eigenvalue3(functions.Panel(1).mu(alfa),functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),kk)).real)
# plt.plot(tt,expdens,color='C2',label="$n_y \;(k=2\pi/L)$ ")

# plt.plot(kk,functions.eigenvalue1(functions.Panel(1).mu(alfa),functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),kk),color='C2',label="$\lambda_1$ ")
# plt.plot(kk,functions.eigen1_taylor(functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),kk),color='C2',linestyle="--",label="$\lambda_1 \; \mathcal{O}(k^3)$ ")
# plt.plot(kk,np.real(functions.eigenvalue2(functions.Panel(1).mu(alfa),functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),kk)),linewidth=1.5,color='C4',label="$\lambda_2$ ")
# plt.plot(kk,functions.eigen2_taylor(functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),kk),color='C4',linestyle="--",label="$\lambda_2 \; \mathcal{O}(k^3)$ ")
# plt.plot(kk,np.real(functions.eigenvalue3(functions.Panel(1).mu(alfa),functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),kk)),color='C3',label="$\lambda_3$ ")
# plt.plot(kk,functions.eigen3_math(functions.Panel(1).mu(alfa),functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),kk),color='C3',linestyle=":",label="$\lambda_3 \; \mathcal{O}(k^3)$ ")
# plt.plot(kk,functions.eigen3_taylor(functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),kk),color='C3',linestyle="--",label="$\lambda_3 \; \mathcal{O}(k^3)$ ")
plt.grid(color='k', linestyle='--', linewidth=0.5,alpha=0.2)
# plt.xlabel ( r' $k$ ', fontsize=30)
# plt.ylabel ( r' $\lambda$ ',rotation=0.0,fontsize=30)
plt.xlabel ( r'$t$', fontsize=30)
plt.ylabel ( r' $n_2$ ',rotation=0.0,fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.title ( r' \textbf {Autovalores en función de k }  ',fontsize=40)



plt.legend(loc=0,fontsize=30)
plt.show()

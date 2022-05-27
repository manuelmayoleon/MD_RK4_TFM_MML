#instability_dens.py
# coding=utf-8
# from matplotlib.lines import _LineStyle
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

#!/usr/bin/python
# -*- coding: ascii -*-
import os, sys

density= pd.read_csv("densitypromy_0.95.txt",header=None,sep='\s+', names=['re','im'])

t= pd.read_csv("tiemposdecol_0.95.txt",header=None,sep='\s+', names=['t'])
ss= pd.read_csv("sparam_0.95.txt",header=None,sep='\s+', names=['s'])
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
vp= 0.0001 



dens = n/(l*(h-sigma))
lin_dens = n/l

s=sum(ss['s'])
corr_t=s/max(t['t'])


kapa=functions.Panel(1).kapa_adim(alfa,lin_dens)
mu=functions.Panel(1).mu_adim(alfa,lin_dens)

print("col_pp")
print(s)
print(corr_t)
print("modo de vibracion")
print(k)
print("densidad superficial")
print(dens)
print("kappa")

print(kapa)
print("mu")

print(mu)
print("autoval")
q=functions.lamda1(alfa,epsilon)
print(functions.lamda1(alfa,epsilon))
# densy=319.60441969489989/l
# print(densy)
# densy_teo=-0.6*(k**4*(0.3*x-y))**(1/3)

# Estos son los desarrollos en serie de los los autovalores obtenidos  en orden 2, o sea, con un error O(k^3)

eigen1=-0.333333 *1.50* (kapa - mu) *k**2


eigen2=( k*1j)-0.333333 *(0.75*(mu+kapa))*k**2


eigen3=-( k*1j)-0.333333 *(0.75*(mu+kapa))*k**2





# print(coef)
tt=np.linspace(0,max(t['t']),1000)

kk=np.linspace(0.0,0.2,1000)


##!!Ajuste lineal del perfil de densidad obtenido con MD ###

min_index = np.argmin(density['re'])
max_index = np.argmax(density['re']) 
# print(min_index) 
# print(max_index) 


##?? Regresion lineal de los datos obtenidos para n_k a traves de MD

density=density['re']+1j*density['im']
denslog=np.log(density)

x=t['t']
colpp= np.linspace(0,2772,len(denslog))

# denslog=denslog[min_index:max_index]
# x=x[min_index :max_index]
# density=density[min_index :max_index]
for i in range(2):
    density=density[min_index :max_index]
    min_index = np.argmin(density)
    max_index = np.argmax(density)
    # print(min_index)
    # print(max_index)
    colpp=colpp[3000000 :max_index]
    density=density[min_index :max_index]
    denslog=denslog[3000000:max_index]
print(colpp)
# reg = LinearRegression().fit(x.values.reshape((-1, 1)),np.real(denslog))
reg = LinearRegression().fit(colpp.reshape(-1,1),np.real(denslog))
r_sq = reg.score(colpp.reshape((-1, 1)), np.real( denslog))
print('coefficient of determination:', r_sq)
print('intercept:', reg.intercept_)
print('slope:', reg.coef_)
linear_reg=reg.coef_*colpp+reg.intercept_
##?? Prediccion teorica de la pendiente 

# tiemp=np.linspace(0.0,100.0,len(t['t']))
# densteo=np.exp(functions.eigenvalue1(functions.Panel(1).mu(alfa,lin_dens),functions.Panel(1).kapa(alfa,lin_dens),functions.lamda1(alfa,epsilon),k)*tt*dens/20)
print('Theoretical slope:', (functions.eigenvalue2(mu,kapa,q,k)*(np.sqrt(2/np.pi)*(1+alfa)*sigma*epsilon*rho)/lin_dens))


##!!Representación del perfil de densidad ###
##**
##????
##///  tachado


# plt.plot(tt,densteo,color='C2',label="$n_y$ ")
# plt.plot(x,density,color='C1',label="$n_{\frac{\pi}{L}}$ (MD)")   
plt.plot(colpp,denslog,color='C1',label="$n_{\frac{\pi}{L}}$ (MD)")   
plt.plot(colpp,linear_reg,color='C2',label="$n_{\frac{\pi}{L}}$ (MD)")   
# print((functions.eigenvalue3(functions.Panel(1).mu(alfa),functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),kk)).real)
# plt.plot(tt,expdens,color='C2',label="$n_y \;(k=2\pi/L)$ ")

# print(functions.Panel(1,sigma,epsilon,vp).T_s(alfa,lin_dens))


# plt.plot(kk,functions.eigenvalue1(functions.Panel(1,sigma,epsilon,vp).mu_adim(alfa,lin_dens),functions.Panel(1,sigma,epsilon,vp).kapa_adim(alfa,lin_dens),functions.lamda1(alfa,epsilon),kk),color='C2',label="$\lambda_1$ ")
# plt.plot(kk,functions.eigenvalue1(functions.Panel(1,sigma,epsilon,vp).mu_adim_max(alfa,lin_dens),functions.Panel(1,sigma,epsilon,vp).kapa_adim_max(alfa,lin_dens),functions.lamda1(alfa,epsilon),kk),color='C2',linestyle="--",label="$\lambda_1(a_2=0)$ ")
# plt.plot(kk,functions.eigen1_mathematica(functions.Panel(1,sigma,epsilon,vp).mu_adim(alfa,lin_dens),functions.Panel(1,sigma,epsilon,vp).kapa_adim(alfa,lin_dens),functions.lamda1(alfa,epsilon),kk),color='C2',linestyle=":",label="$\lambda_1$ ")
# plt.plot(kk,functions.eigen1_mathematica(functions.Panel(1,sigma,epsilon,vp).mu_adim_max(alfa,lin_dens),functions.Panel(1,sigma,epsilon,vp).kapa_adim_max(alfa,lin_dens),functions.lamda1(alfa,epsilon),kk),color='C2',linestyle="--",label="$\lambda_1(a_2=0)$ ")

# plt.plot(kk,functions.eigen1_taylor(functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),kk),color='C2',linestyle="-.",label="$\lambda_1 \; \mathcal{O}(k^3)$ ")
# plt.plot(kk,np.real(functions.eigenvalue2(functions.Panel(1,sigma,epsilon,vp).mu_adim(alfa,lin_dens),functions.Panel(1,sigma,epsilon,vp).kapa_adim(alfa,lin_dens),functions.lamda1(alfa,epsilon),kk)),linewidth=1.5,color='C4',label="$\lambda_2$ ")
# plt.plot(kk,np.real(functions.eigenvalue2(functions.Panel(1,sigma,epsilon,vp).mu_adim_max(alfa,lin_dens),functions.Panel(1,sigma,epsilon,vp).kapa_adim_max(alfa,lin_dens),functions.lamda1(alfa,epsilon),kk)),linewidth=1.5,linestyle="--",color='C4',label="$\lambda_2(a_2=0)$ ")
# plt.plot(kk,functions.eigen2_taylor(functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),kk),color='C4',linestyle="-.",label="$\lambda_2 \; \mathcal{O}(k^3)$ ")
# plt.plot(kk,functions.eigenvalue3(functions.Panel(1,sigma,epsilon,vp).mu_adim(alfa,lin_dens),functions.Panel(1,sigma,epsilon,vp).kapa_adim(alfa,lin_dens),functions.lamda1(alfa,epsilon),kk),color='C3',label="$\lambda_3$ ")
# plt.plot(kk,functions.eigenvalue3(functions.Panel(1,sigma,epsilon,vp).mu_adim_max(alfa,lin_dens),functions.Panel(1,sigma,epsilon,vp).kapa_adim_max(alfa,lin_dens),functions.lamda1(alfa,epsilon),kk),color='C3',linestyle="--",label="$\lambda_3(a_2=0)$ ")
# plt.plot(kk,functions.eigen3_taylor(functions.Panel(1).kapa(alfa),functions.lamda1(alfa,epsilon),kk),color='C3',linestyle="-.",label="$\lambda_3 \; \mathcal{O}(k^3)$ ")
plt.grid(color='k', linestyle='--', linewidth=0.5,alpha=0.2)
# plt.xlabel ( r' $k$ ', fontsize=30)
# plt.ylabel ( r' $\lambda$ ',rotation=0.0,fontsize=30)
plt.xlabel ( r'$s$', fontsize=30)
plt.ylabel ( r' $n_2$ ',rotation=0.0,fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.title ( r' \textbf {Autovalores en función de k.  }  ',fontsize=40)



plt.legend(loc=0,fontsize=30)
plt.show()

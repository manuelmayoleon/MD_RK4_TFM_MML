#functions.py
import numpy as np
import pandas as pd
import scipy.special as special
from numba import jit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib.transforms import (
    Bbox, TransformedBbox, blended_transform_factory)
# Import math Library
import math 

from sympy import *

# PARA USAR A LA HORA DE GUARDAR DOS COLUMNAS EN UN ARCHIVO
import csv

import cmath


np.seterr(divide='ignore', invalid='ignore')

##!! Autovalores de las ecuaciones de temperatura

def lamda1(a,e):
    
    return np.sqrt(-16*(a*e**2.0 - 3*a + 3)*(9*a**3*e**4.0 - 3*a**3*e**6.0 + 72*a**2*e**2.0 - 45*a**2*e**4.0 + 4*a**2*e**6.0 - 144*a*e**2.0 + 39*a*e**4.0 - a*e**6.0 + 72*e**2.0 - 3*e**4.0) + (-48*a**2*e**2.0 + 11*a**2*e**4.0 + 72*a**2 + 48*a*e**2.0 + 4*a*e**4.0 - 144*a + e**4.0 + 72)**2)/(48*(a*e**2.0 - 3*a + 3)) + (48*a**2*e**2.0 - 11*a**2*e**4.0 - 72*a**2 - 48*a*e**2.0 - 4*a*e**4.0 + 144*a - e**4.0 - 72)/(48*(a*e**2.0 - 3*a + 3))

def lamda2(a,e):            
        return -np.sqrt(-16*(a*e**2.0 - 3*a + 3)*(9*a**3*e**4.0 - 3*a**3*e**6.0 + 72*a**2*e**2.0 - 45*a**2*e**4.0 + 4*a**2*e**6.0 - 144*a*e**2.0 + 39*a*e**4.0 - a*e**6.0 + 72*e**2.0 - 3*e**4.0) + (-48*a**2*e**2.0 + 11*a**2*e**4.0 + 72*a**2 + 48*a*e**2.0 + 4*a*e**4.0 - 144*a + e**4.0 + 72)**2)/(48*(a*e**2.0 - 3*a + 3)) + (48*a**2*e**2.0 - 11*a**2*e**4.0 - 72*a**2 - 48*a*e**2.0 - 4*a*e**4.0 + 144*a - e**4.0 - 72)/(48*(a*e**2.0 - 3*a + 3))

def eigen1_mathematica(mu,kapa,q,k):
        
       
                return  -0.333333* (1.* k**2 *kapa + 1.* q) - (0.419973683298291*(3.* k**2 - (1.* k**2* kapa + 
                        1.* q)**2))/(-4.5* k**4* kapa - 2.* k**6 *kapa**3 + 13.5* k**4* mu + 
                        22.5 *k**2* q - 6.* k**4* kapa**2* q - 6.* k**2* kapa* q**2 - 
                        2.* q**3 + np.sqrt((-4.5 *k**4 *kapa - 2.* k**6* kapa**3 + 13.5* k**4* mu + 
                        22.5* k**2* q - 6.* k**4* kapa**2* q - 6.* k**2 *kapa* q**2 - 2. *q**3+0j)**2 + 
                        4* (3.* k**2 - (1.* k**2* kapa + 1.* q)**2 +0j)**3)+0j)**(1/3) + 0.26456684199469993 *(-4.5* k**4* kapa - 2. *k**6* kapa**3 + 13.5* k**4* mu + 22.5* k**2* q - 
                        6. *k**4* kapa**2* q - 6. *k**2* kapa* q**2 - 
                        2.* q**3 +np.sqrt((-4.5 *k**4* kapa - 2.* k**6* kapa**3 + 13.5* k**4* mu + 
                        22.5* k**2* q - 6. *k**4* kapa**2* q - 6. *k**2* kapa* q**2 - 2. *q**3+0j)**2 + 
                        4 *(3.* k**2 - (1.* k**2* kapa + 1.* q + 0j)**2 +0j)**3)+0j)**(1/3)

##!! Autovalores del analisis de estabilidad del modelo hidrodinamico (python)

def eigenvalue1(mu,kapa,q,k):
        return -k**2*kapa/3 - q/3 - 0.176377894663133*(-3.0*k**2 + (k**2*kapa + q)**2)/(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2+0j)**0.5+0j)**(1/3) - 0.629960524947436*(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2+0j)**(1/2)+0j)**(1/3)
def eigenvalue2(mu,kapa,q,k):
    return -0.333333333333333*k**2*kapa - 0.333333333333333*q - 0.176377894663133*(-0.5 - 0.866025403784439*1j)*(-3.0*k**2 + (k**2*kapa + q)**2)/(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2+0j)**0.5+0j)**(1/3) - 0.629960524947436*(-0.5 + 0.866025403784439*1j)*(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2+0j)**0.5+0j)**(1/3)
def eigenvalue3(mu,kapa,q,k):
    return-0.333333333333333*k**2*kapa - 0.333333333333333*q - 0.176377894663133*(-0.5 + 0.866025403784439*1j)*(-3.0*k**2 + (k**2*kapa + q)**2)/(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2+0j)**0.5+0j)**(1/3) - 0.629960524947436*(-0.5 - 0.866025403784439*1j)*(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2+0j)**0.5+0j)**(1/3)
# def eigenvalue2(mu,kapa,q,k):
#     return -0.333333333333333*k**2*kapa - 0.333333333333333*q - 0.176377894663133*(-0.5 - 0.866025403784439*1j)*(-3.0*k**2 + (k**2*kapa + q)**2+0j)/(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2+0j)**0.5+0j )**(1/3) - 0.629960524947436*(-0.5 + 0.866025403784439*1j)*(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + 
#         q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2+0j)**0.5+0j)**(1/3)
# def eigenvalue3(mu,kapa,q,k):
#        return -0.333333333333333*k**2*kapa - 0.333333333333333*q - 0.176377894663133*(-0.5 + 0.866025403784439*I)*(-3.0*k**2 + (k**2*kapa + q)**2)/(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2)**0.5)**(1/3) - 0.629960524947436*(-0.5 - 0.866025403784439*I)*(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2)**0.5)**(1/3)
def eigen3_math(mu,kapa,q,k):
                return -0.3333333333333333 * (1.*  k**2* kapa + 1.  *q) + ((0.2099868416491455  - 0.3637078786572404  *1j)* (3. * k**2 - (1. *  k**2* kapa + 1. * q)**2))/(-4.5*  k**4 *kapa - 2. * k**6* kapa**3 + 13.5*  k**4 *mu + 
                        22.5 * k**2* q - 6.*  k**4* kapa**2* q - 6. * k**2* kapa* q**2 - 2. * q**3 + np.sqrt((-4.5*  k**4* kapa - 2.*  k**6* kapa**3 + 
                        13.5 * k**4 *mu + 22.5  *k**2* q - 6.  *k**4 *kapa**2 *q - 6.*  k**2 *kapa* q**2 - 2.*q**3 +0j)**2 + 
                        4* (3.*  k**2 - (1. * k**2* kapa + 1.*  q)**2)**3+0j)+0j)**(
                        1/3) - (0.13228342099734997  + 
                        0.22912160616643376  *1j)* (-4.5*  k**4 *kapa - 2.*  k**6* kapa**3 + 
                        13.5 * k**4* mu + 22.5*  k**2* q - 6.*  k**4* kapa**2* q - 
                        6. * k**2* kapa *q**2 - 
                        2.*  q**3 + np.sqrt((-4.5 * k**4 *kapa - 2.  *k**6* kapa**3 + 
                        13.5 * k**4 *mu + 22.5 * k**2 *q - 6. * k**4 *kapa**2* q - 
                        6. * k**2* kapa* q**2 - 2. * q**3)**2 + 
                        4 *(3.*  k**2 - (1.*  k**2 *kapa + 1. * q)**2 +0j)**3 +0j)+0j)**(1/3)


##!! Autovalores aproximados calculados a mano

def eigenvalue1_taylor(kapa,q,k):
        return -q*(2+(-1)**(2/3))/3- 1j*0.408248*(1-(-1)**(2/3))*k -(-2.25*(1+(-1)**(2/3))+(2+(-1)**(2/3))*kapa*q)*k**2/(3*q)

def eigenvalue2_taylor(kapa,q,k):
        return q *(0.166667 + 0.288675 *1j)*(-0.25  +  (-1)**(2/3)) - 1j*k*(0.166667 - 0.288675 *1j) +k**2*(0.166667 + 0.288675 *1j)*((1.125*1j + 1.94856)+((-1)**(2/3)*1j-0.25*1j)*kapa*q-2.25*1j*(-1)**(2/3))/q

##!! Autovalores aproximados en serie de Taylor calculados con mathematica 

def eigen1_taylor(kapa,q,k):
        return -(1/(-q**3+0j)**(1/3))*(0.166667 - 0.288675*1j)*((-0.5 + 0.866025 *1j)*q**2 + (0.5 + 0.866025*1j)*q*(-q**3+0j)**(1/3) +  (-q**3+0j)**(2/3)) + (1/(k* q**3* (-q**3+0j)**(1/3)))*(0.204124 - 0.353553 *1j)* np.sqrt(-k**2* q**4+0j)* ((0.5 - 0.866025*1j )* q**2 + 
                (-q**3+0j)**(2/3))*k+1/(q**2* (-q**3+0j)**(1/3))*(0.288675 + 0.166667 *1j)*((1.94856 + 1.125 *1j)* q**2 - (0.866025 + 0.5 *1j) *kapa* q**3 - (0.866025 - 0.5 *1j) *kapa* q**2 *(-q**3+0j)**(
                1/3) - (0. + 2.25 *1j)* (-q**3+0j)**(2/3) + (0. + 1. *1j)* kapa* q *(-q**3+0j)**(2/3))* k**2
def eigen1_taylormath(kapa,q,k):
      return   (0.33333333333333337*(0.9999999999999999* q**2 - 
                        0.9999999999999999* q *(-q**3)**(1/3) + 1.* (-q**3)**(2/3)))/(-q**3)**(1/3) - (0.408248290463863* (1j*np.sqrt(k**2*q**4)* (-0.9999999999999999* q**2 
                        + 1.* (-q**3)**(2/3)))* k)/(k* q**3 (-q**3)**(1/3)) + 1/(q**2 *(-q**3)**(1/3))*0.33333333333333337* (-2.2499999999999996* q**2 + 1.* kapa* q**3 - 0.9999999999999999* kapa* q**2 (-q**3)**(1/3) - 2.25* (-q**3)**(2/3) + 
                        1.* kapa * q * (-q**3)**(2/3)) * k**2

def eigen2_taylor(kapa,q,k):
        return (1/((-q**3+0j)**(1/3)))*0.333333* (q**2 -  q* (-q**3+0j)**(1/3) +  (-q**3+0j)**(2/3)) - (0.408248 *(np.sqrt(-k**2* q**4+0j)* (- q**2 +  (-q**3)**(2/3)))* k)/(k *q**3 *(-q**3)**(1/3)) + 1/(q**2* (-q**3+0j)**(1/3)+0j)*0.333333* (-2.25* q**2 +  
        kapa* q**3 -  kapa* q**2 *(-q**3)**(1/3) -2.25 *(-q**3+0j)**(2/3) +  kapa* q *(-q**3+0j)**(2/3))*k**2

def eigen3_taylor(kapa,q,k):
     return -(1/(-q**3+0j)**(1/3)) *(0.166667 + 0.288675 *1j) *((-0.5 - 0.866025 *1j) *q**2 + (0.5 - 0.866025 *1j)* q* (-q**3+0j)**(1/3) +  (-q**3+0j)**(2/3)) + (0.204124 + 0.353553 *1j)* np.sqrt(-k**2 *q**4+0j)* ((0.5 + 0.866025 *1j) *q**2 + 
        ((-q**3+0j)**(2/3)+0j))* k/(k *q**3* (-q**3+0j)**(1/3)+0j) - 1/(q**2* (-q**3+0j)**(1/3)) *(0.166667 + 0.288675 *1j)* ((1.125 + 1.94856 *1j)* q**2 - (0.5 + 
       0.866025 *1j)* kapa* q**3 + (0.5 - 0.866025 *1j)* kapa* q**2 *(-q**3+0j)**(1/3) - (2.25+0j)*(-q**3+0j)**(2/3) + (1+0j)* kapa *q* (-q**3+0j)**(2/3))*  k**2

def hcs1(kapa,mu,eta,k):
        return -0.5*k**2*kapa - 0.111111111111111*(3.0*eta**2 + 4.5*eta*k**2*kapa + 2.25*k**4*kapa**2 - 9.0*k**2)/(-0.25*eta*k**2 + 0.125*k**6*kapa**3 + 0.75*k**4*kapa - 0.75*k**4*mu - 0.25*k**2*kapa*(-eta**2 - 1.5*eta*k**2*kapa + 3.0*k**2) + (-(0.333333333333333*eta**2 + 0.5*eta*k**2*kapa + 0.25*k**4*kapa**2 - k**2)**3 + 0.5625*(-0.333333333333333*eta*k**2 + 0.166666666666667*k**6*kapa**3 + k**4*kapa - k**4*mu - 0.333333333333333*k**2*kapa*(-eta**2 - 1.5*eta*k**2*kapa + 3.0*k**2))**2)**0.5)**(1/3) - 1.0*(-0.25*eta*k**2 + 0.125*k**6*kapa**3 + 0.75*k**4*kapa - 0.75*k**4*mu - 0.25*k**2*kapa*(-eta**2 - 1.5*eta*k**2*kapa + 3.0*k**2) + (-(0.333333333333333*eta**2 + 0.5*eta*k**2*kapa + 0.25*k**4*kapa**2 - k**2)**3 + 0.5625*(-0.333333333333333*eta*k**2 + 0.166666666666667*k**6*kapa**3 + k**4*kapa - k**4*mu - 0.333333333333333*k**2*kapa*(-eta**2 - 1.5*eta*k**2*kapa + 3.0*k**2))**2)**0.5)**(1/3)
def hcs2(kapa,mu,eta,k):
        return -0.5*k**2*kapa - 0.111111111111111*(-0.5 - 0.866025403784439*1j)*(3.0*eta**2 + 4.5*eta*k**2*kapa + 2.25*k**4*kapa**2 - 9.0*k**2)/(-0.25*eta*k**2 + 0.125*k**6*kapa**3 + 0.75*k**4*kapa - 0.75*k**4*mu - 0.25*k**2*kapa*(-eta**2 - 1.5*eta*k**2*kapa + 3.0*k**2) + (-(0.333333333333333*eta**2 + 0.5*eta*k**2*kapa + 0.25*k**4*kapa**2 - k**2)**3 + 0.5625*(-0.333333333333333*eta*k**2 + 0.166666666666667*k**6*kapa**3 + k**4*kapa - k**4*mu - 0.333333333333333*k**2*kapa*(-eta**2 - 1.5*eta*k**2*kapa + 3.0*k**2))**2)**0.5)**(1/3) - 1.0*(-0.5 + 0.866025403784439*1j)*(-0.25*eta*k**2 + 0.125*k**6*kapa**3 + 0.75*k**4*kapa - 0.75*k**4*mu - 0.25*k**2*kapa*(-eta**2 - 1.5*eta*k**2*kapa + 3.0*k**2) + (-(0.333333333333333*eta**2 + 0.5*eta*k**2*kapa + 0.25*k**4*kapa**2 - k**2)**3 + 0.5625*(-0.333333333333333*eta*k**2 + 0.166666666666667*k**6*kapa**3 + k**4*kapa - k**4*mu - 0.333333333333333*k**2*kapa*(-eta**2 - 1.5*eta*k**2*kapa + 3.0*k**2))**2+0j)**0.5)**(1/3)
def hcs3(kapa,mu,eta,k):
        return -0.5*k**2*kapa - 0.111111111111111*(-0.5 + 0.866025403784439*1j)*(3.0*eta**2 + 4.5*eta*k**2*kapa + 2.25*k**4*kapa**2 - 9.0*k**2)/(-0.25*eta*k**2 + 0.125*k**6*kapa**3 + 0.75*k**4*kapa - 0.75*k**4*mu - 0.25*k**2*kapa*(-eta**2 - 1.5*eta*k**2*kapa + 3.0*k**2) + (-(0.333333333333333*eta**2 + 0.5*eta*k**2*kapa + 0.25*k**4*kapa**2 - k**2)**3 + 0.5625*(-0.333333333333333*eta*k**2 + 0.166666666666667*k**6*kapa**3 + k**4*kapa - k**4*mu - 0.333333333333333*k**2*kapa*(-eta**2 - 1.5*eta*k**2*kapa + 3.0*k**2))**2)**0.5)**(1/3) - 1.0*(-0.5 - 0.866025403784439*1j)*(-0.25*eta*k**2 + 0.125*k**6*kapa**3 + 0.75*k**4*kapa - 0.75*k**4*mu - 0.25*k**2*kapa*(-eta**2 - 1.5*eta*k**2*kapa + 3.0*k**2) + (-(0.333333333333333*eta**2 + 0.5*eta*k**2*kapa + 0.25*k**4*kapa**2 - k**2)**3 + 0.5625*(-0.333333333333333*eta*k**2 + 0.166666666666667*k**6*kapa**3 + k**4*kapa - k**4*mu - 0.333333333333333*k**2*kapa*(-eta**2 - 1.5*eta*k**2*kapa + 3.0*k**2))**2+0j)**0.5)**(1/3)
##!! Coeficientes de transporte 
 
def a2(alpha):    
     return  16*(1-2*alpha**2.00)/(33+30*alpha**2.00)
class Panel:
    def __init__(self,d=1,sigma=1.0,epsilon= 0.5 ,vp=0.0001):
        self.d = d
        self.sigma = sigma
        self.vp=  vp
        self.epsilon = epsilon 
        self.eta0 = (2+self.d)*np.sqrt(np.pi/2)/8
        self.kapa0= (2+self.d)*self.eta0/2
    def a2d(self,alpha):
            return  16*(1-2*alpha**2.00)*(1-alpha)/(9+24*self.d+(8*self.d-41)*alpha+30*alpha**2.00*(1-alpha))
    def eta(self, alpha):
            return ((2+self.d)/4*self.d)*(1-alpha**2.00)*(1+3*self.a2d(alpha)/16)
    def eta_max(self, alpha):
            return ((2+self.d)/4*self.d)*(1-alpha**2.00)
    def gamma(self,alpha):
            return (4*alpha*self.epsilon**2.00+12*(1-alpha))/((1+3*alpha)*self.epsilon**2.00)
    def T_s(self,alpha,density):
            return ((3*np.sqrt(np.pi)*self.gamma(alpha))/((1+alpha)*(self.gamma(alpha)-(1+alpha)/2)*self.epsilon**3.00*density*self.sigma))**2.00*self.vp**2
    def nu(self,alpha):
            return (1+alpha)*((self.d-1)/2+3*(self.d+8)*(1-alpha)+(4+5*self.d-3*(4-self.d)*alpha)*self.a2d(alpha)/512)
    def nu_max(self,alpha):
            return (1+alpha)*((self.d-1)/2+3*(self.d+8)*(1-alpha))
    def kapa(self,alpha):
            return (1+2*self.a2d(alpha))/(self.nu(alpha)+2*self.d*self.eta(alpha))*self.kapa0
    def mu(self,alpha):
            return 2*self.eta(alpha)*(self.kapa(alpha)+self.a2d(alpha)/(self.d*self.eta(alpha)))/(2*(self.d-1)*self.nu(alpha)/self.d-3*self.eta(alpha))*self.kapa0
    def kapa_max(self,alpha):
            return (1)/(self.nu_max(alpha)+2*self.d*self.eta_max(alpha))
    def mu_max(self,alpha):
            return 2*self.eta_max(alpha)*(self.kapa_max(alpha))/(2*(self.d-1)*self.nu_max(alpha)/self.d-3*self.eta_max(alpha))
    def kc(self,alpha):
            return np.sqrt(2/(self.d+2)*(self.kapa(alpha)-self.mu(alpha)))
    def factor(self,lin_dens,rho,alpha):
         return (lin_dens*np.sqrt(np.pi/2)/(rho*self.epsilon*(1+alpha)*self.sigma))*(2*np.sqrt( self.T_s(alpha,lin_dens)))/(2+self.T_s(alpha,lin_dens))
            

# print(lamda1(0.95,0.5))

# print(eigenvalue1(Panel(1).mu(0.95),Panel(1).kapa(0.95),lamda1(0.95,0.5),9.424777960769378e-05))


#!!! Random colors 
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)






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

def lamda1(a,e):
    return -np.sqrt(-16*(a*e**2.0 - 3*a + 3)*(9*a**3*e**4.0 - 3*a**3*e**6.0 + 72*a**2*e**2.0 - 45*a**2*e**4.0 + 4*a**2*e**6.0 - 144*a*e**2.0 + 39*a*e**4.0 - a*e**6.0 + 72*e**2.0 - 3*e**4.0) + (-48*a**2*e**2.0 + 11*a**2*e**4.0 + 72*a**2 + 48*a*e**2.0 + 4*a*e**4.0 - 144*a + e**4.0 + 72)**2)/(48*(a*e**2.0 - 3*a + 3)) + (48*a**2*e**2.0 - 11*a**2*e**4.0 - 72*a**2 - 48*a*e**2.0 - 4*a*e**4.0 + 144*a - e**4.0 - 72)/(48*(a*e**2.0 - 3*a + 3))
def lamda2(a,e):            
        return np.sqrt(-16*(a*e**2.0 - 3*a + 3)*(9*a**3*e**4.0 - 3*a**3*e**6.0 + 72*a**2*e**2.0 - 45*a**2*e**4.0 + 4*a**2*e**6.0 - 144*a*e**2.0 + 39*a*e**4.0 - a*e**6.0 + 72*e**2.0 - 3*e**4.0) + (-48*a**2*e**2.0 + 11*a**2*e**4.0 + 72*a**2 + 48*a*e**2.0 + 4*a*e**4.0 - 144*a + e**4.0 + 72)**2)/(48*(a*e**2.0 - 3*a + 3)) + (48*a**2*e**2.0 - 11*a**2*e**4.0 - 72*a**2 - 48*a*e**2.0 - 4*a*e**4.0 + 144*a - e**4.0 - 72)/(48*(a*e**2.0 - 3*a + 3))

# def eigenvalue1(mu,kapa,q,k):
#     return -0.333333333333333*k**2*kapa - 0.333333333333333*q - 0.176377894663133*(-3.0*k**2 + (k**2*kapa + q)**2)/(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2)**0.5)**(1/3) - 0.629960524947436*(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2)**0.5)**(1/3)
# def eigenvalue1(mu,kapa,q,k):
#     return -0.333333333333333*k**2*kapa - 0.333333333333333*q - 0.176377894663133*(-3.0*k**2 + (k**2*kapa + q)**2)/(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2)**0.5)**(1/3) - 0.629960524947436*(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2)**0.5)**(1/3)
def eigenvalue1(mu,kapa,q,k):
        
       
                return  -0.333333* (1.* k**2 *kapa + 1.* q) - (0.419974 *(3.* k**2 - (1.* k**2* kapa + 
                        1.* q)**2))/(-4.5* k**4* kapa - 2.* k**6 *kapa**3 + 13.5* k**4* mu + 
                        22.5 *k**2* q - 6.* k**4* kapa**2* q - 6.* k**2* kapa* q**2 - 
                        2.* q**3 + np.sqrt((-4.5 *k**4 *kapa - 2.* k**6* kapa**3 + 13.5* k**4* mu + 
                        22.5* k**2* q - 6.* k**4* kapa**2* q - 6.* k**2 *kapa* q**2 - 2. *q**3+0j)**2 + 
                        4* (3.* k**2 - (1.* k**2* kapa + 1.* q)**2 +0j)**3)+0j)**(1/3) + 0.26456 *(-4.5* k**4* kapa - 2. *k**6* kapa**3 + 13.5* k**4* mu + 22.5* k**2* q - 
                        6. *k**4* kapa**2* q - 6. *k**2* kapa* q**2 - 
                        2.* q**3 +np.sqrt((-4.5 *k**4* kapa - 2.* k**6* kapa**3 + 13.5* k**4* mu + 
                        22.5* k**2* q - 6. *k**4* kapa**2* q - 6. *k**2* kapa* q**2 - 2. *q**3+0j)**2 + 
                        4 *(3.* k**2 - (1.* k**2* kapa + 1.* q + 0j)**2 +0j)**3)+0j)**(1/3)
     
        
def eigenvalue2(mu,kapa,q,k):
    return -0.333333333333333*k**2*kapa - 0.333333333333333*q - 0.176377894663133*(-0.5 - 0.866025403784439*1j)*(-3.0*k**2 + (k**2*kapa + q)**2+0j)/(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2+0j)**0.5+0j )**(1/3) - 0.629960524947436*(-0.5 + 0.866025403784439*1j)*(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + 
        q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2+0j)**0.5+0j)**(1/3)
def eigenvalue3(mu,kapa,q,k):
    return -0.333333333333333*k**2*kapa - 0.333333333333333*q + 0.176377894663133*(-0.5 + 0.866025403784439*1j)*(-3.0*k**2 + (k**2*kapa + q)**2+0j)/(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2+0j)**0.5+0j)**(1/3) - 0.629960524947436*(-0.5 - 0.866025403784439*1j)*(k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3 + (-0.592592592592593*(-k**2 + 0.333333333333333*(k**2*kapa + 
        q)**2)**3 + (k**4*kapa - k**4*mu - k**2*q - 0.0740740740740741*k**2*(9.0*k**2*kapa + 9.0*q) + 0.148148148148148*(k**2*kapa + q)**3)**2+0j)**0.5+0j)**(1/3)
    
    

    
def a2(alpha):    
     return  16*(1-2*alpha**2.00)/(33+30*alpha**2.00)
class Panel:
    def __init__(self,d=1):
        self.d = d
    def a2d(self,alpha):
            return  16*(1-2*alpha**2.00)*(1-alpha)/(9+24*self.d+(8*self.d-41)*alpha+30*alpha**2.00*(1-alpha))
    def eta(self, alpha):
            return ((2+self.d)/4*self.d)*(1-alpha**2.00)*(1+3*self.a2d(alpha)/16)

    def nu(self,alpha):
            return (1+alpha)*((self.d-1)/2+3*(self.d+8)*(1-alpha)+(4+5*self.d-3*(4-self.d)*alpha)*self.a2d(alpha)/512)
    def kapa(self,alpha):
            return (1+2*self.a2d(alpha))/(self.nu(alpha)+2*self.d*self.eta(alpha))
    def mu(self,alpha):
            return 2*self.eta(alpha)* (self.kapa(alpha)+self.a2d(alpha)/(self.d*self.eta(alpha)))/(2*(self.d-1)*self.nu(alpha)/self.d-3*self.eta(alpha))
    def kc(self,alpha):
            return np.sqrt(2/(self.d+2)*(self.kapa(alpha)-self.mu(alpha)))

# print(lamda1(0.95,0.5))

# print(eigenvalue1(Panel(1).mu(0.95),Panel(1).kapa(0.95),lamda1(0.95,0.5),9.424777960769378e-05))
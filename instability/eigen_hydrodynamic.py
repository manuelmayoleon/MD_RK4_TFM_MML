from numpy.core.function_base import linspace
import pandas as pd
import numpy as np
import scipy . stats as ss
import math
import matplotlib . mlab as mlab
import matplotlib . pyplot as plt
from scipy import optimize
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib.transforms import (
    Bbox, TransformedBbox, blended_transform_factory)
from sympy import *

# import scipy.linalg as la


kapa= Symbol("kapa",positive=True)
mu=Symbol("mu",positive=True)
k= Symbol("k")
q=Symbol("q")

ik=k*(1j)

# M = np.matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
M=np.matrix(( [0 ,-ik ,  0],[-ik*0.5 ,0, -ik*0.5  ],[-2*q -mu*(k**2) , -ik , -q - kapa*(k**2) ]  ))
# print(M)
# eigenval=la.eigvals(M)
# print(eigenval)

# lamda=M.eigenvals()
# print(lamda)
# data = list(lamda.items())
# # print(data)
# lamda_array=np.array(data)
# print(lamda_array)

x = Symbol('x')



# p = M.charpoly(l)
# polynomial=- 0.5 * k**4*kapa + 0.5 * k**4*mu + 0.5* k**2*q -  k**2 *x - k**2 *kapa* x**2 -  q* x**2 - x**3
polynomial=- 0.5 * k**4*kapa + 0.5 * k**4*mu  -  k**2 *x - k**2 *kapa* x**2 - x**3




quadratic_equation = Eq(polynomial, 0)

solucion=solve(quadratic_equation, x)

print(solucion[0])

# coeff = [- 0.5 * k**4*kapa + 0.5 * k**4*mu + 0.5* k**2*q , -  k**2,  - k**2 *kapa-  q, -1]

# print(np.roots(coeff))

# s=solveset(p.as_expr(),l)
# # cojer cada autovalor con .args[0] y .args[1]
# print(s.args[0])

# lam1=s.args[0]
# data = list(s.items())
# # print(data)
# lamda_array=np.array(data)
# print(lamda_array)


# def lamda1(a,e):
#     return -np.sqrt(-16*(a*e**2.0 - 3*a + 3)*(9*a**3*e**4.0 - 3*a**3*e**6.0 + 72*a**2*e**2.0 - 45*a**2*e**4.0 + 4*a**2*e**6.0 - 144*a*e**2.0 + 39*a*e**4.0 - a*e**6.0 + 72*e**2.0 - 3*e**4.0) + (-48*a**2*e**2.0 + 11*a**2*e**4.0 + 72*a**2 + 48*a*e**2.0 + 4*a*e**4.0 - 144*a + e**4.0 + 72)**2)/(48*(a*e**2.0 - 3*a + 3)) + (48*a**2*e**2.0 - 11*a**2*e**4.0 - 72*a**2 - 48*a*e**2.0 - 4*a*e**4.0 + 144*a - e**4.0 - 72)/(48*(a*e**2.0 - 3*a + 3))
# def lamda2(a,e):            
#         return np.sqrt(-16*(a*e**2.0 - 3*a + 3)*(9*a**3*e**4.0 - 3*a**3*e**6.0 + 72*a**2*e**2.0 - 45*a**2*e**4.0 + 4*a**2*e**6.0 - 144*a*e**2.0 + 39*a*e**4.0 - a*e**6.0 + 72*e**2.0 - 3*e**4.0) + (-48*a**2*e**2.0 + 11*a**2*e**4.0 + 72*a**2 + 48*a*e**2.0 + 4*a*e**4.0 - 144*a + e**4.0 + 72)**2)/(48*(a*e**2.0 - 3*a + 3)) + (48*a**2*e**2.0 - 11*a**2*e**4.0 - 72*a**2 - 48*a*e**2.0 - 4*a*e**4.0 + 144*a - e**4.0 - 72)/(48*(a*e**2.0 - 3*a + 3))
                
                
# alfa=np.linspace(0.0,1.0)

# epsilon=0.85

# fig, ax = plt.subplots(1,1)

# plt.plot(alfa,lamda2(alfa,epsilon),color="C0",label="$\lambda_1,\epsilon=0.5 $")
# plt.plot(alfa,lamda1(alfa,epsilon),color="C0",linestyle="--",label="$\lambda_2  , \epsilon=0.5 $")
# plt.plot(alfa,lamda2(alfa,0.8),color="C1",label="$\lambda_1  , \epsilon=0.8 $")
# plt.plot(alfa,lamda1(alfa,0.8),color="C1",linestyle="--",label="$\lambda_2  , \epsilon=0.8 $")

# plt.grid(color='k', linestyle='--', linewidth=0.5,alpha=0.2)
# plt.xlabel ( r' $\alpha$ ', fontsize=30)
# plt.ylabel ( r'  ',fontsize=30)
# plt.legend(loc=0,fontsize=30)
# plt.title ( r' \textbf {Autovalores de la matriz $M$}  ',fontsize=40)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.show()
import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib.transforms import (
    Bbox, TransformedBbox, blended_transform_factory)

alpha=np.linspace(0.1,0.99,100)

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

# print(Panel(1).kc(0.9))

plt.plot(alpha, Panel(1).kapa(alpha),color='C0',label=r'$\kappa(\alpha), \, d=1$ ')
plt.plot(alpha, Panel(1).mu(alpha),color='C1',label=r'$\mu(\alpha),\, d=1$ ')

# plt.plot(alpha, Panel(1).kc(alpha),color='C0',label=r'$k_c(\alpha),\, d=1$ ')


# plt.plot(a, a2(a),color='C0',label=r'$a_2( \alpha )\, d=1$ ')

# plt.plot(a, Panel.a2d(a,2),color='C2',label=r'$a_2( \alpha )\, d=2$ ')

# plt.plot(a, Panel.a2d(a,3),color='C3',label=r'$a_2( \alpha )\, d=3$ ')
plt.legend(loc=0,fontsize=30)
plt.grid(color='k', linestyle='--', linewidth=0.5,alpha=0.2)

plt.xlabel( r' $\alpha$ ', fontsize=30)

plt.show()
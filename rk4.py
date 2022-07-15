import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib.transforms import (
    Bbox, TransformedBbox, blended_transform_factory)
import functions

temp= pd.read_csv("temperaturas_0.995_0.50.txt" ,header=None,sep='\s+' ,names= ["y","z"])
tiempo= pd.read_csv("tiemposdecol_0.995.txt",names=["t"])
# datos= pd.read_csv("data.txt")
 
n=500
h=1.5
alfa=0.995
epsilon=0.5
rho=0.03
vp=0.0001
l=n/(rho*(h-1.0))       
k=2*np.pi/l
lin_dens=n/l             
ts = functions.Panel(1).T_s(alfa,lin_dens) #!! Esto cambia en funcion de alfa
print("temp_s")
print(ts)
gamma = functions.Panel(1).gamma(alfa)
# print(datos)                   
#?? Datos para el método RK

# x0=1.0
# y0=5.0
# x0=ts+k
# y0=gamma*x0
x0= temp['y'][0]
y0=temp['z'][0]

a=0
b=int(tiempo["t"][len(tiempo)-1])
# b=30050
h=0.2

# print(temp)
# print(tiempo)

class Panel:
    def __init__(self,alfa=0.95,epsilon= 0.5,rho=0.03 ,vp=0.0001):
        self.rho = rho
        self.vp=  vp
        self.epsilon = epsilon 
        self.alfa = alfa
        
    def f(self,t1,t2,t):
        
        # return np.sqrt(np.pi)*(1+alfa)*epsilon*rho*np.sqrt(t1)*( -(1-alfa)*t1+epsilon**2.0*(-(5*alfa-1)*t1 +(3*alfa+1)*t2 )/12)
        return 2.0*(1+self.alfa)*self.epsilon*self.rho*np.sqrt(t1)*( -(1-self.alfa)*t1+self.epsilon**2.0*(-self.alfa*t1*4.0 +(3.0*self.alfa+1.0)*t2 )/12)/np.sqrt(np.pi)
        # return 4.0*epsilon**3.0*rho*np.sqrt(tx)*( ty -tx )/(3.0*np.sqrt(np.pi))

    def g(self,t1,t2,t):
       
        # return   2.0*np.sqrt(np.pi)*(1+alfa)*epsilon**3.0*rho*np.sqrt(t1)*( 0.5*(1+alfa)*t1-t2)/3.0 +2.0*vp*t2/epsilon
        return   2.0*(1+self.alfa)*self.epsilon**3.0*self.rho*np.sqrt(t1)*( 0.5*(1+self.alfa)*t1-t2)/(3.0*np.sqrt(np.pi)) +2.0*self.vp*t2/self.epsilon
        
    # return -4.0*epsilon**3.0*rho*np.sqrt(tx)*( ty -tx )/(3.0*np.sqrt(np.pi))
# tiemp=np.linspace(0.0,30050.0,len(temp))
plt.plot(tiempo["t"],temp["z"],color='C0',label="$T_z$ (MD)")
plt.plot(tiempo["t"],temp["y"],color='C1',label="$T_y$ (MD)")      
plt.grid(color='k', linestyle='--', linewidth=0.5,alpha=0.2)
plt.xlabel ( r' $t(T_0/m\sigma^2)^{1/2}$ ', fontsize=30)
plt.ylabel ( r' $T$ ',rotation=0.0,fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.title ( r' \textbf {Solución de las ecuaciones para las temperaturas}  ',fontsize=40)


@jit
def runge_kutta_system(f, g, x0, y0, a, b, h):
    t = np.arange(a, b + h, h)
    n = len(t)
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = x0
    y[0] = y0
    for i in range(n - 1):
        k1 = h * f(x[i], y[i], t[i])
        l1 = h * g(x[i], y[i], t[i])
        k2 = h * f(x[i] + k1 / 2, y[i] + l1 / 2, t[i] + h / 2)
        l2 = h * g(x[i] + k1 / 2, y[i] + l1 / 2, t[i] + h / 2)
        k3 = h * f(x[i] + k2 / 2, y[i] + l2 / 2, t[i] + h / 2)
        l3 = h * g(x[i] + k2 / 2, y[i] + l2 / 2, t[i] + h / 2)
        k4 = h * f(x[i] + k3, y[i] + l3, t[i] + h)
        l4 = h * g(x[i] + k3, y[i] + l3, t[i] + h)
        x[i + 1] = x[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + 2 * k4)
        y[i + 1] = y[i] + (1 / 6) * (l1 + 2 * l2 + 2 * l3 + 2 * l4)
    plt.plot(t, x,color='C2',label='$T_y$ ')
    plt.plot(t, y,color='C3',label='$T_z$')
    print(min(x))
    plt.legend(loc=0,fontsize=30)
    plt.show()
# np.seterr('raise')
runge_kutta_system(Panel(alfa,epsilon,rho,vp).f,Panel(alfa,epsilon,rho,vp).g,x0,y0,a,b,h)


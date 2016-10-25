# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:37:57 2016

@author: Eric Schmidt
"""


import numpy as np
import matplotlib.pyplot as plt

R0 = 0.063 # (m) Pipe radius
Ri = 0.0024 # (m) Wire radius
d = 0.026
phi = np.pi/4

V0 = 30000

N = 20

rho_array = np.arange(0,R0,0.005)
theta_array = np.arange(0,2*np.pi,0.05)

X_array = np.arange(0,R0,0.01)
Y_array = np.arange(0,2*np.pi,0.1)

E_x = np.zeros((N,N))
E_y = np.zeros((N,N))

def main():
    EE = getMagE(np.pi/2,d)
    print(EE)

#E_r = np.zeros((N,N))
#E_theta = np.zeros((N,N))
#E_r1 = np.zeros((N,N))
#E_theta1 = np.zeros((N,N))
#E_r2 = np.zeros((N,N))
#E_theta2 = np.zeros((N,N))
#E_r3 = np.zeros((N,N))
#E_theta3 = np.zeros((N,N))
#E_r4 = np.zeros((N,N))
#E_theta4 = np.zeros((N,N))
    
def getTheta(theta, rho):
    
    E_theta1 = ((d*(d-R0)*(d+R0)*(R0-rho)*(R0+rho)*V0*np.sin(theta-1*np.pi/4)) / 
    ((d**2+rho**2-2*d*rho*np.cos(theta-1*np.pi/4))*(R0**4+d**2*rho**2-2*d*R0**2*rho*np.cos(theta-1*np.pi/4)) * (np.log(R0/d) - 
    np.log((R0**2-d*(d+Ri))/(d*Ri))))
    )
    
    E_theta2 = -((d*(d-R0)*(d+R0)*(R0-rho)*(R0+rho)*V0*np.sin(theta-3*np.pi/4)) / 
    ((d**2+rho**2-2*d*rho*np.cos(theta-3*np.pi/4))*(R0**4+d**2*rho**2-2*d*R0**2*rho*np.cos(theta-3*np.pi/4)) * (np.log(R0/d) - 
    np.log((R0**2-d*(d+Ri))/(d*Ri))))
    )
    
    E_theta3 = ((d*(d-R0)*(d+R0)*(R0-rho)*(R0+rho)*V0*np.sin(theta-5*np.pi/4)) / 
    ((d**2+rho**2-2*d*rho*np.cos(theta-5*np.pi/4))*(R0**4+d**2*rho**2-2*d*R0**2*rho*np.cos(theta-5*np.pi/4)) * (np.log(R0/d) - 
    np.log((R0**2-d*(d+Ri))/(d*Ri))))
    )
    
    E_theta4 = -((d*(d-R0)*(d+R0)*(R0-rho)*(R0+rho)*V0*np.sin(theta-7*np.pi/4)) / 
    ((d**2+rho**2-2*d*rho*np.cos(theta-7*np.pi/4))*(R0**4+d**2*rho**2-2*d*R0**2*rho*np.cos(theta-7*np.pi/4)) * (np.log(R0/d) - 
    np.log((R0**2-d*(d+Ri))/(d*Ri))))
    )
    
    E_theta = E_theta1 + E_theta2 + E_theta3 + E_theta4
#    E_theta = E_theta1
    
#    E_theta[E_theta > 5*10**6] = 5*10**6
    
    return E_theta
    
def getRho(theta, rho):
    
    E_r1 = -(((-d**4+R0**4)*rho*V0 + d*(d-R0)*(d+R0)*(R0**2+rho**2)*V0*np.cos(theta-1*np.pi/4)) / 
    ((d**2+rho**2-2*d*rho*np.cos(theta-1*np.pi/4))*(R0**4+d**2*rho**2-2*d*R0**2*rho*np.cos(theta-1*np.pi/4))*(np.log(R0/d) - 
    np.log((R0**2-d*(d+Ri))/(d*Ri))))
    )
    
    E_r2 = (((-d**4+R0**4)*rho*V0 + d*(d-R0)*(d+R0)*(R0**2+rho**2)*V0*np.cos(theta-3*np.pi/4)) / 
    ((d**2+rho**2-2*d*rho*np.cos(theta-3*np.pi/4))*(R0**4+d**2*rho**2-2*d*R0**2*rho*np.cos(theta-3*np.pi/4))*(np.log(R0/d) - 
    np.log((R0**2-d*(d+Ri))/(d*Ri))))
    )
    
    E_r3 = -(((-d**4+R0**4)*rho*V0 + d*(d-R0)*(d+R0)*(R0**2+rho**2)*V0*np.cos(theta-5*np.pi/4)) / 
    ((d**2+rho**2-2*d*rho*np.cos(theta-5*np.pi/4))*(R0**4+d**2*rho**2-2*d*R0**2*rho*np.cos(theta-5*np.pi/4))*(np.log(R0/d) - 
    np.log((R0**2-d*(d+Ri))/(d*Ri))))
    )
    
    E_r4 = (((-d**4+R0**4)*rho*V0 + d*(d-R0)*(d+R0)*(R0**2+rho**2)*V0*np.cos(theta-7*np.pi/4)) / 
    ((d**2+rho**2-2*d*rho*np.cos(theta-7*np.pi/4))*(R0**4+d**2*rho**2-2*d*R0**2*rho*np.cos(theta-7*np.pi/4))*(np.log(R0/d) - 
    np.log((R0**2-d*(d+Ri))/(d*Ri))))
    )
    
    E_r = E_r1 + E_r2 + E_r3 + E_r4
#    E_r = E_r1
    
#    E_r[E_r > 5*10**6] = 5*10**6
    
    return E_r
    
def getMagE(theta, rho):
    
    E_theta = getTheta(theta, rho)
    E_r = getRho(theta, rho)
    
#    E_x = E_r*np.cos(theta) + E_theta*np.sin(theta)
#    E_x[E_x > 5*10**6] = 5*10**6
#    E_y = E_r*np.sin(theta) - E_theta*np.cos(theta)
#    E_y[E_y > 5*10**6] = 5*10**6
#    E_r = np.sqrt(E_theta**2 + E_r**2)
#    E = E_theta
    
#    E_x = E_r*np.cos(theta)
#    E_y = E_r*np.sin(theta)
#    E = np.sqrt(E_x**2 + E_y**2)
    E = np.sqrt(E_theta**2 + E_r**2)
#    E = E_y

    
#    E = np.array([E_x,E_y,0])
#    E[E > 5*10**6] = 5*10**6
    print(E)
    
    return E
    
#### Plotting ####
    
t, r = np.meshgrid(theta_array,rho_array) 
#X, Y = np.meshgrid(X_array, Y_array)

#U = getTheta(t,r)
#V = getRho(t,r)
#
#Norm = np.sqrt(U**2 + V**2)
#
#dx = 1
#dy = 1
#
#f = plt.figure()
#ax = f.add_subplot(111)
##ax.quiver(r*np.cos(t), r*np.sin(t), dx*U/Norm, dy*V/Norm)
#ax.quiver(X, Y, dx*U/Norm, dy*V/Norm)

axs = plt.subplot(111,projection='polar')
#p1 = axs.contourf(r*np.cos(t), r*np.sin(t), getMagE(t,r), 500)
#p1 = axs.contourf(X, Y, getMagE(t,r), 500)
p1 = axs.contourf(t,r,getMagE(t,r),500)
cbar = plt.colorbar(p1, ax=axs)

plt.show()
    
#t, r = np.meshgrid(theta_array,rho_array) 
#
#
#dr = 1
#dt = 1
#
#f = plt.figure()
#ax = f.add_subplot(111, polar=True)
#ax.quiver(t, r, dt*getTheta(t, r), dr*getRho(t, r))
#
#axs = plt.subplot(111, polar=True)
#p1 = axs.contourf(t, r, getMagE(t,r), 500)
#cbar = plt.colorbar(p1, ax=axs)

plt.show()
    
main()
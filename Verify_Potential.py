# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 08:37:56 2016

@author: Eric Schmidt
"""

import numpy as np
import matplotlib.pyplot as plt

save_plots = 0

pipeR = 0.063 # (m) Pipe radius
wireR = 0.0024 # (m) Wire radius
d = 0.026
phi = np.pi/4

V0 = 30000

N = 500

rho_array = np.linspace(0,pipeR,N)
theta_array = np.linspace(0,2*np.pi,N)

x = rho_array*np.cos(theta_array)
y = rho_array*np.sin(theta_array)

#rho = d - wireR
#theta = 0
#
#V = (((V0/2)/((np.log((pipeR**2 - d*(d+wireR))/(wireR*d))) - np.log(pipeR/d))) * 
#(np.log((rho**2 + (pipeR**4/d**2) - 2*rho*(pipeR**2/d)*np.cos(theta)) / 
#(rho**2 + d**2 - 2*rho*d*np.cos(theta))) - 
#np.log(pipeR**2/d**2)))
#
#print(V)

V = np.zeros((N,N))
V1 = np.zeros((N,N))
V2 = np.zeros((N,N))
V3 = np.zeros((N,N))
V4 = np.zeros((N,N))

def main():
        
    #### Plotting ####
        
    r, t = np.meshgrid(rho_array, theta_array) 
    
    fig, axs = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
#    p1 = axs.contourf(t, r, V, 100)
    p1 = axs.contourf(t, r, getField(t,r), 100)
    cbar = plt.colorbar(p1, ax=axs)
    axs.set_title("Potential Field (V)")
    
    if save_plots == 1:
        plt.savefig('Images/Potential_Field.png', bbox_inches='tight', dpi=300)
    
    
    plt.show()
    
def getField(theta,rho):
    
    V1 = -(((V0/2)/((np.log((pipeR**2 - d*(d+wireR))/(wireR*d))) - np.log(pipeR/d))) * 
    (np.log((rho**2 + (pipeR**4/d**2) - 2*rho*(pipeR**2/d)*np.cos(theta - 1*np.pi/4)) / 
    (rho**2 + d**2 - 2*rho*d*np.cos(theta - 1*np.pi/4))) - 
    np.log(pipeR**2/d**2)))
    
    V2 = (((V0/2)/((np.log((pipeR**2 - d*(d+wireR))/(wireR*d))) - np.log(pipeR/d))) * 
    (np.log((rho**2 + (pipeR**4/d**2) - 2*rho*(pipeR**2/d)*np.cos(theta - 3*np.pi/4)) / 
    (rho**2 + d**2 - 2*rho*d*np.cos(theta - 3*np.pi/4))) - 
    np.log(pipeR**2/d**2)))
    
    V3 = -(((V0/2)/((np.log((pipeR**2 - d*(d+wireR))/(wireR*d))) - np.log(pipeR/d))) * 
    (np.log((rho**2 + (pipeR**4/d**2) - 2*rho*(pipeR**2/d)*np.cos(theta - 5*np.pi/4)) / 
    (rho**2 + d**2 - 2*rho*d*np.cos(theta - 5*np.pi/4))) - 
    np.log(pipeR**2/d**2)))
    
    V4 = (((V0/2)/((np.log((pipeR**2 - d*(d+wireR))/(wireR*d))) - np.log(pipeR/d))) * 
    (np.log((rho**2 + (pipeR**4/d**2) - 2*rho*(pipeR**2/d)*np.cos(theta - 7*np.pi/4)) / 
    (rho**2 + d**2 - 2*rho*d*np.cos(theta - 7*np.pi/4))) - 
    np.log(pipeR**2/d**2)))
    
    V1[V1 < -V0] = -V0
    V2[V2 > V0] = V0
    V3[V3 < -V0] = -V0
    V4[V4 > V0] = V0

    V = V1 + V2 + V3 + V4
    
    return V
    
main()
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:37:57 2016

@author: schmidte
"""


import numpy as np
import matplotlib.pyplot as plt
import math

pipeR = 0.063 # (m) Pipe radius
wireR = 0.0024 # (m) Wire radius
d = 0.026
phi = np.pi/4

V0 = 30000

N = 100

rho_array = np.linspace(0,pipeR,N)
theta_array = np.linspace(0,2*np.pi,N)

x_array = rho_array*np.cos(theta_array)
y_array = rho_array*np.sin(theta_array)
print(min(x_array))

#x_array = np.linspace(-pipeR,pipeR,N)
#y_array = np.linspace(-pipeR,pipeR,N)

Y, X = np.mgrid[-pipeR:pipeR:15j, -pipeR:pipeR:15j]
U = -1 - np.cos(X**2 + Y)
V = 1 + X - Y
speed = np.sqrt(U**2 + V**2)
UN = U/speed
VN = V/speed

E = np.zeros((N,N))
E_x = np.zeros((N,N))
E_y = np.zeros((N,N))

i = 0

for rho in theta_array:
    
    j = 0
    
    for theta in rho_array:

        x = rho*np.cos(theta)
        y = rho*np.sin(theta)

        E_x[i,j] = (((d-pipeR)*(d+pipeR)*V0*(x**2+y**2) * 
            (-d*(pipeR**2+x**2-y**2)*np.sqrt(x**2+y**2) + d**2*x**2*np.sqrt(1+(y**2/x**2)) + pipeR**2*x**2*np.sqrt(1+(y**2/x**2)))) / 
            (x**3*np.sqrt(1+(y**2/x**2))*(-2*d*np.sqrt(x**2+y**2) + d**2*np.sqrt(1+(y**2/x**2)) + (x**2+y**2)*np.sqrt(1+(y**2/x**2))) * 
            (-2*d*pipeR**2*np.sqrt(x**2+y**2) + pipeR**4*np.sqrt(1+(y**2/x**2)) + d**2*(x**2+y**2)*np.sqrt(1+(y**2/x**2)))*(np.log(pipeR/d) - np.log((pipeR**2 - d*(d+wireR))/(d*wireR))))
            ) 
                
        E_y[i,j] = (((d-pipeR)*(d+pipeR)*V0*y*(-2*d*np.sqrt(x**2+y**2) + d**2*np.sqrt(1+(y**2/x**2)) + pipeR**2*np.sqrt(1+(y**2/x**2)))) / 
            ((x**2*np.sqrt(1+(y**2/x**2))*(-2*d*np.sqrt(x**2+y**2)*np.sqrt(1+(y**2/x**2)))) * 
            (-2*d*pipeR**2*np.sqrt(x**2+y**2)+pipeR**4*np.sqrt(1+(y**2/x**2)) + d**2*(x**2+y**2)*np.sqrt(1+(y**2/x**2))) * 
            (np.log(pipeR/d) - np.log((pipeR**2 - d*(d+wireR))/(d*wireR))))                
            )
            
        if E_x[i,j] > 5500000:
            E_x[i,j] = 5500000
            
        if E_x[i,j] < -5500000:
            E_x[i,j] = -5500000
            
        if E_y[i,j] > 5500000:
            E_y[i,j] = 5500000
            
        if E_y[i,j] < -5500000:
            E_y[i,j] = -5500000
            
        if math.isnan(E_x[i,j]):
            E_x[i,j] = 0
            
        if math.isnan(E_y[i,j]):
            E_y[i,j] = 0
        
        E[i,j] = E_x[i,j] + E_y[i,j]
        
        j = j + 1
    
    i = i + 1
    
#print(E_x[1,:])
    
#### Plotting ####
    
plot1 = plt.figure()
plt.quiver(X, Y, UN, VN,        # data
           U,                   # colour the arrows based on this array
           cmap=plt.cm.seismic,     # colour map
           headlength=7)        # length of the arrows

plt.colorbar()                  # adds the colour bar

plt.title('Quive Plot, Dynamic Colours')
plt.show(plot1)                 # display the plot
    
#X, Y = np.meshgrid(x_array, y_array) 
#
#fig, axs = plt.subplots(1, 1)
#p1 = axs.contourf(X, Y, E, 100)
#cbar = plt.colorbar(p1, ax=axs)
#
#plt.show()
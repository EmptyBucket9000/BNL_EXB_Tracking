# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:17:19 2016

@author: Eric Schmidt
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

#E_0 = 12.6*10**3 # (V/m) Electric field in radial direction (2.6cm)
E_0 = 13.54*10**3 # (V/m) Electric field in radial direction (2.2cm)
#E_0 = 10.48*10**3 # (V/m) Electric field in radial direction (4.2cm)

#V = (((V0/2)/((np.log((R0**2 - d*(d+Ri))/(Ri*d))) - np.log(R0/d))) * 
#(np.log((rho**2 + (R0**4/d**2) - 2*rho*(R0**2/d)*np.cos(theta)) / 
#(rho**2 + d**2 - 2*rho*d*np.cos(theta))) - 
#np.log(R0**2/d**2)))

R0 = 0.063 # (m) Pipe radius
Ri = 0.0024 # (m) Wire radius
d = 0.026
V0 = 3*10**4 # (V) Potential at wire surface

q = -1.60217662*10**(-19) # (C) Electron charge
m = 9.10938356*10**(-31) # (kg) Electron mass

def main():
    
    createPlots = 0 # 0 for No, 1 for Yes
    savePlots = 0
    saveOutput = 0
    
    steps = 10**8
    dt = 10**(-15)
#    t = np.linspace(0,steps*dt,num=steps)
    
    BfieldPowerDrop = -2.25
    
    text = "No contact"
    time = steps*dt
    
#    B_desired_array = np.arange(0.0300,0.0400,0.0001)
    B_desired_array = np.array([0.025])
    
#    theta_array = np.arange(0,7,1)*np.pi/16
#    R_array = np.arange(1,4,1)*(1/3)*R0
    
    theta_array = np.array([24*np.pi/32])
    R_array = np.array([R0-0.00001])
    
    out = np.array([["Contact","Theta (deg)","R (m)","Distance Traveled (m)","B_i (T)","B_f (T)","Delta B (T)","Time (ns)"]])    
    
    for theta in theta_array:
        
        for R in R_array:

#            V = (((V0/2)/((np.log((R0**2 - d*(d+Ri))/(Ri*d))) - np.log(R0/d))) * 
#            (np.log((R**2 + (R0**4/d**2) - 2*R*(R0**2/d)*np.cos(theta)) / 
#            (R**2 + d**2 - 2*R*d*np.cos(theta))) - 
#            np.log(R0**2/d**2)))
    
            for B_desired in B_desired_array:
                
                x = np.zeros((steps,3))
                v = np.zeros((steps,3))
                E = np.zeros((steps,3))
                B = np.zeros((steps,3))
                
                
                x_0 = R*np.cos(theta)
                y_0 = R*np.sin(theta)
                z_0 = np.exp((np.log(B_desired)/BfieldPowerDrop))
                v_x0 = 0
                v_y0 = 0
                v_z0 = 0
            
                x[0,0] = x_0
                x[0,1] = y_0
                x[0,2] = z_0
                v[0,0] = v_x0
                v[0,1] = v_y0
                v[0,2] = v_z0
            
                E[0,0] = (E_0/(np.sqrt(x[0,0]**2 + x[0,1]**2))) * np.cos(np.arctan(x[0,1]/x[0,0]))
                if x_0 > 0 and y_0 > 0:
                    E[0,0] = np.abs(E[0,0])
            
#                E[0,0] = (((d-R0)*(d+R0)*V0*(x[0,0]**2+x[0,1]**2) * 
#                (-d*(R0**2+x[0,0]**2-x[0,1]**2)*np.sqrt(x[0,0]**2+x[0,1]**2) + d**2*x[0,0]**2*np.sqrt(1+(x[0,1]**2/x[0,0]**2)) + R0**2*x[0,0]**2*np.sqrt(1+(x[0,1]**2/x[0,0]**2)))) / 
#                (x[0,0]**3*np.sqrt(1+(x[0,1]**2/x[0,0]**2))*(-2*d*np.sqrt(x[0,0]**2+x[0,1]**2) + d**2*np.sqrt(1+(x[0,1]**2/x[0,0]**2)) + (x[0,0]**2+x[0,1]**2)*np.sqrt(1+(x[0,1]**2/x[0,0]**2))) * 
#                (-2*d*R0**2*np.sqrt(x[0,0]**2+x[0,1]**2) + R0**4*np.sqrt(1+(x[0,1]**2/x[0,0]**2)) + d**2*(x[0,0]**2+x[0,1]**2)*np.sqrt(1+(x[0,1]**2/x[0,0]**2)))*(np.log(R0/d) - np.log((R0**2 - d*(d+Ri))/(d*Ri))))
#                )
                
#                if x_0 > 0 and y_0 > 0:
#                    
#                    if x_0 > d:
#                        E[0,0] = -np.abs(E[0,0])
#                        
#                    if x_0 < d:
#                        E[0,0] = np.abs(E[0,0])
#                    
#                if x_0 > 0 and y_0 < 0:
#                    
#                    if x_0 > d:
#                        E[0,0] = np.abs(E[0,0])
#                        
#                    if x_0 < d:
#                        E[0,0] = -np.abs(E[0,0])
#                    
#                if x_0 < 0 and y_0 > 0:
#                    
#                    if x_0 < -d:
#                        E[0,0] = -np.abs(E[0,0])
#                        
#                    if x_0 > -d:
#                        E[0,0] = np.abs(E[0,0])
#                    
#                if x_0 < 0 and y_0 < 0:
#                    
#                    if x_0 < -d:
#                        E[0,0] = np.abs(E[0,0])
#                        
#                    if x_0 > -d:
#                        E[0,0] = -np.abs(E[0,0])
                    
                    
                E[0,1] = (E_0/(np.sqrt(x[0,0]**2 + x[0,1]**2))) * np.sin(np.arctan(x[0,1]/x[0,0]))
                
#                E[0,1] = (((d-R0)*(d+R0)*V0*x[0,1]*(-2*d*np.sqrt(x[0,0]**2+x[0,1]**2) + d**2*np.sqrt(1+(x[0,1]**2/x[0,0]**2)) + R0**2*np.sqrt(1+(x[0,1]**2/x[0,0]**2)))) / 
#                ((x[0,0]**2*np.sqrt(1+(x[0,1]**2/x[0,0]**2))*(-2*d*np.sqrt(x[0,0]**2+x[0,1]**2)*np.sqrt(1+(x[0,1]**2/x[0,0]**2)))) * 
#                (-2*d*R0**2*np.sqrt(x[0,0]**2+x[0,1]**2)+R0**4*np.sqrt(1+(x[0,1]**2/x[0,0]**2)) + d**2*(x[0,0]**2+x[0,1]**2)*np.sqrt(1+(x[0,1]**2/x[0,0]**2))) * 
#                (np.log(R0/d) - np.log((R0**2 - d*(d+Ri))/(d*Ri))))                
#                )
                
#                if x_0 > 0 and y_0 > 0:
#                    
#                    if y_0 > d:
#                        E[0,1] = -np.abs(E[0,1])
#                        
#                    if y_0 < d:
#                        E[0,1] = np.abs(E[0,1])
#                    
#                if x_0 > 0 and y_0 < 0:
#                    
#                    if y_0 > -d:
#                        E[0,1] = np.abs(E[0,1])
#                        
#                    if y_0 < -d:
#                        E[0,1] = -np.abs(E[0,1])
#                    
#                if x_0 < 0 and y_0 > 0:
#                    
#                    if y_0 < d:
#                        E[0,1] = np.abs(E[0,1])
#                        
#                    if y_0 > d:
#                        E[0,1] = -np.abs(E[0,1])
#                    
#                if x_0 < 0 and y_0 < 0:
#                    
#                    if y_0 < -d:
#                        E[0,1] = np.abs(E[0,1])
#                        
#                    if y_0 > -d:
#                        E[0,1] = -np.abs(E[0,1])
                    
                B[0] = getBField(x[0],BfieldPowerDrop)
                
                i = 0
                
                while i < steps-1:
                    
                    x[i+1], v[i+1], E[i+1], B[i+1] = updatePostion(x[i],v[i],dt,BfieldPowerDrop)
                    
                    r = np.sqrt(x[i,0]**2 + x[i,1]**2)
                    
                    if r > R0:
                    
                        time = i*dt
                        text = "Escape"
                        break
                    
                    nd = (np.sqrt(2)/2)*d                    
                    
                    dx1 = np.abs(x[i,0]) - nd
                    dy1 = np.abs(x[i,1]) - nd
                    dd = np.sqrt(dx1**2 + dy1**2)
                    
                    if dd < Ri:
                    
                        time = i*dt
                        text = "Contact"            
                        break
                    
                    i = i + 1
                
                B = B[1:i-1:1]
                E = E[1:i-1:1]
                x = x[1:i-1:1]
                v = v[1:i-1:1]
                
                if createPlots == 1:    
                
                    n = 1
                    
                    props = dict(boxstyle='square', facecolor='wheat', alpha=0.8)
                    
                    plt.figure(n)
                    n = n + 1
                    plt.plot(v[:,0],v[:,2])  
                    plt.xlabel('x-velocity (m/s)')
                    plt.ylabel('z-velocity (m/s)')  
                #    
                    plt.figure(n)
                    n = n + 1
                    ax = plt.subplot(1,1,1)
                    ax.plot(x[:,0],x[:,1])
                    plt.xlabel('x-position (m)')
                    plt.ylabel('y-position (m)')
                    textstr = 'B-field: %0.3f (T)\nStarting y: %0.3f (m)\nStarting x: %0.3f (m)'%(B_desired,y_0,x_0)
                    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                            fontsize=12, verticalalignment='top', bbox=props)
                    if savePlots == 1:
                        plt.savefig('Images/B-%0.3f_y-pos_vs_x-pos.png'%B_desired, bbox_inches='tight', dpi=300)
                    
                    plt.figure(n)
                    n = n + 1
                    ax = plt.subplot(1,1,1)
                    ax.plot(x[:,0],x[:,2])
                    plt.xlabel('x-position (m)')
                    plt.ylabel('z-position (m)')
                    textstr = 'B-field: %0.3f (T)\nStarting y: %0.3f (m)\nStarting x: %0.3f (m)'%(B_desired,y_0,x_0)
                    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                            fontsize=12, verticalalignment='top', bbox=props)
                    if savePlots == 1:
                        plt.savefig('Images/B-%0.3f_z-pos_vs_x-pos.png'%B_desired, bbox_inches='tight', dpi=300)
                    
                    plt.figure(n)
                    n = n + 1
                    plt.plot(x[:,2],B[:,1])
                    plt.xlabel('z-position (m)')
                    plt.ylabel('B-field (T)')
                    
                    plt.figure(n)
                    n = n + 1
                    xn = np.sqrt(x[:,0]**2+x[:,1]**2)
                    En = np.sqrt(E[:,0]**2+E[:,1]**2+E[:,2]**2)
                    plt.plot(xn,En)
                    plt.xlabel('Distance from wire center (m)')
                    plt.ylabel('E (V/m)')
                    
                    plt.show()
                
                out = np.append(out, [['%s'%text,theta*180/np.pi,R,(x[i-3,2] - z_0),B_desired,B[i-3,1],(B[i-3,1]-B_desired),time*10**9]], axis=0)
        
        print('\n%s'%text)
        print('Distance traveled: %0.5e'%(x[i-3,2] - z_0))
        print('Starting B: %0.5e'%B_desired)
        print('Ending B: %0.5e'%B[i-3,1])
        print('B difference: %0.5e'%(B[i-3,1]-B_desired))
        print('Time passed: %0.5e'%time)
        
    if saveOutput == 1:
        path = "output.csv"
        with open(path, "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for row in out:
                writer.writerow(row)
    
def updatePostion(x,v,dt,BfieldPowerDrop):

    E = getEField(x)    
    B = getBField(x,BfieldPowerDrop)
    
    F = q*(E + np.cross(v,B))
    a = F/m
    v_new = v + a*dt
    x_new = x + v_new*dt
    
    return x_new, v_new, E, B
    
def getEField(x):
    
#    E_x = (E_0/(np.sqrt(x[0]**2 + x[1]**2))) * np.cos(np.arctan(x[1]/x[0]))
    
    E_x = (((d-R0)*(d+R0)*V0*(x[0]**2+x[1]**2) * 
                (-d*(R0**2+x[0]**2-x[1]**2)*np.sqrt(x[0]**2+x[1]**2) + d**2*x[0]**2*np.sqrt(1+(x[1]**2/x[0]**2)) + R0**2*x[0]**2*np.sqrt(1+(x[1]**2/x[0]**2)))) / 
                (x[0]**3*np.sqrt(1+(x[1]**2/x[0]**2))*(-2*d*np.sqrt(x[0]**2+x[1]**2) + d**2*np.sqrt(1+(x[1]**2/x[0]**2)) + (x[0]**2+x[1]**2)*np.sqrt(1+(x[1]**2/x[0]**2))) * 
                (-2*d*R0**2*np.sqrt(x[0]**2+x[1]**2) + R0**4*np.sqrt(1+(x[1]**2/x[0]**2)) + d**2*(x[0]**2+x[1]**2)*np.sqrt(1+(x[1]**2/x[0]**2)))*(np.log(R0/d) - np.log((R0**2 - d*(d+Ri))/(d*Ri))))
                ) 
                
#    if x[0] > 0 and x[1] > 0:
#        
#        if x[0] > d:
#            E_x = -np.abs(E_x)
#            
#        if x[0] < d:
#            E_x = np.abs(E_x)
#        
#    if x[0] > 0 and x[1] < 0:
#        
#        if x[0] > d:
#            E_x = np.abs(E_x)
#            
#        if x[0] < d:
#            E_x = -np.abs(E_x)
#        
#    if x[0] < 0 and x[1] > 0:
#        
#        if x[0] < -d:
#            E_x = -np.abs(E_x)
#            
#        if x[0] > -d:
#            E_x = np.abs(E_x)
#        
#    if x[0] < 0 and x[1] < 0:
#        
#        if x[0] < -d:
#            E_x = np.abs(E_x)
#            
#        if x[0] > -d:
#            E_x = -np.abs(E_x)
        
#    E_y = (E_0/(np.sqrt(x[0]**2 + x[1]**2))) * np.sin(np.arctan(x[1]/x[0]))
        
    E_y = (((d-R0)*(d+R0)*V0*x[1]*(-2*d*np.sqrt(x[0]**2+x[1]**2) + d**2*np.sqrt(1+(x[1]**2/x[0]**2)) + R0**2*np.sqrt(1+(x[1]**2/x[0]**2)))) / 
                ((x[0]**2*np.sqrt(1+(x[1]**2/x[0]**2))*(-2*d*np.sqrt(x[0]**2+x[1]**2)*np.sqrt(1+(x[1]**2/x[0]**2)))) * 
                (-2*d*R0**2*np.sqrt(x[0]**2+x[1]**2)+R0**4*np.sqrt(1+(x[1]**2/x[0]**2)) + d**2*(x[0]**2+x[1]**2)*np.sqrt(1+(x[1]**2/x[0]**2))) * 
                (np.log(R0/d) - np.log((R0**2 - d*(d+Ri))/(d*Ri))))                
                )
                
#    if x[0] > 0 and x[1] > 0:
#        
#        if x[1] > d:
#            E_y = -np.abs(E_y)
#            
#        if x[1] < d:
#            E_y = np.abs(E_y)
#        
#    if x[0] > 0 and x[1] < 0:
#        
#        if x[1] > -d:
#            E_y = np.abs(E_y)
#            
#        if x[1] < -d:
#            E_y = -np.abs(E_y)
#        
#    if x[0] < 0 and x[1] > 0:
#        
#        if x[1] < d:
#            E_y = np.abs(E_y)
#            
#        if x[1] > d:
#            E_y = -np.abs(E_y)
#        
#    if x[0] < 0 and x[1] < 0:
#        
#        if x[1] < -d:
#            E_y = np.abs(E_y)
#            
#        if x[1] > -d:
#            E_y = -np.abs(E_y)
    
    E = np.array([E_x,E_y,0])
    
    return E
    
def getBField(x,BfieldPowerDrop):
    
    B = np.array([0,x[2]**BfieldPowerDrop,0])
    
    return B
    
main()
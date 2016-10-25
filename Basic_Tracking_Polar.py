# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:17:19 2016

@author: Eric Schmidt
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

#E_0 = 12.6*10**3 # (V/m) Electric field in radial direction (2.6cm)
E_0 = 13.42*10**3 # (V/m) Electric field in radial direction (2.2cm)
#E_0 = 10.48*10**3 # (V/m) Electric field in radial direction (4.2cm)

c = 2.99792458*10**8 # (m/s) Speed of light

#V = (((V0/2)/((np.log((R0**2 - d*(d+Ri))/(Ri*d))) - np.log(R0/d))) * 
#(np.log((rho**2 + (R0**4/d**2) - 2*rho*(R0**2/d)*np.cos(theta)) / 
#(rho**2 + d**2 - 2*rho*d*np.cos(theta))) - 
#np.log(R0**2/d**2)))

part_type = "e"
createPlots = 0 # 0 for No, 1 for Yes
savePlots = 0
saveOutput = 1
batman_test = 1

R0 = 0.063 # (m) Pipe radius
Ri = 0.0024 # (m) Wire radius
d = 0.03578
V0 = 2.96*10**4 # (V) Potential at wire surface

if part_type == "p":
    q = 1.60217662*10**(-19) # (C) Proton charge
    m = 1.6726219*10**(-27) # (kg) Proton mass
    dt = 10**(-13)
    
if part_type == "e":
    q = -1.60217662*10**(-19) # (C) Electron charge
    m = 9.10938356*10**(-31) # (kg) Electron mass
    dt = 10**(-14)

# B-field fit parameters
fit_a = 0.036
fit_b = -0.017

def main():
    
    steps = 10**7
#    t = np.linspace(0,steps*dt,num=steps)
    
    BfieldPowerDrop = -2.25
    
    text = "No contact"
    time = steps*dt
    
    B_batman = 0.5 # (T) B-field at batman
    
    B_desired_array = np.arange(0.595,0.607,0.002)
#    B_desired_array = np.array([0.300])
    
    theta_array = np.arange(0,8,1)*np.pi/16
    R_array = np.arange(1,7,1)*(1/6)*R0-0.001
    
#    theta_array = np.array([12*np.pi/16])
#    R_array = np.array([R0*(3/6)-0.003])
    
    if batman_test == 1:
        out = np.array([["Contact","Theta (deg)","R (m)","Distance Traveled (m)","B_i (T)","B_f (T)","Delta B (T)","Time (ns)","Max Gamma","Batman Theta (deg)","Batman R (m)","Batman Gamma"]])    
        
    if batman_test == 0:
        out = np.array([["Contact","Theta (deg)","R (m)","Distance Traveled (m)","B_i (T)","B_f (T)","Delta B (T)","Time (ns)","Max Gamma"]])    
    
    for theta in theta_array:
        
        for R in R_array:
    
            for B_desired in B_desired_array:
                
                print('Particle: %s'%part_type)
                print('Starting B: %0.3f'%B_desired)
                print('Time step: %e'%dt)
                
                gamma = np.zeros(steps)
                
                x = np.zeros((steps,3))
                v = np.zeros((steps,3))
                E = np.zeros((steps,3))
                B = np.zeros((steps,3))
                
                x_0 = R*np.cos(theta)
                y_0 = R*np.sin(theta)
                z_0 = np.exp((np.log((B_desired-fit_b)/fit_a)/BfieldPowerDrop))
                v_x0 = 0
                v_y0 = 0
                v_z0 = 0
#                v_z0 = 2.4*10**6
#                v_z0 = 4*10**7
            
                x[0,0] = x_0
                x[0,1] = y_0
                x[0,2] = z_0
                v[0,0] = v_x0
                v[0,1] = v_y0
                v[0,2] = v_z0
            
                E[0] = getEField(x[0])
                B[0] = getBField(x[0],BfieldPowerDrop)
                
                i = 0
                
                while i < steps-1:
                    
                    x[i+1], v[i+1], E[i+1], B[i+1], gamma[i] = updatePostion(x[i],v[i],dt,BfieldPowerDrop,gamma)
                    
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
                    
                    if x[i,2] - z_0 < -0.05:
                        
                        time = i*dt
                        text = "Moving Upstream"
                        break
                    
                    if batman_test == 1:
                        if B[i+1,1] < B_batman:
                            
                            time = i*dt
                            text = "Batman Position"
                            break
                    
                    i = i + 1
                
                B = B[1:i-1:1]
                E = E[1:i-1:1]
                x = x[1:i-1:1]
                v = v[1:i-1:1]
#                print(max(gamma))
                
                if createPlots == 1:
                    plotAll(B_desired, x, v, B, E, savePlots)
                    
                max_gamma = max(gamma)
                
                if batman_test == 1:
                    out = np.append(out, [['%s'%text,theta*180/np.pi,R,(x[i-3,2] - z_0),B_desired,B[i-3,1],(B[i-3,1]-B_desired),time*10**9,max_gamma,np.arctan2(x[i-3,1],x[i-3,0])*180/np.pi,np.sqrt(x[i-3,0]**2 + x[i-3,1]**2),gamma[i-3]]], axis=0)
                if batman_test == 0:
                    out = np.append(out, [['%s'%text,theta*180/np.pi,R,(x[i-3,2] - z_0),B_desired,B[i-3,1],(B[i-3,1]-B_desired),time*10**9,max_gamma]], axis=0)
        
                if saveOutput == 1:
                    if batman_test == 1:
                        if v_z0 > 0:
                            path = "Output/Batman_Test/zvel/%s/%s_Out_B_%0.3f_R-%0.3f_theta-%0.3f.csv"%(part_type,part_type,B_desired,R,theta)
                        if v_z0 == 0:
                            path = "Output/Batman_Test/%s/%s_Out_B_%0.3f_R-%0.3f_theta-%0.3f.csv"%(part_type,part_type,B_desired,R,theta)
                    if batman_test !=1:
                        if v_z0 > 0:
                            path = "Output/zvel/%s/%s_Out_B_%0.3f_R-%0.3f_theta-%0.3f.csv"%(part_type,part_type,B_desired,R,theta)
                        if v_z0 == 0:
                            path = "Output/%s/%s_Out_B_%0.3f_R-%0.3f_theta-%0.3f.csv"%(part_type,part_type,B_desired,R,theta)
                    with open(path, "w", newline='') as csv_file:
                        writer = csv.writer(csv_file, delimiter=',')
                        for row in out:
                            writer.writerow(row)
        
                print('\n%s'%text)
                print('Distance traveled: %0.5e'%(x[i-3,2] - z_0))
                print('Starting B: %0.5e'%B_desired)
                print('Ending B: %0.5e'%B[i-3,1])
                print('B difference: %0.5e'%(B[i-3,1]-B_desired))
                print('Time passed: %0.5e'%time)
                
                print(E)
        
#    if saveOutput == 1:
#        path = "output-t.csv"
#        with open(path, "w", newline='') as csv_file:
#            writer = csv.writer(csv_file, delimiter=',')
#            for row in out:
#                writer.writerow(row)

def updatePostion(x,v,dt,BfieldPowerDrop,gamma):

    E = getEField(x)    
    B = getBField(x,BfieldPowerDrop)
    
    F = q*(E + np.cross(v,B))
    beta = np.sqrt(np.dot(v,v))/c
    gamma = 1/np.sqrt(1-beta**2)
    a = F/(m*gamma)
    v_new = v + a*dt
    x_new = x + v_new*dt
    
    return x_new, v_new, E, B, gamma
    
def getEField(x):
    
    theta = np.arctan2(x[1],x[0])
    rho = np.sqrt(x[0]**2 + x[1]**2)
    
    E_theta = getThetaField(theta, rho)
    E_rho = getRhoField(theta, rho)
    
    E_x = E_rho*np.cos(theta) + E_theta*np.sin(theta)
    E_y = E_rho*np.sin(theta) - E_theta*np.cos(theta)
    
    E = np.array([E_x,E_y,0])
    
    return E
    
def getThetaField(theta,rho):

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
#    E_theta = 0
    
    return E_theta
    
def getRhoField(theta, rho):
    
    E_r1 = (((-d**4+R0**4)*rho*V0 + d*(d-R0)*(d+R0)*(R0**2+rho**2)*V0*np.cos(theta-1*np.pi/4)) / 
    ((d**2+rho**2-2*d*rho*np.cos(theta-1*np.pi/4))*(R0**4+d**2*rho**2-2*d*R0**2*rho*np.cos(theta-1*np.pi/4))*(np.log(R0/d) - 
    np.log((R0**2-d*(d+Ri))/(d*Ri))))
    )
    
    E_r2 = -(((-d**4+R0**4)*rho*V0 + d*(d-R0)*(d+R0)*(R0**2+rho**2)*V0*np.cos(theta-3*np.pi/4)) / 
    ((d**2+rho**2-2*d*rho*np.cos(theta-3*np.pi/4))*(R0**4+d**2*rho**2-2*d*R0**2*rho*np.cos(theta-3*np.pi/4))*(np.log(R0/d) - 
    np.log((R0**2-d*(d+Ri))/(d*Ri))))
    )
    
    E_r3 = (((-d**4+R0**4)*rho*V0 + d*(d-R0)*(d+R0)*(R0**2+rho**2)*V0*np.cos(theta-5*np.pi/4)) / 
    ((d**2+rho**2-2*d*rho*np.cos(theta-5*np.pi/4))*(R0**4+d**2*rho**2-2*d*R0**2*rho*np.cos(theta-5*np.pi/4))*(np.log(R0/d) - 
    np.log((R0**2-d*(d+Ri))/(d*Ri))))
    )
    
    E_r4 = -(((-d**4+R0**4)*rho*V0 + d*(d-R0)*(d+R0)*(R0**2+rho**2)*V0*np.cos(theta-7*np.pi/4)) / 
    ((d**2+rho**2-2*d*rho*np.cos(theta-7*np.pi/4))*(R0**4+d**2*rho**2-2*d*R0**2*rho*np.cos(theta-7*np.pi/4))*(np.log(R0/d) - 
    np.log((R0**2-d*(d+Ri))/(d*Ri))))
    )
    
    E_r = E_r1 + E_r2 + E_r3 + E_r4
#    E_r = 0
    
    return E_r
    
def getBField(x,BfieldPowerDrop):
    
#    B = np.array([0,x[2]**BfieldPowerDrop,0])
    B = np.array([0,fit_a*x[2]**BfieldPowerDrop + fit_b,0])
    
    return B
    
def plotAll(B_desired, x, v, B, E, savePlots):
                
    n = 1
    
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.8)
    
    plt.figure(n)
    n = n + 1
    ax = plt.subplot(1,1,1)
    ax.plot(v[:,0],v[:,2])  
    plt.xlabel('x-velocity (m/s)')
    plt.ylabel('z-velocity (m/s)')  
    textstr = 'B-field: %0.3f (T)\nStarting y: %0.3f (m)\nStarting x: %0.3f (m)'%(B_desired,x[0,1],x[0,0])
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)
    if savePlots == 1:
        plt.savefig('Output/Saved_Plots/B-%0.3f_y-vel-vs-x-vel_x-%0.3f_y-%0.3f.png'%(B_desired,x[0,0],x[0,1]), bbox_inches='tight', dpi=300)
    
    plt.figure(n)
    n = n + 1
    ax = plt.subplot(1,1,1)
    ax.plot(x[:,0],x[:,1])
    plt.xlabel('x-position (m)')
    plt.ylabel('y-position (m)')
    textstr = 'B-field: %0.3f (T)\nStarting y: %0.3f (m)\nStarting x: %0.3f (m)'%(B_desired,x[0,1],x[0,0])
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)
    if savePlots == 1:
        plt.savefig('Output/Saved_Plots/B-%0.3f_y-vs-x_x-%0.3f_y-%0.3f.png'%(B_desired,x[0,0],x[0,1]), bbox_inches='tight', dpi=300)
    
    plt.figure(n)
    n = n + 1
    ax = plt.subplot(1,1,1)
    ax.plot(x[:,0],x[:,2])
    plt.xlabel('x-position (m)')
    plt.ylabel('z-position (m)')
    textstr = 'B-field: %0.3f (T)\nStarting z: %0.3f (m)\nStarting x: %0.3f (m)'%(B_desired,x[0,2],x[0,0])
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)
    if savePlots == 1:
        plt.savefig('Output/Saved_Plots/B-%0.3f_z-vs-x-x_%0.3f_z-%0.3f.png'%(B_desired,x[0,0],x[0,2]), bbox_inches='tight', dpi=300)
    
    plt.figure(n)
    n = n + 1
    plt.plot(x[:,2],B[:,1])
    plt.xlabel('z-position (m)')
    plt.ylabel('B-field (T)')
    
    plt.figure(n)
    n = n + 1
    plt.plot(np.sqrt(x[:,0]**2 + x[:,1]**2),v[:,2])
    plt.xlabel('Radial position (m)')
    plt.ylabel('z-velocity (m/s)')
    
#    plt.figure(n)
#    n = n + 1
#    xn = np.sqrt(x[:,0]**2+x[:,1]**2)
#    En = np.sqrt(E[:,0]**2+E[:,1]**2+E[:,2]**2)
#    plt.plot(xn,En)
#    plt.xlabel('Distance from pipe center (m)')
#    plt.ylabel('E (V/m)')
    
    plt.show()
    
main()
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:11:20 2016

@author: Eric Schmidt
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
import os

os.chdir(os.path.dirname(__file__))

save_plots = 1
batman_test = 1
B_i = 0.550
B_i2 = 0.650
R_0 = 0.063

def main():
    
    part_type = "e"
    file_name = "Batman_Test/e/e_Out_B_0.650_R-0.062_theta-6.087"
    files = glob.glob("%s/Output/%s.csv"%(os.getcwd(),file_name))
    
    if batman_test == 1:
        columns = np.arange(1,12,1)
        
    if batman_test == 0:
        columns = np.arange(1,9,1)
        
    contact_i = 0
    no_contact_i = 0
    escape_i = 0
    moving_upstream_i = 0
    batman_position_i = 0
    M = len(columns)
    i = 0

    for file in files:
        
        with open(file, "rt") as inf:
            reader = csv.reader(inf, delimiter=',')
            next(reader, None)  # skip the headers
    
            stuff = list(reader)
            N = len(stuff)
            contact = np.zeros((N,M))
            no_contact = np.zeros((N,M))
            escape = np.zeros((N,M))
            moving_upstream = np.zeros((N,M))
            batman_position = np.zeros((N,M))
            
#            if batman_test == 1:
#                data = np.zeros((N,6))
                
#            if batman_test == 0:
            data = np.zeros((N,3))
            
            for row in stuff:
                data[i,0] = row[1]
                data[i,1] = row[2]
                data[i,2] = row[5]
                
                # Currently not used
#                if batman_test == 1:
#                    data[i,3] = row[9]
#                    data[i,4] = row[10]
#                    data[i,5] = row[11]
                    
                i = i + 1
                j = 0
                if row[0] == "Contact":
                    for column in columns:
                        contact[contact_i,j] = row[column]
                        j = j + 1
                    contact_i = contact_i + 1
                if row[0] == "No Contact":
                    for column in columns:
                        no_contact[no_contact_i,j] = row[column]
                        j = j + 1
                    no_contact_i = no_contact_i + 1
                if row[0] == "Escape":
                    for column in columns:
                        escape[escape_i,j] = row[column]
                        j = j + 1
                    escape_i = escape_i + 1
                if row[0] == "Moving Upstream":
                    for column in columns:
                        moving_upstream[moving_upstream_i,j] = row[column]
                        j = j + 1
                    moving_upstream_i = moving_upstream_i + 1
                if row[0] == "Batman Position":
                    for column in columns:
                        batman_position[batman_position_i,j] = row[column]
                        j = j + 1
                    batman_position_i = batman_position_i + 1
       
    ## Remove rows of all zeros
           
    contact = contact[~np.all(contact == 0, axis=1)]
    no_contact = no_contact[~np.all(no_contact == 0, axis=1)]
    escape = escape[~np.all(escape == 0, axis=1)]
    moving_upstream = moving_upstream[~np.all(moving_upstream == 0, axis=1)]
    batman_position = batman_position[~np.all(batman_position == 0, axis=1)]
    
    ## Plotting
    
    n = 0
    
    # Lead positions
    a_p=[[3*np.pi/4,0.03578],[7*np.pi/4,0.03578]]
    a_n=[[np.pi/4,0.03578],[5*np.pi/4,0.03578]]
    
    # Plots color-coded points based on final outcome of the particle
    plt.figure(n)
    n = n + 1
    
    ax = plt.subplot(111, projection='polar')
    ax.scatter(contact[:,0]*np.pi/180, contact[:,1], color='g', label='Contact')
#    ax.scatter(no_contact[:,0]*np.pi/180, no_contact[:,1], color='purple', label='No Contact')
    ax.scatter(escape[:,0]*np.pi/180, escape[:,1], color='b', label = 'Escape')
    ax.scatter(moving_upstream[:,0]*np.pi/180, moving_upstream[:,1], color='k', label = 'Moving Upstream')
    if batman_test == 1:
        ax.scatter(batman_position[:,0]*np.pi/180, batman_position[:,1], color='y', label = 'Batman Position')
    ax.plot(*zip(*a_p), marker='o', color='r', ls='', ms = 8, label = 'Positive Lead')
    ax.plot(*zip(*a_n), marker='x', color='r', ls='', mew=3, ms=8, label = 'Negative Lead')
    ax.set_rmax(R_0)
    ax.grid(True)
    lgd = ax.legend(bbox_to_anchor=(1.73,1.11))
    ax.set_title("Starting B-Field: %0.2f T"%B_i, va='bottom')
    
    if save_plots == 1:
        
        if batman_test == 1:
            plt.savefig('Output/Saved_Plots/%s_Batman_Out_B_%0.3f.png'%(part_type,B_i), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
            
        if batman_test == 0:
            plt.savefig('Output/Saved_Plots/%s_Out_B_%0.3f.png'%(part_type,B_i), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    
    if batman_test == 1:
        
        plt.figure(n)
        n = n + 1
        axbat = plt.subplot(111, projection='polar')
        cm = plt.cm.get_cmap('RdYlBu')
        sc = axbat.scatter(batman_position[:,8]*np.pi/180, batman_position[:,9], c = batman_position[:,10], vmin=1.000, vmax=max(batman_position[:,10]), s=50, cmap=cm)
        axbat.plot(*zip(*a_p), marker='o', color='r', ls='', ms = 8, label = 'Positive Lead')
        axbat.plot(*zip(*a_n), marker='x', color='r', ls='', mew=3, ms=8, label = 'Negative Lead')
        axbat.set_rmax(R_0)
        axbat.set_title('Particle Positions before Batman (Beginning %0.3f-%0.3f T)'%(B_i,B_i2))
        plt.colorbar(sc)
        txtb = axbat.text(10.1*np.pi/8, 0.11, 'Colorbar gives relativistic gamma', fontsize=15)   
        
        if save_plots == 1:
            plt.savefig('Output/Saved_Plots/%s_Batman_Out_B_Values_%0.3f-%0.3f.png'%(part_type,B_i,B_i2), bbox_extra_artists=(lgd,txtb), bbox_inches='tight', dpi=300)     
        
    if batman_test == 0:
        
        # Plots color-coded points based on the final B-field attained by the particle        
        
        plt.figure(n)
        n = n + 1
        ax2 = plt.subplot(111, projection='polar')
        cm = plt.cm.get_cmap('RdYlBu')
        sc = ax2.scatter(data[:,0]*np.pi/180, data[:,1], c = data[:,2], vmin=min(data[:,2]), vmax=max(data[:,2]), s=(100*data[:,1])**2*10, cmap=cm)
        ax2.plot(*zip(*a_p), marker='o', color='r', ls='', ms = 8, label = 'Positive Lead')
        ax2.plot(*zip(*a_n), marker='x', color='r', ls='', mew=3, ms=8, label = 'Negative Lead')
        ax2.set_rmax(R_0)
        ax2.set_title("Final Magnetic Field Values (T)")
        plt.colorbar(sc)
        txtb = ax2.text(10.5*np.pi/8, 0.095, 'Minimum B-Field: %0.3f'%min(data[:,2]), fontsize=15)
        
        if save_plots == 1:
            plt.savefig('Output/Saved_Plots/%s_Out_B_Values_%0.3f.png'%(part_type,B_i), bbox_extra_artists=(lgd,txtb), bbox_inches='tight', dpi=300)
            
        # Plots line graph of final B-field attained as a function of theta.
        # Each value of rho gets graphed
            
        plt.figure(n)
        n = n + 1
        
        rho_set = set(data[:,1])
        unique_rho = list(rho_set)
        unique_rho.sort()
        theta_set = set(data[:,0])
        unique_theta = list(theta_set)
        unique_theta = [float(i) for i in unique_theta]
        unique_theta.sort()
        new_data = np.zeros((len(theta_set),len(rho_set)))
    
        j = 0
        for rho in unique_rho:
            
            i = 0
            k = 0
            while i < N:
                if data[i,1] == rho:
                    new_data[k,j] = data[i,2]
                    k = k + 1
                i = i + 1
            j = j + 1
        
    #    new_data = new_data[new_data[:,0].argsort()]
        axlin = plt.subplot(111)
        axlin.plot(unique_theta,new_data[:,0], label='1/6 Radius')
        axlin.plot(unique_theta,new_data[:,1], label='2/6 Radius')
        axlin.plot(unique_theta,new_data[:,2], label='3/6 Radius')
        axlin.plot(unique_theta,new_data[:,3], label='4/6 Radius')
        axlin.plot(unique_theta,new_data[:,4], label='5/6 Radius')
        axlin.plot(unique_theta,new_data[:,5], label='6/6 Radius')
        lgd = axlin.legend(bbox_to_anchor=(1.36,1.025))
        axlin.set_title("B-Field Values")
        axlin.set_xlabel('Theta (deg)')
        axlin.set_ylabel('B-Field (T)')
        
        if save_plots == 1:
            plt.savefig('Output/Saved_Plots/%s_Out_B_Values_by_Radius_%0.3f.png'%(part_type,B_i), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    
    plt.show()

main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:04:52 2018

@author: xiang
"""

# 1D ADVECTION EQUATION with FD #

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.linalg import solve

#### PART 1: Lax-Wendroff scheme #######################

### Parameteres 
pi = 3.14159265358979323846
a = 1.  #coefficient physique
domain_x = 1  #taille du domaine
Time = 0.1  #temps d'integration  # print("Time final=",Time)

##  PARAMETRES NUMERIQUES
NrX = 1001  #nombre de points de grille
dx = domain_x / (NrX - 1) #pas de grille (espace)

## CFL control
# dt = 1 / 1000 # make dt the parameter to control the CFL, or make CFL to control dt
# CFL = a * dt / dx
CFL = 0.5 # CFL = a * dt / dx
dt = CFL * dx / a   #pas de grille (temps)
NrT = int(Time / dt)  #nombre de pas de temps
# print("Nombre pas de temps= ",NrT, "  ;  ", "dt= ",dt)


xx = np.zeros(NrX)
for i in np.arange(0,NrX):
      xx[i]=i * dx

### Initialisation
ddU = np.zeros(NrX)
U_data = np.zeros(NrX)
U_old = np.zeros(NrX)
U_int = np.zeros(NrX)
U_new = np.zeros(NrX)

# **** please choose between case 1 or case 2 

### ** Case 1:  u_0 = sin(2 * pi * x) *************************
U_data = np.sin(2 * pi * xx)

#### ** Case 2:  u_0 = piecewise function *************************
#for j in np.arange(0,NrX):
#    U_data[j] = 0.
#    if (0.25< xx[j] < 0.75): 
#        U_data[j] = 1.
#        
        
### assign the value to U_old
for i in np.arange(0,NrX):
    U_old[i]= U_data[i]
    U_int[i]= U_data[i]


plt.figure(-1)
plt.plot(xx,U_old, 'r.')
# plt.legend(("Time=0."), 'best')
plt.xlabel('x')
plt.ylabel('u')
plt.title('U value at time: t = 0')

random.seed()

time=0.



####  solve the problem with Lax-Wendroff scheme
'''
# here we calculate the u(x, t) for different t at all x, in each iteration, 
# u only saves the value for all x of a specific time t
for n in np.arange(0,NrT):
    
    time = time + dt    
    if (n % 100 == 0): # display current time
        print ("Time=",time)    
        
    #  Lax-Wendroff scheme
    for j in np.arange(1,NrX-1):
        ddU[j] = (U_old[j+1] - U_old[j-1] + 
                 CFL * (2 * U_old[j] - U_old[j+1] - U_old[j-1])) / 2.
    
    # The following holds becase the function is periodic, with      
    ddU[0] = (U_old[1] - U_old[NrX - 2] + 
       CFL * (2 * U_old[0] - U_old[1] - U_old[NrX - 2])) / 2.      

    # Acutalisation:   LW scehams explicites
    for j in np.arange(0,NrX-1):
        #U_new[j] = U_old[j] - (dt / dx) * ddU[j]
        U_new[j] = U_old[j] - CFL * ddU[j]
        
    U_new[NrX - 1] = U_new[0] 
    # update the u value for the current time 
    U_old = U_new    
        
'''



####  solve the problem with implicit scheme
'''
##  Fill the matrix for the implicit scheme Remplissage de la matrice pour le schema implicite
Nr_MX = NrX

second_membre = np.ones(Nr_MX-1)
res = np.zeros(Nr_MX - 1)
A = np.zeros((Nr_MX - 1, Nr_MX - 1), float)

for j in np.arange(0,Nr_MX - 1):
    A[j,j]=1.
for j in np.arange(0,Nr_MX - 2):
    A[j,j + 1] = CFL / 2.
    A[j + 1, j] = - CFL / 2.
    
A[Nr_MX - 2, 0] = CFL / 2.
A[0, Nr_MX - 2] = - CFL / 2.
    
B = np.linalg.inv(A)

for n in np.arange(0,NrT):    
    time = time + dt   
    U_new[:NrX-1] = solve(A, U_old[:NrX-1])   
    U_old[:NrX-1] = U_new[:NrX-1]  
''' 
        

    
####  solve the problem with centered explicit scheme 
'''
for n in np.arange(0,NrT):
    
    time = time + dt             
    #  centered explicit scheme 
    for j in np.arange(1,NrX-1):
        ddU[j] = (U_old[j+1] - U_old[j-1]) / 2.
    
    # The following holds becase the function is periodic    
    ddU[0] = (U_old[1] - U_old[NrX - 2] ) / 2.      

    # Acutalisation:   centered explicit scheme
    for j in np.arange(0,NrX-1):
        U_new[j] = U_old[j] - CFL * ddU[j]
        
    U_new[NrX - 1] = U_new[0] 
    # update the u value for the current time 
    U_old[:] = U_new[:]  
'''   
    
####  solve the problem with upwind scheme
'''
for n in np.arange(0,NrT):
    
    time = time + dt             
    for j in np.arange(1,NrX-1):
        ddU[j] = U_old[j] - U_old[j-1]
    
    # The following holds becase the function is periodic    
    ddU[0] = U_old[1] - U_old[NrX - 2]       

    # Acutalisation:   upwind scheme
    for j in np.arange(0,NrX-1):
        U_new[j] = U_old[j] - CFL * ddU[j]
        
    U_new[NrX - 1] = U_new[0] 
    # update the u value for the current time 
    U_old[:] = U_new[:] 
'''    
    

####  solve the problem with Monte-Carlo technique
for n in np.arange(0,NrT):
    
    # Acutalisation:   Monte-Carlo technique
    for j in np.arange(1,NrX-1):
        ran_num = random.uniform(0,1) 
        if ran_num >= 0 and ran_num < CFL:
            U_new[j] = U_old[j-1]
        elif CFL <= ran_num and ran_num <= 1:
            U_new[j] = U_old[j]
            
    ran_num = random.uniform(0,1) 
    if ran_num >= 0 and ran_num < CFL:
        U_new[0] = U_old[NrX-2]
    elif CFL <= ran_num and ran_num <= 1:
        U_new[0] = U_old[0]
                    
    # update the u value for the current time 
    U_old[:] = U_new[:] 
    
    
####  solve the problem with Glimm scheme
'''
for n in np.arange(0,NrT):
    
    # Acutalisation:   Glimm scheme
    ran_num = random.uniform(0,1) 
    if ran_num >= 0 and ran_num < CFL:
        U_new[0] = U_old[NrX-2]
        for j in np.arange(1,NrX-1):
            U_new[j] = U_old[j-1]
    elif CFL <= ran_num and ran_num <= 1:
        for j in np.arange(0,NrX-1):
            U_new[j] = U_old[j]
            
                    
    # update the u value for the current time 
    U_old[:] = U_new[:]      

'''    
    
### Plot the numerical result
print ("tFinal=",time)
plt.figure(0)
plt.plot(xx, U_new,  'r.')
plt.legend(("Time=Time"), 'best')
plt.xlabel('x')
plt.ylabel('u')    
plt.title('Numerical solution at the end of the time')
plt.show()    
    

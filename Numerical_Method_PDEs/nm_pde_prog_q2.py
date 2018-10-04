# 1D ADVECTION EQUATION with FD #
# 1D ADVECTION EQUATION with FD #
# 1D ADVECTION EQUATION with FD #

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
from matplotlib.lines import Line2D

import numpy as np
import random

pi=3.14159265358979323846


# definition de la vitesse fonction de x: a(x)=celerite(x)
def celerite(x):
    pi=3.14159265358979323846
    return -np.abs(x) ** 0.5




# PARAMETRES PHYSIQUES
Coeff = 1.  #coefficient physique
Lx = 1.0  #taille du domaine

T = 1. #temps d'integration
#print("T final=",T)

#print (np.log(2))

# PARAMETRES NUMERIQUES
NX = 81  #nombre de points de grille
dx = 2*Lx/(NX-1) #pas de grille (espace)
CFL=1

dt = 0.01    #pas de grille (temps)



NT = int(T/dt)  #nombre de pas de temps
print("NX= ",NX)
print("NT= ",NT)
print ("dt= ",dt)
#print("dt= ",dt)



# Pour la figure
xx = np.zeros(NX)
#print ("xx1",xx)
for i in np.arange(0,NX):
      xx[i]=-1+i*dx

#print ("xx",xx)

#plt.ion()
#ax = plt.gca(projection='3d')


#Initialisation
aa   = np.ones(NX)
vvt=np.zeros(NT)
vvs=np.zeros(NT)
U_data = np.zeros((NX,NT))
U_old = np.zeros(NX)
U_int = np.zeros(NX)
U_new = np.zeros(NX)
#print ("vvt",vvt)

for i in np.arange(0,NX):
    aa[i] = celerite(xx[i])
#print ("aa",aa)

for i in np.arange(0,NX):
    U_data[i,0]=xx[i]
    

# Creneau
""""
for j in np.arange(0,NX):
    U_data[j]=0.
    if (0.25<xx[j]<0.75): 
        U_data[j]=1.
"""


plt.figure(-1)
plt.plot(xx,aa)
plt.legend(("vitesse"), 'best')
plt.xlabel('x')
plt.ylabel('v')
#plt.ylabel('YY')
#plt.show()



random.seed()


time=0.
for n in np.arange(1,NT):
    
    time=time+dt    
    if (n%100==0):    
        print (n,"time= ",time,dt)  
    for j in np.arange(0,NX):
        U_data[j,n] = U_data[j,n-1]+dt*celerite(U_data[j,n-1]) 
        #U_data[j,n] = U_data[j,n-1]+dt*celerite(U_data[j,0]) 
        
#print ("U_data",U_data)    
 
# Visualisation
fig = plt.figure()
ax = fig.add_subplot(111)

for j in np.arange(0,NX):
    for n  in np.arange(0,NT):
        vvt[n]=n*dt
        vvs[n]=U_data[j,n]
#    print ("vvt",vvt)
#    print ("vvs",vvs)
    line = Line2D(vvs, vvt)
    ax.add_line(line)
    ax.set_xlim(-1, 1)

    ax.set_ylim(0,1)

#print(U_new)   
plt.show()


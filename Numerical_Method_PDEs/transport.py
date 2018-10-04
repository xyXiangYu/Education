# 1D ADVECTION EQUATION with FD #
# 1D ADVECTION EQUATION with FD #
# 1D ADVECTION EQUATION with FD #

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random


pi=3.14159265358979323846

# definition de la vitesse fonction de x: a(x)=celerite(x)
def celerite(x):
    pi=3.14159265358979323846
    return x #np.sin(pi*x)



# PARAMETRES PHYSIQUES
Coeff = 1.  #coefficient physique
Lx = 1.0  #taille du domaine

T = 0.4 #temps d'integration
print("T final=",T)

#print (np.log(2))

# PARAMETRES NUMERIQUES
NX = 101  #nombre de points de grille
dx = 2*Lx/(NX-1) #pas de grille (espace)
CFL=0.5

dt = CFL*dx/Coeff    #pas de grille (temps)

NT = int(T/dt)  #nombre de pas de temps
#print("Nombre pas de temps= ",NT)
#print("dt= ",dt)


# Pour la figure
xx = np.zeros(NX)
for i in np.arange(0,NX):
      xx[i]=-1+i*dx

#plt.ion()
#ax = plt.gca(projection='3d')


#Initialisation
ddU   = np.zeros(NX)

# u_0=sin
aa   = np.zeros(NX)
for i in np.arange(0,NX):
    aa[i] = celerite(xx[i])
    
U_data = np.zeros(NX)
U_old =  np.zeros(NX)
U_int =  np.zeros(NX)
U_new =  np.zeros(NX)
U_sol =  np.zeros(NX)

# Creneau
""""
for j in np.arange(0,NX):
    U_data[j]=0.
    if (0.25<xx[j]<0.75): 
        U_data[j]=1.
"""
### the following code is to generate the hat function (non-regular initial condition)
for i in np.arange(0,NX):
    U_old[i]=0
    if (-0.2<xx[i]<0.2):
        U_old[i]=1
    U_sol[i]=0
    if (-0.2<xx[i]*np.exp(-T)<0.2):
        U_sol[i]=0
        
    


U_new = np.zeros(NX)

#print("init, U_new",U_new)
#print("init, U_old",U_old)
#print (xx)
#print (xx[5])


plt.figure(-1)
plt.plot(xx,U_old)
plt.legend(("t=0."), 'best')
plt.xlabel('x')
plt.ylabel('u')
#plt.ylabel('YY')
#plt.show()


# Boucle en temps

random.seed()


time=0.
for n in np.arange(0,NT):
    
    time=time+dt    
    if (n%10==0):
 #       print("")
 #       print("")
        print ("t=",time)
 #       print ("U_old, begin0",U_old)
        
 

    
# Upwind      
    # within for loop: calculate ddU for index from 1 to the end  
    for j in np.arange(1,NX-1):
        # where a[j_1]_plus = max(aa[j-1],0), and a[j]_minus = max(-aa[j],0)
        
        #ddU[j] =(max(aa[j],0)*U_old[j] -  max(-aa[j],0)*U_old[j+1])-(max(aa[j-1],0)*U_old[j-1] -  max(-aa[j-1],0)*U_old[j])
        ddU[j] =max(-aa[j],0)*(U_old[j] -  U_old[j+1])-max(aa[j-1],0)*(U_old[j-1] - U_old[j])
        #U_old[j]-U_old[j-1]   
    
    # the following code updates the ddU for the first index, i.e. j == 0    
    j=0
    #ddU[j] =(max(aa[j],0)*U_old[j] -  max(-aa[j],0)*U_old[j+1])-(max(aa[j-1],0)*U_old[NX-2] -  max(-aa[j-1],0)*U_old[j])
    ddU[j] =max(-aa[j],0)*(U_old[j] -  U_old[j+1])-max(aa[NX-2],0)*(U_old[NX-2] - U_old[j])
    
    

    
    
   
# Acutalisation   scehams explicites
    for j in np.arange(0,NX-1):
        U_new[j]=U_old[j]-(dt/dx)*ddU[j]
 #   print ("U_old, begin3",U_old)
    
    U_new[NX-1]=U_new[0]
    
#    print ("U_old, begin4",U_old)
    for j in np.arange(0,NX-1):
        U_old[j]=U_new[j]
    



print ("tFinal=",time)

# caclul erreur numerique
norme=0.
for j in np.arange(0,NX-1):
    norme=norme + dx*np.fabs(U_new[j]-U_sol[j])
#norme=np.sqrt(norme)
print ("Erreur/norme L1",norme)
#print(U_new)   
plt.figure(0)

plt.plot(xx,U_new,"r",marker='x')
plt.plot(xx,U_sol,"b",marker='o')
plt.legend(("t=T"), 'best')
plt.xlabel('x')
plt.ylabel('u')
      #  plt.draw()
    # plt.show()
     
 #       print ("A34",n)
     
       
  #   plotlabel= "N = " + str(n+1)

     #ax.cla()
     #ax.plot_surface(xx,yy,U_data,vmin=-0.1,vmax=0.1,cmap=cm.jet,antialiased=False,linewidth=0,rstride=1,cstride=1)
     #ax.set_zlim3d(-0.1,0.1)

 #    plt.pcolormesh(xx,yy,U_data)
#     plt.axis('image')
 #    plt.clim(-0.1,0.1)

#     plt.title(plotlabel)
 #    plt.draw()
 #    if 'qt' in plt.get_backend().lower():
 #       QtGui.qApp.processEvents()

plt.show()









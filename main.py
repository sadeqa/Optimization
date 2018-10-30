import numpy as np
import time

# Definition des constante A, B et S 
def definition_constantes():
    B = np.matrix('-1.0; 2.0; -3.5; 1.2; 1.5')
    A = np.matrix([[ 1.0, -1.0, 2.0, -0.9, 2.1],
                   [ 1.25, 2.0, 0.5, 1.2, -0.5],
                   [ -3.0, 2.3, 0.5, 1.3, -2.5],
                   [ -2.2, 2.3, 1.5, 0.5, 1.45],
                   [ -1.2, 3.0, -0.5, 0.75, -1.5]])
    
    S = A*np.transpose(A)
    return A, B, S

def f1(U,B,S):
    n=U.shape[0]
    U=np.matrix(U)
    U.shape=(n,1)
    fU = np.transpose(U) * S * U - np.transpose(B) * U;
    return float(fU)




def df1(U,B,S):
    n=U.shape[0]
    U=np.matrix(U)
    U.shape=(n,1)
    dfU = 2 * S * U - B
    dfU = np.array(dfU)
    dfU.shape=(n,)
    return dfU

def gradient_rho_constant(fun, fun_der, U0, rho, tol,args):
# Fonction permettant de minimiser la fonction f(U) par rapport au vecteur U 
# Méthode : gradient à pas fixe
# INPUTS :
# - han_f   : handle vers la fonction à minimiser
# - han_df  : handle vers le gradient de la fonction à minimiser
# - U0      : vecteur initial 
# - rho     : paramètre gérant l'amplitude des déplacement 
# - tol     : tolérance pour définir le critère d'arrêt
# OUTPUT : 
# - GradResults : structure décrivant la solution 


    itermax=10000  # nombre maximal d'itérations 
    xn=U0
    f=fun(xn,*args) # point initial de l'algorithme
    it=0         # compteur pour les itérations
    converged = False;
    
    while (~converged & (it < itermax)):
        it=it+1
        dfx=fun_der(xn,*args)       # valeur courante de la fonction à minimiser
        xnp1=xn-rho*dfx # nouveau point courant (x_{n+1})
        fnp1=fun(xnp1,*args)
        if abs(fnp1-f)<tol:
            converged = True
        xn=xnp1; f=fnp1;           # xnp1 : nouveau point courant

    GradResults = {
            'initial_x':U0,
            'minimum':xnp1,
            'f_minimum':fnp1,
            'iterations':it,
            'converged':converged
            }
    return GradResults


A,B,S = definition_constantes()
d=B.shape[0]
x0 = np.ones((d,))

#debut = time.time()
#GradResults=gradient_rho_constant(f1,df1,x0,rho=0.02,tol=1e-6,args=(B,S))
#tps_ecoule = time.time()-debut
#print('tps écoulé (gradient_rho_constant):',tps_ecoule)
#print(GradResults['minimum'])


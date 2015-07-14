import numpy as np
from matplotlib import pyplot as plt
import MH_module as MH
import ploter_MH as PMH
import scipy.stats

#A = np.matrix([[3,2],[2,6]])
#x_mean = np.array([1,-5])

A = np.matrix([[8,2],[3,4]])
x_mean = np.array([4,-5])
prop="Gamma"

#prop="Gaussian"
if prop=="Gaussian":  
    argf = [np.array([0,0]),5*np.matrix(np.random.randn(2,2))] # this is B
    while np.all(np.linalg.eigvals(argf[1]) > 0)==0:
        argf = [np.array([0,0]),5*np.matrix(np.random.randn(2,2))]
    #here we checked that B is positive definite, otherwise, we generate a new one
    def prop_dist(*arg):
        return np.random.multivariate_normal(*arg)

    def prop_func(*args):
        return MH.simple_2D_Gauss(*args)


if prop=="Gamma":    
    argf = [2] # this is the shape of the Gamma function
    def prop_dist(*args):
        return np.random.gamma(*args,size=2)

    def prop_func(*args):
        return scipy.stats.gamma.pdf(*args)
        #return MH.simple_2D_Gauss(*args)

    



niter = 10000
i=0

while i < 1:
    guess = 10*np.random.uniform(-10,10,2)
    #try:
    if 0==0:
        testss = np.array(MH.MCMC(guess, MH.simple_2D_Gauss,prop_dist, prop_func,niter,[x_mean,A],argf))#np.array([0,0]),B]))
        print "argument of the proposal = ",argf,"\ninitial guess = ",guess
        plt.figure(figsize=(8,8))
        #plt.hist2d(testss[:,0],testss[:,1],100,cmap='Blues')
        axes = PMH.scat_and_1D(testss[:,0],testss[:,1])
        axes[0].plot(guess[0],guess[1],"or",markersize=8,mew=3,label="initial guess")
        axes[0].plot(x_mean[0],x_mean[1],"xg",markersize=8,mew=3,label="real mean")        
        if prop == "Gaussian":
            PMH.plot_theoretical(axes,A,x_mean,1,alpha = 0.2,color="red")
            PMH.plot_theoretical(axes,argf[1],guess,1,marginals =0,alpha = 0.2,color="black")
        plt.legend()
        plt.show()
        i+=1
    #except:
    #    pass
    
    
#plt.show()
#stop




#grid = np.zeros((dim,dim))


#grid = MH.grid_test(minmax,dim, MH.quadratic_form,A,b)

#for i,xi in enumerate(np.arange(minmax[0],minmax[1],(minmax[1]-minmax[0])/float(dim))):
#    for j,xj in enumerate(np.arange(minmax[0],minmax[1],(minmax[1]-minmax[0])/float(dim))):
#        grid[i,j] = MH.quadratic_form(np.array([xi,xj]),A,b)


#plt.imshow(grid,interpolation='nearest')

#plt.xticks(np.linspace(0,dim,6),np.linspace(minmax[0],minmax[1],6))
#plt.yticks(np.linspace(0,dim,6),np.linspace(minmax[0],minmax[1],6))
#plt.yticks(np.arange(minmax[0],minmax[1],(minmax[1]-minmax[0])/float(dim)))

#plt.show()

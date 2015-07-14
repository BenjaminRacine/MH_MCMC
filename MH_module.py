import numpy as np



def Equation_ap(x,A,b):
    """
    returns Ax-b
    Keyword Arguments:
    A -- Matrix
    x -- guess vector
    b -- vector
    """
    return np.dot(A,x) - b

def quadratic_form(x,A,b):
    """
    Keyword Arguments:
    A -- 
    x -- 
    b -- 
    """
    return 1/2. * np.dot(np.dot(x.T,A),x) - np.dot(b.T,x) 


def grid_test(minmax,dim,func,*args):
    grid = np.zeros((dim,dim))
    for i,xi in enumerate(np.arange(minmax[0],minmax[1],(minmax[1]-minmax[0])/float(dim))):
        for j,xj in enumerate(np.arange(minmax[0],minmax[1],(minmax[1]-minmax[0])/float(dim))):
            grid[i,j] = func(np.array([xi,xj]),*args)
    return grid
    





def MCMC(guess,step,func,niter,*arg):
    tests = []
    i=0
    out = func(guess,*arg)
    a = np.exp(-0.5*(out**2))
    while i<niter:
        guess_new=guess+step*np.random.randn(2)
        out_new = func(guess_new,*arg)
        a_new = np.exp(-0.5*(out_new**2))
        rat = a_new/a
        if rat>=1:
            out=out_new
            tests.append((guess_new))
        elif rat<1:
            u = np.random.rand(1)
            if u > rat:
                out=out_new
                tests.append((guess_new))
            else:
                pass
        
        i+=1
    return tests


def simple_2D_Gauss(x,*arg):
    """
    Keyword Arguments:
    x -- vector (np.array)
    *arg are:
    x_mean -- the mean vector (np.array)
    Cov -- covariance Matrix (np.matrix)
    
    faster than scipy.stats.multivariate_normal.pdf(x,mean,cov) for some reason
    """
    
#    return float(np.exp(-0.5 * np.dot(np.dot((x-x_mean).T,Cov.I),x-x_mean)))
    return float(np.exp(-0.5 * np.dot(np.dot((x-arg[0]).T,arg[1].I),x-arg[0])))




def MCMC(guess,functional_form,proposal,proposal_fun,niter,*arg):
    """
    Keyword Arguments:
    guess -- initial guess vector (np.array)
    functional_form -- form you are trying to sample
    proposal -- proposal random generator for next step
    proposal_fun -- proposal function to calculate ratios (next version should have both proposal and proposal_fun in one go)
    niter -- number of iterations
    *arg -- arguments that could be used by the functional form, and the proposal: *arg[0] for the functional, *arg[1] for the proposal. It can be for example *arg = [A,x],[]
    """
    tests = []
    i=0
    pi_out = functional_form(guess,*arg[0])
    tests.append((guess))
    while i<niter:
        guess_new = guess + proposal(*arg[1])
        A = min(1,functional_form(guess_new,*arg[0])/functional_form(guess,*arg[0])*proposal_fun(guess-guess_new,*arg[1])/proposal_fun(guess_new-guess,*arg[1]))
        #### will have to implement a log 
        if A==1:
            guess=guess_new
            tests.append((guess))
        elif A<1:
            u = np.random.rand(1)
            if u <= A:
                guess=guess_new
                tests.append((guess))
            else:
                pass
        i+=1
    return tests

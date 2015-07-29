import numpy as np
from matplotlib import pyplot as plt
import sys

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
    acceptance = 0
    tests = []
    i=0
    pi_out = functional_form(guess,*arg[0])
    tests.append((guess))
    while i<niter:
        guess_new = guess + proposal(*arg[1])
        A = min(1,functional_form(guess_new,*arg[0])/functional_form(guess,*arg[0])*proposal_fun(guess,guess_new,*arg[1])/proposal_fun(guess_new,guess,*arg[1]))
        #### will have to implement a log 
        if A==1:
            guess=guess_new
            tests.append((guess))
            acceptance+=1
        elif A<1:
            u = np.random.rand(1)
            if u <= A:
                guess=guess_new
                tests.append((guess))
                acceptance+=1
            else:
                pass
        i+=1
    print "acceptance rate = ",float(acceptance)/niter
    return tests


def MCMC_log(guess,functional_form,proposal,proposal_fun,niter,*arg):
    """
    Same as previous, but for log likelihood

    Keyword Arguments:
    guess -- initial guess vector (np.array)
    functional_form -- form you are trying to sample
    proposal -- proposal random generator for next step
    proposal_fun -- proposal function to calculate ratios (next version should have both proposal and proposal_fun in one go)
    niter -- number of iterations
    *arg -- arguments that could be used by the functional form, and the proposal: *arg[0] for the functional, *arg[1] for the proposal. It can be for example *arg = [A,x],[]
    """
    acceptance = 0
    tests = []
    failed = []
    i=0
    f_old = functional_form(guess,*arg[0])
    tests.append((guess))
    while i<niter:
        print i
        guess_new = guess + proposal(*arg[1])
        f_new = functional_form(guess_new,*arg[0])
        A = min(0,f_new-f_old+proposal_fun(guess,guess_new,*arg[1])-proposal_fun(guess_new,guess,*arg[1]))
        print A,"f_new = ",f_new,"f_old = ",f_old#,"prop(old/new) = ",proposal_fun(guess,guess_new,*arg[1]), "prop(new/old) = ",proposal_fun(guess_new,guess,*arg[1])
        #### will have to implement a log 
        if A==0:
            guess=guess_new
            tests.append((guess))
            acceptance+=1
            f_old = f_new
        elif A<0:
            u = np.log(np.random.rand(1))
            if u <= A:
                guess=guess_new
                tests.append((guess))
                acceptance+=1
                f_old = f_new
            else:
                failed.append((guess_new))
                pass
        i+=1
    print "acceptance rate = ",float(acceptance)/niter
    return tests,failed




def MCMC_log_test(guess,functional_form,proposal,proposal_fun,niter,*arg):
    """
    Same as previous, but for log likelihood

    Keyword Arguments:
    guess -- initial guess vector (np.array)
    functional_form -- form you are trying to sample
    proposal -- proposal random generator for next step
    proposal_fun -- proposal function to calculate ratios (next version should have both proposal and proposal_fun in one go)
    niter -- number of iterations
    *arg -- arguments that could be used by the functional form, and the proposal: *arg[0] for the functional, *arg[1] for the proposal. It can be for example *arg = [A,x],[]
    """
    Pid = np.random.randint(0,10000)
    print "Pid = %d"%Pid
    acceptance = 0
    tests = []
    failed = []
    i=0
    f_old,Cl = functional_form(guess,*arg[0])
    #plt.figure(10)
    #plt.loglog(Cl,"y",label="initial guess",)
    tests.append((guess))
    Cls = []
    guesses = []
    As = []
    flag = []
    like = []
    failed = 0
    while i<niter:
        try: 
            print i
            guess_new = guess + proposal(*arg[1])
            guesses.append(guess_new)
            f_new,Cl = functional_form(guess_new,*arg[0])
            Cls.append(Cl)
            A = min(0,f_new-f_old+proposal_fun(guess,guess_new,*arg[1])-proposal_fun(guess_new,guess,*arg[1]))
            print A,"f_new = ",f_new,"f_old = ",f_old, "guess_new = ", guess_new, "guess_old = ",guess
            As.append(A)
            like.append(f_new)
            if A==0:
                guess=guess_new
                flag.append(1)
                acceptance+=1
                f_old = f_new
            elif A<0:
                u = np.log(np.random.rand(1))
                print "u = ",u
                if u <= A:
                    guess=guess_new
                    flag.append(2)
                    acceptance+=1
                    f_old = f_new
                    print "Lucky choice ! f_old = ",f_old
                else:
                    flag.append(0)
                    pass
            i+=1
            if i%100==0:
                np.save("tempo_MC_chain_%d.npy"%Pid,[guesses,flag,like,As,Cls])
                print "temporary file saved"
        except:
            failed+=1
            #plt.draw()
    print "acceptance rate = ",float(acceptance)/niter
    print "%d fails"%failed
    return guesses,flag,like,As,Cls

def MCMC_log_new(guess,functional_form,proposal,proposal_fun,niter,*arg):
    """
    Same as previous, but for log likelihood

    Keyword Arguments:
    guess -- initial guess vector (np.array)
    functional_form -- form you are trying to sample
    proposal -- proposal random generator for next step
    proposal_fun -- proposal function to calculate ratios (next version should have both proposal and proposal_fun in one go)
    niter -- number of iterations
    *arg -- arguments that could be used by the functional form, and the proposal: *arg[0] for the functional, *arg[1] for the proposal. It can be for example *arg = [A,x],[]
    """
    Pid = np.random.randint(0,10000)
    print "Pid = %d"%Pid
    failed = []
    i=0
    f_old = functional_form(guess,*arg[0])
    #plt.figure(10)
    #plt.loglog(Cl,"y",label="initial guess",)
    guesses = []
    guesses.append(guess)
    flag = []
    failed = 0
    while i<niter:
        try: 
            print i
            guess_new = guess + proposal(*arg[1])
            guesses.append(guess_new)
            if np.min(guess_new)<0:
                print "negative param !"
                flag.append(0)
            else:
                f_new = functional_form(guess_new,*arg[0])
                A = min(0,f_new-f_old+proposal_fun(guess,guess_new,*arg[1])-proposal_fun(guess_new,guess,*arg[1]))
                print A,"f_new = ",f_new,"f_old = ",f_old, "guess_new = ", guess_new, "guess_old = ",guess
                if A==0:
                    guess=guess_new
                    flag.append(1)
                    f_old = f_new
                elif A<0:
                    u = np.log(np.random.rand(1))
                    print "u = ",u
                    if u <= A:
                        guess=guess_new
                        flag.append(2)
                        f_old = f_new
                        print "Lucky choice ! f_old = ",f_old
                    else:
                        flag.append(0)
                        pass
            i+=1
            if i%100==0:
                np.save("tempo_MC_chain_%d.npy"%Pid,[np.array(guesses),np.array(flag)])
                print "temporary file saved"
        except:
            failed+=1
            print "error: %s on line %s"%(sys.exc_info()[0],sys.exc_info()[-1].tb_lineno)
            #plt.draw()
    print "%d fails"%failed
    return np.array(guesses),np.array(flag)


def MCMC_log_new_priors(guess,functional_form,proposal,proposal_fun,niter,forced_priors,*arg):
    """
    Same as previous, but for log likelihood

    Keyword Arguments:
    guess -- initial guess vector (np.array)
    functional_form -- form you are trying to sample
    proposal -- proposal random generator for next step
    proposal_fun -- proposal function to calculate ratios (next version should have both proposal and proposal_fun in one go)
    niter -- number of iterations
    *arg -- arguments that could be used by the functional form, and the proposal: *arg[0] for the functional, *arg[1] for the proposal. It can be for example *arg = [A,x],[]
    """
    Pid = np.random.randint(0,10000)
    print "Pid = %d"%Pid
    failed = []
    i=0
    f_old,Cl = functional_form(guess,*arg[0])
    #plt.figure(10)
    #plt.loglog(Cl,"y",label="initial guess",)
    guesses = []
    guesses.append(guess)
    flag = []
    Cls=[]
    Cls.append(Cl)
    like=[]
    like.append(f_old)
    failed = 0
    while i<niter:
        try: 
            guess_new = guess + proposal(*arg[1])
            guesses.append(guess_new)
            print guess_new
            f_new,Cl = functional_form(guess_new,*arg[0])
            like.append(f_new)
            Cls.append(Cl)
            if (np.array(guess_new)<forced_priors).sum()>0:
                print "Priors : ",guess_new," vs. ", forced_priors
                flag.append(-1)
                #elif np.min(guess_new)<0:
            #    print "Negative param ! ",guess_new
            #    flag.append(0)
            else:
                #f_new,Cl = functional_form(guess_new,*arg[0])
                #like.append(f_new)
                #Cls.append(Cl)
                A = min(0,f_new-f_old+proposal_fun(guess,guess_new,*arg[1])-proposal_fun(guess_new,guess,*arg[1]))
                print A,"f_new = ",f_new,"f_old = ",f_old, "guess_new = ", guess_new, "guess_old = ",guess
                if A==0:
                    guess=guess_new
                    flag.append(1)
                    f_old = f_new
                elif A<0:
                    u = np.log(np.random.rand(1))
                    print "u = ",u
                    if u <= A:
                        guess=guess_new
                        flag.append(2)
                        f_old = f_new
                        print "Lucky choice ! f_old = ",f_old
                    else:
                        flag.append(0)
                        pass
            i+=1
            if i%100==0:
                np.save("tempo_MC_chain_%d.npy"%Pid,[np.array(guesses),np.array(flag),np.array(like),np.array(Cls)])
                print "temporary file saved: %d"%Pid
        except:
            failed+=1
            print "error: %s on line %s"%(sys.exc_info()[0],sys.exc_info()[-1].tb_lineno)
            #plt.draw()
    print "%d fails"%failed
    return np.array(guesses),np.array(flag),np.array(like),np.array(Cls)




def autocorr(x):
    '''
    return the autocorrelation of a given array (much faster than computing the actual function)
    '''
    aa = np.correlate(x-x.mean(), x-x.mean(), mode='full')
    maxcorr = np.argmax(aa)
    result = aa / aa[maxcorr]
    return result[np.argmax(result):]






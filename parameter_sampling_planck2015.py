import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse        
import ploter_parameters as plp
import camber as cb
import healpy as hp
import CG_functions as CG
import MH_module as MH
import PS2param_module as PS2P
import sys
try:
    from local_paths import *
except:
    print "you need to define local_paths.py, that defines, for example: \ncamb_dir = '/Users/benjar/Travail/camb/' \n and the output path for the temporary ini files: \noutput_camb = '../MH_MCMC/camb_ini/test1'"
    sys.exit()
random_id = np.random.randint(0,100000)



dd  = cb.ini2dic(camb_dir+"Planck_params_params.ini")




#which_par = [0] # exclude Theta_MC (for now, since I don't have it in CAMB, should choose another one, perhaps H0)
strings=np.array(['ombh2','omch2',"theta",'re_optical_depth','scalar_amp(1)','scalar_spectral_index(1)'])

titles = np.array(["$\Omega_b h^2$","$\Omega_c h^2$",r"100*$\theta_{MC}$",r"$\tau$","$A_s$","$n_s$"])
#inital guess, planck 2013 planck+WP, corrmat from likelihood paper
dd["output_root"] = output_camb+'_%d'%random_id

dd['ombh2'] =  0.02222
dd['omch2'] =  0.1197 
dd['re_optical_depth'] = 0.078
dd['scalar_amp(1)'] = np.exp(3.089)*1e-10
dd['scalar_spectral_index(1)'] = 0.9655



Correlation_matrix = np.matrix([
    [100,-53,45,41,38,56],
    [0,100,-45,-45,-33,-83],
    [0,0,100,28,23,48],
    [0,0,0,100,98,52],
    [0,0,0,0,100,42],
    [0,0,0,0,0,100]])

a = 0.00023**2
b = 0.0022**2
c = 0.00047**2
d = 0.019**2
e = 0.036**2#(np.exp(3.089)*1e-10)**2
f = 0.0062**2

cov_diag = np.array([a,b,c,d,e,f])
x_mean = np.array([0.02222,0.1197,1.04085,0.078,3.089,0.9655])

x_mean_2013planckWP = np.array([0.02205,0.1199,1.04131,0.089,np.exp(3.089)*1e-10, 0.9603])

#here with theta_MC : Correlation_matrix_2013 = np.matrix([[100,-43,35,27,47,19],[0,100,-39,-25,-76,3],[0,0,100,10,39,1],[0,0,0,100,25,94],[0,0,0,0,100,3],[0,0,0,0,0,100]])


cov_new = plp.cor2cov(cov_diag,Correlation_matrix)


#[which_par,:][:,which_par]
#[which_par]

nl = 1.7504523623688016e-16*1e12 * np.ones(2500)
bl = CG.gaussian_beam(2500,5)

Cl = cb.generate_spectrum(dd)
lmax = Cl.shape[0]-1
alm = hp.synalm(Cl[:,1])
dlm = hp.almxfl(alm,bl[:lmax+1]) + hp.synalm(nl[:lmax+1])


def plot_input(dlm):
    hp.mollview(hp.alm2map(dlm,1024),title="Input data",unit="$\mu K_{\mathrm{CMB}}$")
    plt.figure()
    plt.loglog(Cl[:,1],"r")




def run_MCMC(which_par,niter):
    cov_new_temp = cov_new[which_par,:][:,which_par]
    string_temp = strings[which_par]
    titles_temp = titles[which_par]
    x_mean_temp = x_mean[which_par]
    print titles_temp
    guess_param = PS2P.prop_dist_form_params(x_mean_temp,cov_new_temp)
    testss = np.array(MH.MCMC_log_test(guess_param, PS2P.functional_form_params_n,PS2P.prop_dist_form_params, PS2P.prop_func_form_params,niter,[dlm,string_temp,dd,nl,bl],[x_mean_temp*0,np.matrix(cov_new_temp)]))
    #print "%.2f rejected; %.2f accepted; %.2f Lucky accepted"%((flag==0).mean(),(flag==1).mean(),(flag==2).mean())
    return testss

def plot_chains(guesses,flag,titles,which_par):
    niter = len(flag)
    for i in which_par:
        plt.figure()
        plt.plot(np.arange(niter)[flag==0],guesses[flag==0],'k.',alpha = 0.2,label='Rejected')
        plt.plot(np.arange(niter)[flag==1],guesses[flag==1],'g.',label="Accepted")
        plt.plot(np.arange(niter)[flag==2],guesses[flag==2],'r.',label='Lucky accepted')
        plt.title(titles[i]+"MC chains")
        plt.xlabel("Iterations")
        plt.ylabel(titles[i])
        plt.plot(np.arange(niter),x_mean[i]*np.ones(niter),color='b',label = "Planck prior")
        plt.fill_between(np.arange(niter),x_mean[i]-np.sqrt(cov_diag[i]),x_mean[i]+np.sqrt(cov_diag[i]),color='b',alpha=0.2)
        plt.legend(loc="best")
        print titles[i],": %.2f rejected; %.2f accepted; %.2f Lucky accepted"%((flag==0).mean(),(flag==1).mean(),(flag==2).mean())

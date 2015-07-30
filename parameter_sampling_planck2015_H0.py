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
strings=np.array(['ombh2','omch2','re_optical_depth','scalar_amp(1)','scalar_spectral_index(1)','hubble'])

titles = np.array(["$\Omega_b h^2$","$\Omega_c h^2$",r"$\tau$","$A_s$","$n_s$","$H_0$"])
#inital guess, planck 2013 planck+WP, corrmat from likelihood paper
dd["output_root"] = output_camb+'_%d'%random_id

dd['ombh2'] =  0.02222
dd['omch2'] =  0.1197 
dd['re_optical_depth'] = 0.078
dd['scalar_amp(1)'] = np.exp(3.089)*1e-10
dd['scalar_spectral_index(1)'] = 0.9655
dd['hubble'] = 67.31

priors_central = np.array([0,0,0.07,0,0,0])
priors_invvar = np.array([0,0,1/0.02**2,0,0,0])
#forced_priors = np.array([0,0,0.04,0,0,0])

#Correlation_matrix = np.matrix([
#    [100,-53,45,41,38,56],
#    [0,100,-45,-45,-33,-83],
#    [0,0,100,28,23,48],
#    [0,0,0,100,98,52],
#    [0,0,0,0,100,42],
#    [0,0,0,0,0,100]])

#a = 0.00023**2
#b = 0.0022**2
#c = 0.00047**2
#d = 0.019**2
#e = 0.036**2#(np.exp(3.089)*1e-10)**2
#f = 0.0062**2

#cov_diag = np.array([a,b,c,d,e,f])
#x_mean = np.array([0.02222,0.1197,1.04085,0.078,3.089,0.9655])
x_mean = np.array([0.02222,0.1197,0.078,3.089,0.9655,67.31])

#x_mean_2013planckWP = np.array([0.02205,0.1199,1.04131,0.089,np.exp(3.089)*1e-10, 0.9603])

#here with theta_MC : Correlation_matrix_2013 = np.matrix([[100,-43,35,27,47,19],[0,100,-39,-25,-76,3],[0,0,100,10,39,1],[0,0,0,100,25,94],[0,0,0,0,100,3],[0,0,0,0,0,100]])


#cov_new = plp.cor2cov(cov_diag,Correlation_matrix)
cov_new = np.load("cov_tableTT_lowEB_2_3_5_6_7_23.npy")
cov_diag = np.diag(cov_new)
#[which_par,:][:,which_par]
#[which_par]

nl = 1.7504523623688016e-16*1e12 * np.ones(2500)
bl = CG.gaussian_beam(2500,5)

Cl = cb.generate_spectrum(dd)
lmax = Cl.shape[0]-1
alm = hp.synalm(Cl[:,1])
dlm = hp.almxfl(alm,bl[:lmax+1])
nlm = hp.synalm(nl[:lmax+1])
dlm = dlm+nlm

def plot_input(dlm):
    hp.mollview(hp.alm2map(dlm,1024),title="Input data",unit="$\mu K_{\mathrm{CMB}}$")
    plt.figure()
    plt.loglog(Cl[:,1],"r")





def run_MCMC(which_par,niter,save_title):
    cov_new_temp = cov_new[which_par,:][:,which_par]
    string_temp = strings[which_par]
    titles_temp = titles[which_par]
    x_mean_temp = x_mean[which_par]
    priors_central_temp = priors_central[which_par]
    priors_invvar_temp = priors_invvar[which_par]
    print titles_temp
    guess_param = PS2P.prop_dist_form_params(x_mean_temp,cov_new_temp)
    testss = np.array(MH.MCMC_log(guess_param, PS2P.functional_form_params_n,PS2P.prop_dist_form_params, PS2P.prop_func_form_params,niter,PS2P.Gaussian_priors_func,[dlm,string_temp,dd,nl,bl],[x_mean_temp*0,np.matrix(cov_new_temp)],[priors_central_temp,priors_invvar_temp]))
    #print "%.2f rejected; %.2f accepted; %.2f Lucky accepted"%((flag==0).mean(),(flag==1).mean(),(flag==2).mean())
    np.save("chain_%s_%s_%d_%d.npy"%(save_title,str(which_par).replace(',','').replace('[','').replace(']','').replace(' ',''),np.random.randint(0,100000),niter),testss)
    return testss






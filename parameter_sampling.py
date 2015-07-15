import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse        
import ploter_parameters as plp
import camber as cb
import healpy as hp
import CG_functions as CG
import MH_module as MH

camb_dir = "/Users/benjar/Travail/camb/"
dd  = cb.ini2dic("/Users/benjar/Travail/camb/Planck_params_params.ini")




#inital guess, planck 2013 planck+WP, corrmat from likelihood paper
dd["output_root"] = '/Users/benjar/Travail/Codes/CG_git/MH_MCMC/camb_ini/test1'

dd['ombh2'] =  0.02205
dd['omch2'] =  0.1199
dd['re_optical_depth'] = 0.089
dd['scalar_spectral_index(1)'] = 0.9603
dd['scalar_amp(1)'] = np.exp(3.089)*1e-10


strings=['ombh2','omch2','re_optical_depth','scalar_spectral_index(1)','scalar_amp(1)']

Correlation_matrix = np.matrix([
[100,-43,27,47,19],
[0,100,-25,-76,3],
[0,0,100,25,94],
[0,0,0,100,3],
[0,0,0,0,100]])


a = 0.00028**2
b = 0.0027**2
#c = 0.00063**2
d = 0.012**2
e = 0.0073**2
f = 0.024**2

cov_diag = np.array([a,b,d,e,f])
x_mean = np.array([0.02205,0.1199,0.089,0.9603,3.089])

#cov_diag = np.array([a,b,c,d,e,f])
#x_mean = np.array([0.02205,0.1199,1.04131,0.089,0.9603,3.089])

#here with theta_MC : Correlation_matrix = np.matrix([[100,-43,35,27,47,19],[0,100,-39,-25,-76,3],[0,0,100,10,39,1],[0,0,0,100,25,94],[0,0,0,0,100,3],[0,0,0,0,0,100]])

#covariance_matrix = np.matrix([[a,-0.43*np.sqrt(a*b), ],[-0.43*np.sqrt(a*b),b]])

cb.run_camb(dd,"/Users/benjar/Travail/Codes/CG_git/MH_MCMC/camb_ini/temporary.ini")
Cl = np.loadtxt("%s_scalCls.dat"%(dd["output_root"]))
plt.plot(Cl[:,0],Cl[:,1])

cov_new = plp.cor2cov(cov_diag,Correlation_matrix)
plt.figure()
axS,axh = plp.Triangle_plot_Cov(cov_new,x_mean)


nl = 1.7504523623688016e-16*1e12 * np.ones(2500)
bl = CG.gaussian_beam(2500,5)

def functional_form_params(x,*arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    x_str -- the dictionary strings corresponding to x
    params -- a camber dictionnary
    noise -- a noise power spectrum
    beam -- a beam power spectrum
    """
    strings = arg[0]
    params = arg[1]
    noise = arg[2]
    beam = arg[3]
    for i in range(np.size(x)):
        params[strings[i]]=x[i]
    Cl = cb.generate_spectrum(params)
    lmax = Cl.shape[0]
    alm = hp.synalm(Cl[:,1])
    dlm = hp.almxfl(alm,beam[:lmax]) + hp.synalm(noise[:lmax])
    tt = np.real(np.vdot(dlm.T,hp.almxfl(dlm,1/(Cl[:,1]-noise[:lmax]))))
    #print tt
    #print (noise[:lmax]+Cl[:,1]).sum()
    #determinant is the product of the diagonal element: in log:
    tt = -1/2. * tt  - 1./2 *(noise[:lmax]+Cl[:,1]).sum()
    return tt
    
    
def prop_func_form_params(param1,param2,*arg):
    """
    Keyword Arguments:
    *args are:
    x_mean -- the mean vector (np.array)
    Cov -- covariance Matrix (np.matrix)
    """
    return np.log(MH.simple_2D_Gauss(param1-param2,arg[0]*0,arg[1]))


def prop_dist_form_params(*arg):
    """
    Keyword Arguments:
    *args are:
    x_mean -- the mean vector (np.array)
    Cov -- covariance Matrix (np.matrix)
    """
    return np.random.multivariate_normal(*arg)

    


guess_param = prop_dist_form_params(x_mean,cov_new)

testss = np.array(MH.MCMC_log(guess_param, functional_form_params,prop_dist_form_params, prop_func_form_params,100,[['ombh2','omch2','re_optical_depth','scalar_spectral_index(1)','scalar_amp(1)'],dd,nl,bl],[x_mean*0,np.matrix(cov_new)]))


# def eigsorted(cov):
#     vals, vecs = np.linalg.eigh(cov)
#     order = vals.argsort()[::-1]
#     return vals[order], vecs[:,order]


# vals, vecs = eigsorted(covariance_matrix)
# theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
# width, height = 2 * 1 * np.sqrt(vals)
# ell = Ellipse(xy=x_mean, width=width, height=height, angle=theta,fill=False,color="b")
# width2, height2 = 2 * 2 * np.sqrt(vals)
# ell2 = Ellipse(xy=x_mean, width=width2, height=height2, angle=theta,fill=False,color="r")

# plt.figure()
# ax = plt.gca()
# ax.add_artist(ell)
# ax.add_artist(ell2)
# plt.xlim(0.0205,0.0245)
# plt.ylim(0.098,0.132)

# ax.set_xticks([0.021,0.022,0.023,0.024])
# ax.set_yticks([0.104,0.112,0.120,0.128])
# plt.show()

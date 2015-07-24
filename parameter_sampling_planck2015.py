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

#strings=['ombh2','omch2','re_optical_depth','scalar_amp(1)','scalar_spectral_index(1)']

titles = ["$\Omega_b h^2$","$\Omega_c h^2$",r"100*$\theta_{MC}$",r"$\tau$","$A_s$","$n_s$"]

which_par = [0] # exclude Theta_MC (for now, since I don't have it in CAMB, should choose another one, perhaps H0)


#inital guess, planck 2013 planck+WP, corrmat from likelihood paper
dd["output_root"] = '../Codes/CG_git/MH_MCMC/camb_ini/test4'

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

cov_diag = np.array([a,b,c,d,e,f])[which_par]
x_mean = np.array([0.02205,0.1197,1.04085,0.078,3.089,0.9655])[which_par]

x_mean_2013planckWP = np.array([0.02205,0.1199,1.04131,0.089,np.exp(3.089)*1e-10, 0.9603])

#here with theta_MC : Correlation_matrix_2013 = np.matrix([[100,-43,35,27,47,19],[0,100,-39,-25,-76,3],[0,0,100,10,39,1],[0,0,0,100,25,94],[0,0,0,0,100,3],[0,0,0,0,0,100]])


cov_new = plp.cor2cov(cov_diag,Correlation_matrix[which_par,:][:,which_par])
plt.figure()



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
    
#    print "input map: cl[10] = ", Cl[10,1]


def functional_form_params_o(x,*arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    dlm -- input map
    x_str -- the dictionary strings corresponding to x
    params -- a camber dictionnary
    noise -- a noise power spectrum
    beam -- a beam power spectrum
    """
    dlm = arg[0]
    strings = arg[1]
    params = arg[2].copy()
    noise = arg[3]
    beam = arg[4]
    #params["output_root"] = '../Codes/CG_git/MH_MCMC/camb_ini/test%d'%np.random.randint(100)
    for i in range(np.size(x)):
        print strings[i]
        if strings[i]=='scalar_amp(1)':
            print params[strings[i]]
            params[strings[i]]=np.exp(x[i])*1e-10
            print params[strings[i]]
        else:
            params[strings[i]]=x[i]
    Cl = cb.generate_spectrum(params)
    print params["ombh2"]
    plt.figure(10)
    plt.loglog(Cl[:,1],",")
    print "cl[10] = ", Cl[10,1]
    lmax = Cl.shape[0]
    #alm = hp.synalm(Cl[:,1])
    #dlm = hp.almxfl(alm,beam[:lmax]) + hp.synalm(noise[:lmax])
    tt = np.real(np.vdot(dlm.T,hp.almxfl(dlm,1/(beam[:lmax]**2*Cl[:,1]+noise[:lmax]))))
    print "dlm[10]",dlm[10]
    print tt
    #print (noise[:lmax]+Cl[:,1]).sum()
    #determinant is the product of the diagonal element: in log:
    tt = -1/2. * tt  - 1./2 *(noise[:lmax]+Cl[:,1]*beam[:lmax]**2).sum()
    return tt,Cl[:,1]
    

def functional_form_params(x,*arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    dlm -- input map
    x_str -- the dictionary strings corresponding to x
    params -- a camber dictionnary
    noise -- a noise power spectrum
    beam -- a beam power spectrum
    """
    dlm = arg[0]
    strings = arg[1]
    params = arg[2].copy()
    noise = arg[3]
    beam = arg[4]
    #params["output_root"] = '../Codes/CG_git/MH_MCMC/camb_ini/test%d'%np.random.randint(100)
    for i in range(np.size(x)):
        print strings[i]
        if strings[i]=='scalar_amp(1)':
            print params[strings[i]]
            params[strings[i]]=np.exp(x[i])*1e-10
            print params[strings[i]]
        else:
            params[strings[i]]=x[i]
    Cl = cb.generate_spectrum(params)
    print params["ombh2"]
    plt.figure(10)
    plt.loglog(Cl[:,1],",")
    print "cl[10] = ", Cl[10,1]
    lmax = Cl.shape[0]
    alm = hp.synalm(Cl[:,1])
    dlm = hp.almxfl(alm,beam[:lmax]) + hp.synalm(noise[:lmax])
    tt = np.real(np.vdot(dlm.T,hp.almxfl(dlm,1/(beam[:lmax]**2*Cl[:,1]+noise[:lmax]))))
    print "dlm[10]",dlm[10]
    print tt
    #print (noise[:lmax]+Cl[:,1]).sum()
    #determinant is the product of the diagonal element: in log:
    tt = -1/2. * tt  - 1./2 *(noise[:lmax]+Cl[:,1]*beam[:lmax]**2).sum()
    return tt,Cl[:,1]
    
def prop_func_form_params(param1,param2,*arg):
    """
    Keyword Arguments:
    *args are:
    x_mean -- the mean vector (np.array)
    Cov -- covariance Matrix (np.matrix)
    """
    return np.log(MH.simple_2D_Gauss(param1-param2,arg[0],arg[1]))


def prop_dist_form_params(*arg):
    """
    Keyword Arguments:
    *args are:
    x_mean -- the mean vector (np.array)
    Cov -- covariance Matrix (np.matrix)
    """
    return np.random.multivariate_normal(*arg)




guess_param = prop_dist_form_params(x_mean,cov_new)




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


def test_loglike(dlm,Cl,noise,beam):
    tt_exp = -1./2 * np.real(np.vdot(dlm.T,hp.almxfl(dlm,1/(beam[:lmax+1]**2*Cl[:,1]+noise[:lmax+1]))))
    #plt.plot(Cl[:,1])
    tt_det = - 1./2 *(np.arange(1,lmax+2)*np.log((noise[:lmax+1]+Cl[:,1]*beam[:lmax+1]**2))).sum() 
    tt_f = tt_exp  + tt_det
    return tt_exp,tt_det,tt_f#,Cl[:,1]




def functional_form_params_n(x,*arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    dlm -- input map
    x_str -- the dictionary strings corresponding to x
    params -- a camber dictionnary
    noise -- a noise power spectrum
    beam -- a beam power spectrum
    """
    dlm = arg[0]
    strings = arg[1]
    params = arg[2].copy()
    noise = arg[3]
    beam = arg[4]
    #params["output_root"] = '../Codes/CG_git/MH_MCMC/camb_ini/test%d'%np.random.randint(100)
    for i in range(np.size(x)):
        print strings[i]
        if strings[i]=='scalar_amp(1)':
            print params[strings[i]]
            params[strings[i]]=np.exp(x[i])*1e-10
            print params[strings[i]]
        else:
            params[strings[i]]=x[i]
    Cl = cb.generate_spectrum(params)
    plt.figure(10)
    plt.loglog(Cl[:,1],",")
    lmax = Cl.shape[0]
    #alm = hp.synalm(Cl[:,1])
    #dlm = hp.almxfl(alm,beam[:lmax]) + hp.synalm(noise[:lmax])
    tt = np.real(np.vdot(dlm.T,hp.almxfl(dlm,1/(beam[:lmax]**2*Cl[:,1]+noise[:lmax]))))
    #print (noise[:lmax]+Cl[:,1]).sum()
    #determinant is the product of the diagonal element: in log:
    tt = -1/2. * tt  - 1./2 *(np.arange(1,lmax+1)*np.log(noise[:lmax]+Cl[:,1]*beam[:lmax]**2)).sum()
    return tt,Cl[:,1]

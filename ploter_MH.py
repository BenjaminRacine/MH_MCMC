from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Ellipse        





def determine_bin_size(array_in,method):
    """
    returns the optimal binning, according to Freedman-Diaconis rule: see http://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    """
    sorted_arr = np.sort(array_in)
    if method == "FD":
        Q3 = np.percentile(sorted_arr,75.)
        Q1 = np.percentile(sorted_arr,25.)
        IQR = Q3-Q1
        bin_size = 2.*IQR*np.size(array_in)**(-1/3.)
    if method == "scott":
        bin_size = sorted_arr.std() *3.5/np.size(sorted_arr)**(1./3)
    return bin_size


def plot_theoretical(axes,A,x,n_sigma,marginals=1,**kwargs):
    """
    addapted from http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    Keyword Arguments:
    A -- the covariance matrix
    x -- the mean
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    vals, vecs = eigsorted(A)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * n_sigma * np.sqrt(vals)
    ell = Ellipse(xy=x, width=width, height=height, angle=theta,**kwargs)
    axes[0].add_artist(ell)
    if marginals ==1:
        x1 = np.linspace(axes[1].get_xlim()[0],axes[1].get_xlim()[1],200)
        y2 = np.linspace(axes[2].get_ylim()[0],axes[2].get_ylim()[1],200)
        axes[1].plot(x1,np.exp(-0.5*(x1-x[0])**2/A[0,0])/np.sqrt(2*np.pi*A[0,0]),**kwargs)
        axes[2].plot(np.exp(-0.5*(y2-x[1])**2/A[1,1])/np.sqrt(2*np.pi*A[1,1]),y2,**kwargs)

def scat_and_1D(x,y,**kwargs):
    """
    adapted from http://matplotlib.org/examples/pylab_examples/scatter_hist.html
    """
    
    nullfmt   = NullFormatter()
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    axScatter.scatter(x, y,marker = '.',alpha = 0.2)
    
    # now determine nice limits by hand:
    binwidthx = determine_bin_size(x,"FD")
    binwidthy = determine_bin_size(y,"FD")
    #xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    
        
    #axScatter.set_xlim( (-limx, limx) )
    #axScatter.set_ylim( (-limy, limy) )
    
    binsx = np.arange(min(x), max(x) + binwidthx, binwidthx)
    binsy = np.arange(min(y), max(y) + binwidthy, binwidthy)
    axHistx.hist(x, bins=binsx,histtype='step',normed=True)
    axHisty.hist(y, bins=binsy, histtype= 'step',orientation='horizontal',normed=True)
    
    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )
    return axScatter, axHistx,axHisty



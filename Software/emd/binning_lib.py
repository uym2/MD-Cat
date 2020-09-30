from treeswift import *
from scipy.stats import expon, lognorm
from math import *
from emd.emd_normal_lib import *

def make_bins(wl,wr,nbins,p=1.0):
    step = (log(wr)-log(wl))/(nbins-1)
    omega = [exp(log(wl)+i*step) for i in range(nbins) ]
    #if p is None:        
    #    phi = [expon.cdf(log(omega[i]+step/2)) -  expon.cdf(log(omega[i])-step/2) for i in range(nbins)]
    #else:
    #    phi = [p/nbins]*nbins    
    #step = (wr-wl)/(nbins-1)
    #omega = [wl+i*step for i in range(nbins)]
    phi = [p/nbins]*nbins
    return omega,phi

def init_bins(mu,nbins):
    sd = 0.4 
    sigma = sqrt(log(sd*sd+1))
    scale = 1/sqrt(sd*sd+1)
    wl = mu*lognorm.ppf(1.0/5/nbins,sigma,0,scale)
    wr = mu*lognorm.ppf(1-1.0/5/nbins,sigma,0,scale)
    #wl = expon.ppf(1.0/2/nbins,scale=mu)
    #wr = expon.ppf(1-1.0/2/nbins,scale=mu)
    return make_bins(wl,wr,nbins)
    #p = [i/(nbins+1) for i in range(1,nbins+1)]
    #omega = [expon.ppf(q,scale=mu) for q in p]
    #phi = [1.0/nbins]*nbins
    #return omega,phi

def refine_bins(omega,phi,k,alpha=1):
    nbins = len(omega)
    #p_high = alpha*sqrt(k)/nbins
    #p_low = 1/(alpha*sqrt(k)*nbins)
    p_high = 2.0/nbins
    p_low = 0.1/nbins

    # split all bins i which have phi[i] > p_high into k bins
    i = 0
    j = 0
    while i < len(omega): 
        if phi[i] > p_high:
            p = phi[i]
            j = i+1
            l = len(omega)
            while j<l and phi[j] > p_high:
                p += phi[j]
                j += 1    
            j -= 1                                
            #x = log(omega[i])
            #y = log(omega[j])
            #xl = log(omega[i-1]) if i>0 else None
            #yr = log(omega[j+1]) if j<l-1 else None
            #wl = exp((x+xl)/2) if xl is not None else exp(x-(yr-y)/2)
            #wr = exp((y+yr)/2) if yr is not None else exp(y+(x-xl)/2)
            x = omega[i]
            y = omega[j]
            xl = omega[i-1] if i>0 else None
            yr = omega[j+1] if j<l-1 else None
            wl = ((x+xl)/2) if xl is not None else (x-(yr-y)/2)
            wr = ((y+yr)/2) if yr is not None else (y+(x-xl)/2)
            omg,ph = make_bins(wl,wr,k*(j-i+1),p=p)
            #print(i,j)
            #print(omega[:i],omg,omega[j+1:])
            omega = omega[:i] + omg + omega[j+1:]
            phi = phi[:i] + ph + phi[j+1:]
        else:
            i += 1

    # remove all bins i which have phi[i] < p_low
    nbins = len(omega)
    i = 0
    while i < nbins:
        if phi[i] < p_low:
            omega = omega[:i] + omega[i+1:] 
            phi = phi[:i] + phi[i+1:]
            nbins -= 1
        else:
            i += 1              
    
    # normalize
    s = sum(phi)
    phi = [p/s for p in phi]            
    #phi = [1.0/len(omega)]*len(omega)
    return omega,phi        

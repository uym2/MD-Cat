from random import random
from math import *
from scipy.stats import lognorm, expon
import jenkspy

def cdf_from_pdf(p):
    c = [0]*len(p)
    c[0] = p[0]

    for i in range(1,len(p)):
        c[i] = c[i-1] + p[i]

    return c

def binary_search(arr,v,start=0,end=None):
# assuming arr is sorted from start to end
# return the index i such that arr[i-1] < v <= arr[i]
    if end is None:
        end = len(arr)-1
    if end < start:
        return None
    
    m = floor((start+end)/2) # middle index
    if v <= arr[m]:
        return m if (m == start or arr[m-1] < v) else binary_search(arr,v,start,m-1)
    else:
        return binary_search(arr,v,m+1,end)         

class multinomial:
    def __init__(self,omega,phi):
        # omega stores the values, phi stores the probability
        self.omega = omega
        self.phi = phi
        self.acc = cdf_from_pdf(phi) # accumulative density        
    
    def randomize(self):
        r = random()
        i = binary_search(self.acc,r)
        return i,self.omega[i]
    
def discrete_lognorm(mu,sd,k):
    # scipy reference https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm
    p = [i/(k+1) for i in range(1,k+1)] 
    sigma = sqrt(log(sd*sd+1))
    scale = 1/sqrt(sd*sd+1)
    nu = lognorm.ppf(p,sigma,0,scale)
    density = lognorm.pdf(nu,sigma,0,scale)
    omega = mu*nu
    phi = density/sum(density)
    
    return multinomial(omega,phi)
    
def discrete_exponential(mu,k):
    p = [i/(k+1) for i in range(1,k+1)] 
    omega = expon.ppf(p,scale=mu)
    density = expon.pdf(omega,scale=mu)
    phi = density/sum(density)
    
    return multinomial(omega,phi)

def emperical_histogram(observes,k):   
# observes is a list of observations
# k is the number of bins
# here we use Jenks natural breaks optimization to select the bins
    observes.sort()
    breaks = jenkspy.jenks_breaks(observes, nb_class=k) 
    print(observes)
    print(breaks)

    omega = []
    phi = []
    curr_sum = 0
    curr_count = 0 
    brk_idx = 1
    curr_break = breaks[brk_idx]
    N = len(observes)

    for v in observes:
        #print(v,curr_break)
        if v < curr_break:
            curr_sum += v
            curr_count += 1
        else:
            omega.append(curr_sum/curr_count)
            phi.append(curr_count/N)
            curr_sum = v
            curr_count = 1
            if brk_idx < k:
                brk_idx += 1
                curr_break = breaks[brk_idx]

    print("Precision of the phi values: " + str(abs(1-sum(phi))))

    return multinomial(omega,phi)            

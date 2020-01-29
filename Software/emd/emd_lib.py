from treeswift import *
import numpy as np
from math import exp,log, sqrt
from scipy.optimize import minimize, LinearConstraint,Bounds
from scipy.stats import poisson, expon,lognorm
from os.path import basename, dirname, splitext,realpath,join,normpath,isdir,isfile,exists
from subprocess import check_output,call
from tempfile import mkdtemp
from shutil import copyfile, rmtree
from os import remove
from emd.util import bitset_from_tree, bitset_index
from scipy.sparse import diags
from scipy.sparse import csr_matrix
import cvxpy as cp

EPS_tau=1e-10

lsd_exec=normpath(join(dirname(realpath(__file__)),"../lsd-0.2/src/lsd")) # temporary solution

def EM_date(tree,smpl_times,root_age=None,refTreeFile=None,s=1000,k=100,df=1e-2,maxIter=500,eps_tau=EPS_tau):
    M, dt, x = setup_constr(tree,smpl_times,s,root_age=root_age,eps_tau=eps_tau)
    tau, phi, omega = init_EM(tree,smpl_times,k,s=s,refTreeFile=refTreeFile)
    
    print("Initialized EM")
    pre_llh = f_ll(x,s,tau,omega,phi)
    print("Initial likelihood: " + str(pre_llh))

    for i in range(1,maxIter+1):
        #for (o,p) in zip(omega,phi):
        #    print(o,p)

        print("EM iteration " + str(i))
        print("Estep ...")
        Q = run_Estep(x,s,omega,tau,phi)
        print("Mstep ...")
        phi,tau = run_Mstep(x,s,omega,tau,phi,Q,M,dt,eps_tau=eps_tau,fixed_phi=True)
        llh = f_ll(x,s,tau,omega,phi)
        #llh = elbo(tau,phi,omega,Q,b,s)
        print("Current llh: " + str(llh))
        curr_df = None if pre_llh is None else llh - pre_llh
        print("Current df: " + str(curr_df))
        if curr_df is not None and curr_df < df:
            break
        pre_llh = llh    

    # convert branch length to time unit
    for node in tree.traverse_postorder():
        if not node.is_root():
            node.set_edge_length(tau[node.idx])

    return tau,omega,phi

def init_EM(tree,sampling_time,k,s=1000,refTreeFile=None,eps_tau=EPS_tau):
    if refTreeFile is None:
        mu,tau = run_lsd(tree,sampling_time,s=s,eps_tau=eps_tau)
    else:
        tau = init_tau_from_refTree(tree,refTreeFile,eps_tau=eps_tau)

    omega,phi = discrete_exponential(0.006,k)
    #omega,phi = discretize(mu,k)
    #omega,phi = discretize_uniform(k)
    #omega,phi = discrete_lognorm(0.006,0.4,k)

    return tau,phi,omega

def discretize_uniform(k,Min=0.0005,Max=0.02):
    delta = (Max-Min)/(k-1)
    omega = [ Min + delta*i for i in range(k) ]
    phi = np.zeros(len(omega)) + 1/(len(omega))

    return omega,phi

def discretize(mu,k,Min=0.0005,Max=0.02):
    Min = mu/10 if Min is None else Min
    Max = mu*10 if Max is None else Max

    delta = (Max-Min)/(k-1)

    omega = [ Min + delta*i for i in range(k) ]
    
    density = np.zeros(len(omega)) + 1.0/100/(len(omega)-1)
    #density = np.zeros(len(omega))
    
    diff = abs(omega[0] - mu)
    best_idx = 0

    for i,y in enumerate(omega):
        d = abs(y-mu)
        if d < diff:
            diff = d
            best_idx = i        
    
    print(mu,omega[best_idx])
    density[best_idx] = 99.0/100

    phi = density/sum(density)

    return omega,phi

def discrete_lognorm(mu,sd,k):
    # scipy reference https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm
    p = [i/(k+1) for i in range(1,k+1)] 
    sigma = sqrt(log(sd*sd+1))
    scale = 1/sqrt(sd*sd+1)
    nu = lognorm.ppf(p,sigma,0,scale)
    density = lognorm.pdf(nu,sigma,0,scale)
    omega = mu*nu
    phi = density/sum(density)
    
    return omega,phi 

def discrete_exponential(mu,k):
    p = [i/(k+1) for i in range(1,k+1)] 
    omega = expon.ppf(p,scale=mu)
    density = expon.pdf(omega,scale=mu)
    phi = density/sum(density)
    
    return omega,phi 

def get_tree_bitsets(tree):
    BS = bitset_from_tree(tree)
    bitset_index(tree,BS)
    bits2idx = {}
    for node in tree.traverse_postorder():
        bits2idx[node.bits] = node.idx    
    return BS,bits2idx


def run_lsd(tree,sampling_time,s=1000,eps_tau=EPS_tau):
    '''
    BS = bitset_from_tree(tree)
    bitset_index(tree,BS)
    bits2idx = {}
    for node in tree.traverse_postorder():
        bits2idx[node.bits] = node.idx    
    '''
    #BS, bits2idx = get_tree_bitsets(tree)
    wdir = mkdtemp()
    treeFile = normpath(join(wdir,"mytree.tre"))
    tree.write_tree_newick(treeFile)
    
    stFile = normpath(join(wdir,"sampling_time.txt"))
    with open(stFile,"w") as fout:
        fout.write(str(len(sampling_time))+"\n")
        for x in sampling_time:
            fout.write(x + " " + str(sampling_time[x]) + "\n") 

    call([lsd_exec,"-i",treeFile,"-d",stFile,"-v","-c","-s",str(s)])

    # suppose LSD was run on the "mytree.newick" and all the outputs are placed inside wdir
    log_file = normpath(join(wdir, "mytree.tre.result")) 
    result_tree_file = normpath(join(wdir, "mytree.tre.result.date.newick")) 

    # getting mu
    s = open(log_file,'r').read()
    i = s.find("Tree 1 rate ") + 12
    mu = ""
    found_dot = False

    # reading mu
    while (s[i] == '.' and not found_dot) or  (s[i] in [str(x) for x in range(10)]):
        mu += s[i]
        if s[i] == '.':
            found_dot = True
        i += 1
    
    mu = float(mu)

    # getting tau
    #tau = init_tau_from_refTree(BS,bits2idx,result_tree_file,eps_tau=eps_tau)
    tau = init_tau_from_refTree(tree,result_tree_file,eps_tau=eps_tau)
    '''tree = read_tree_newick(result_tree_file)
    bitset_index(tree,BS)
    n = len(list(tree.traverse_leaves()))
    N = 2*n-2
    tau = np.zeros(N)
    
    for node in tree.traverse_postorder():
        if not node.is_root():
            tau[bits2idx[node.bits]] = max(node.edge_length,eps_tau) '''
    
    return mu,tau

def init_tau_from_refTree(my_tree,ref_tree_file,eps_tau=EPS_tau):
    BS, bits2idx = get_tree_bitsets(my_tree)
    refTree = read_tree_newick(ref_tree_file)
    bitset_index(refTree,BS)
    n = len(list(refTree.traverse_leaves()))
    N = 2*n-2
    tau = np.zeros(N)
    
    for node in refTree.traverse_postorder():
        if not node.is_root():
            tau[bits2idx[node.bits]] = max(node.edge_length,eps_tau)

    return tau        

def setup_constr(tree,smpl_times,s,root_age=None,eps_tau=EPS_tau):
    n = len(list(tree.traverse_leaves()))
    N = 2*n-2

    M = []
    dt = []
    
    idx = 0
    x = np.zeros(N)

    for node in tree.traverse_postorder():
        node.idx = idx
        idx += 1
        if node.is_leaf():
            node.constraint = np.zeros(N)
            node.constraint[node.idx] = 1
            node.t = smpl_times[node.get_label()]
            x[node.idx] = int(node.edge_length*s)
        else:
            children = node.child_nodes()      
            m = children[0].constraint - children[1].constraint
            dt_i = children[0].t - children[1].t
            M.append(m)
            dt.append(dt_i)

            if not node.is_root(): 
                node.constraint = children[0].constraint
                node.constraint[node.idx] = 1
                node.t = children[0].t
                x[node.idx] = int(node.edge_length*s)
            elif root_age is not None:
                m = children[0].constraint
                dt_i = children[0].t - root_age
                M.append(m) 
                dt.append(dt_i)  
    
    #bounds = Bounds(eps_tau,np.inf)
    #bounds = LinearConstraint(np.eye(N),eps_tau,np.inf,keep_feasible=True)

    return M,dt,x

def run_Estep(x,s,omega,tau,phi,p_eps=EPS_tau):
    N = len(x)
    k = len(omega)
   
    Q = np.zeros((N,k)) 

    for i in range(N): 
        x_i = x[i]
        tau_i = tau[i]
        q_i = np.zeros(k)

        for j in range(k):
            omega_j = omega[j]
            phi_j = phi[j]
            q_i[j] = max(p_eps,poisson.pmf(x_i,s*omega_j*tau_i)*phi_j)
            #q_i[j] = (omega_j**x_i)*exp(-s*omega_j*tau_i)*phi_j
        
        Q[i] = q_i/sum(q_i)
        
    return np.matrix(Q)

def run_Mstep(x,s,omega,tau,phi,Q,M,dt,eps_tau=EPS_tau,fixed_phi=False):
    phi_star = compute_phi_star(Q) if not fixed_phi else phi
    tau_star = compute_tau_star_cvx(tau,omega,Q,x,s,M,dt,eps_tau=EPS_tau)

    return phi_star, tau_star
    
def elbo(tau,phi,omega,Q,x,s):
    Qt = Q.transpose()
    return sum(x*np.log(tau) - s*(np.array(omega*Qt)*tau)[0] - ((np.array(np.log(phi))*Qt))[0])

def f_ll(x,s,tau,omega,phi):
    ll = 0
    for i in range(len(x)):
        ll_i = 0
        tau_i = tau[i]
        x_i = x[i]
        for j in range(len(omega)):
            omega_j = omega[j]
            phi_j = phi[j]
            ll_i += poisson.pmf(x_i,s*omega_j*tau_i)*phi_j
        ll += log(ll_i)    

    return ll

def compute_phi_star(Q):
    return np.array(np.mean(Q,axis=0,dtype=float))[0]

def compute_tau_star_cvx(tau,omega,Q,x,s,M,dt,eps_tau=EPS_tau):
    Qt = Q.transpose()
    v2 = s*np.array(omega*Qt)
    param_1 = cp.Parameter((1,len(x)),nonneg=True,value=np.reshape(np.array(x),(1,len(x))))
    param_2 = cp.Parameter(v2.shape,nonneg=True,value=v2)
    N = len(tau)
    var_tau = cp.Variable(N)
       
    objective = cp.Minimize(-param_1*cp.log(var_tau)+param_2*var_tau)
    constraints = [np.zeros(N)+eps_tau <= var_tau, csr_matrix(M)*var_tau == np.array(dt)]

    prob = cp.Problem(objective,constraints)
    f_star = prob.solve(verbose=True)
    tau_star = var_tau.value

    return tau_star

def compute_tau_star(tau,omega,Q,x,s,M,dt,eps_tau=EPS_tau):
    def f(tau,*args):
        omega,Q,x = args
        Qt = Q.transpose()
        term1 = x*np.log(np.abs(tau))
        term2 = s*np.array(omega*Qt)*tau
        return sum(-term1+term2[0])/1000
   
    def g(tau,*args):
        omega,Q,x = args
        Qt = Q.transpose()
                
        return ((-x/tau + s*np.array(omega*Qt))[0])/1000

    def h(tau,*args):
        omega,Q,x = args
        return diags(x/(tau*tau))/1000
   
     
    bounds = Bounds(eps_tau,np.inf)
    linear_constraint = LinearConstraint(csr_matrix(M),dt,dt)
    opt = minimize(fun=f,method="trust-constr",x0=tau,bounds=bounds,args=(omega,Q,x),constraints=[linear_constraint],jac=g,hess=h,options={'verbose':3})
    #opt = minimize(fun=f,method="SLSQP",x0=tau,args=(omega,Q,x),bounds=bounds,constraints=[linear_constraint],options={'disp':True,'maxiter':1000},jac=g)
    tau_star = np.array(opt.x)
    return tau_star 

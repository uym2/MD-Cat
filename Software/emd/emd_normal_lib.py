from treeswift import *
import numpy as np
from math import exp,log, sqrt, pi
from scipy.optimize import minimize, LinearConstraint,Bounds
from scipy.stats import poisson, expon,lognorm,norm
from os.path import basename, dirname, splitext,realpath,join,normpath,isdir,isfile,exists
from subprocess import check_output,call
from tempfile import mkdtemp
from shutil import copyfile, rmtree
from os import remove
from emd.util import bitset_from_tree, bitset_index
from scipy.sparse import diags
from scipy.sparse import csr_matrix
import cvxpy as cp
#from emd.quadprog_solvers import *
from random import uniform, random

EPS_tau=1e-3
EPSILON=1e-4
INF = float("inf")
PSEUDO = 1.0
MIN_b=0
MIN_ll = -700 # minimum log-likelihood; due to overflow/underflow issue

#lsd_exec=normpath(join(dirname(realpath(__file__)),"../lsd-0.2/src/lsd")) # temporary solution
lsd_exec="/calab_data/mirarab/home/umai/my_gits/EM_Date/Software/lsd-0.2/bin/lsd.exe"

def EM_date_adapt_bins(tree,smpl_times,init_mu,nbins=130,s=10000,maxIter=100,nrep=100):
    k = 1
    omega = [init_mu]
    o_min = init_mu/2
    o_max = init_mu*2
    refTree = None
    while k < nbins:
        print("Solving new EM-Date with " + str(k) + " bins")
        new_tree = read_tree_newick(tree.newick())
        best_tree,best_llh,best_phi,best_omega = EM_date_random_init(new_tree,smpl_times,s=s,input_omega=omega,maxIter=maxIter,refTree=refTree,nrep=nrep)
        refTree = best_tree
        if k == 1:
            new_omega = [o_min,init_mu,o_max]
        else:
            new_omega = [o_min]    
            for o in best_omega[1:]:
                new_omega.append(exp((log(new_omega[-1])+log(o))/2))
                new_omega.append(o)
        omega = new_omega        
        k = len(omega)
    return best_tree,best_llh,best_phi,best_omega

def EM_date_random_init(tree,smpl_times,input_omega=None,init_rate_distr=None,s=1000,k=100,nrep=100,maxIter=100,refTree=None,fixed_phi=False,fixed_tau=False,verbose=False,extra_data={},do_sca=True):
    best_llh = -float("inf")
    best_tree = None
    best_phi = None
    best_omega = None
    for r in range(nrep):
        print("Solving EM with init point + " + str(r+1))
        new_tree = read_tree_newick(tree.newick())
        #try:
        tau,omega,phi,llh = EM_date(new_tree,smpl_times,s=s,input_omega=input_omega,init_rate_distr=init_rate_distr,maxIter=maxIter,refTree=refTree,fixed_phi=fixed_phi,fixed_tau=fixed_tau,verbose=verbose,extra_data=extra_data,do_sca=do_sca)
        print("New llh: " + str(llh))
        print(new_tree.newick())  
        if llh > best_llh:
            best_llh = llh  
            best_tree = new_tree
            best_phi = phi
            best_omega = omega
        #except:
        #    print("Failed to optimize using this init point!")        
    return best_tree,best_llh,best_phi,best_omega        

def EM_date(tree,smpl_times,root_age=None,refTree=None,trueTreeFile=None,s=1000,k=100,input_omega=None,df=0.01,maxIter=100,eps_tau=EPS_tau,fixed_phi=False,fixed_tau=False,init_rate_distr=None,verbose=False,extra_data={},do_sca=True):
    M, dt, b = setup_constr(tree,smpl_times,s,root_age=root_age,eps_tau=eps_tau,trueTreeFile=trueTreeFile,extra_data=extra_data)
    b_avg = [sum(b_i)/len(b_i) for b_i in b] 
    b_sq = [sum(x*x for x in b_i)/len(b_i) for b_i in b] 
    tau, phi, omega = init_EM(tree,smpl_times,k=k,input_omega=input_omega,s=s,refTree=refTree,init_rate_distr=init_rate_distr)
    if verbose:
        print("Initialized EM")
    pre_llh = f_ll(b,s,tau,omega,phi)
    if verbose:
        print("Initial likelihood: " + str(pre_llh))
    #Q = [[1.0/k]*k for i in range(len(tau))]
    for i in range(1,maxIter+1):
        if verbose:
            print("EM iteration " + str(i))
            print("Estep ...")
        #Q = run_Estep_naive(b,s,omega,tau,phi)
        Q = run_Estep(b,s,omega,tau,phi)
        if verbose:
            print("Mstep ...")   
        if fixed_phi: # in the new method, if phi is fixed, then omega must be optimized and vice versa
            next_phi,next_tau,next_omega = run_MMstep(b,b_avg,b_sq,s,omega,tau,phi,Q,M,dt,eps_tau=eps_tau,fixed_phi=fixed_phi,fixed_tau=fixed_tau,do_sca=do_sca)
        else:
            next_phi,next_tau,next_omega = run_MMstep(b,b_avg,b_sq,s,omega,tau,phi,Q,M,dt,eps_tau=eps_tau,fixed_phi=fixed_phi,fixed_tau=fixed_tau,do_sca=do_sca)
        llh = f_ll(b,s,next_tau,next_omega,next_phi)
        #llh = elbo(tau,phi,omega,Q,b,s)
        if verbose:
            print("Current llh: " + str(llh))
        curr_df = None if pre_llh is None else llh - pre_llh
        if verbose:
            print("Current df: " + str(curr_df))
        #if curr_df is not None and curr_df < df:
        #    break
        phi = next_phi
        tau = next_tau    
        omega = next_omega
        pre_llh = llh    
        #Q = run_Estep(b,s,omega,tau,phi)

    # convert branch length to time unit and compute mu for each branch
    for node in tree.traverse_postorder():
        if not node.is_root():
            node.set_edge_length(tau[node.idx])
            node.mu = sum(o*p for (o,p) in zip(omega,Q[node.idx]))
        #else:
        #    node.mu = sum(o*p for (o,p) in zip(omega,phi))

    # compute divergence times
    compute_divergence_time(tree,smpl_times)

    return tau,omega,phi,llh

def compute_divergence_time(tree,sampling_time,bw_time=False,as_date=False):
# compute and place the divergence time onto the node label of the tree
# must have at least one sampling time. Assumming the tree branches have been
# converted to time unit and are consistent with the given sampling_time
    calibrated = []
    for node in tree.traverse_postorder():
        node.time,node.mutation_rate = None,None
        lb = node.get_label()
        if lb in sampling_time:
            node.time = sampling_time[lb]
            calibrated.append(node)

    stk = []
    # push to stk all the uncalibrated nodes that are linked to (i.e. is parent or child of) any node in the calibrated list
    for node in calibrated:
        p = node.get_parent()
        if p is not None and p.time is None:
            stk.append(p)
        if not node.is_leaf():
            stk += [ c for c in node.child_nodes() if c.time is None ]            
    
    # compute divergence time of the remaining nodes
    while stk:
        node = stk.pop()
        lb = node.get_label()
        p = node.get_parent()
        t = None
        if p is not None:
            if p.time is not None:
                t = p.time + node.get_edge_length()
            else:
                stk.append(p)    
        for c in node.child_nodes():
            if c.time is not None:
                t1 = c.time - c.get_edge_length()
                t = t1 if t is None else t
                if abs(t-t1) > EPSILON:
                    print("Inconsistent divergence time computed for node " + lb + ". Violate by " + str(abs(t-t1)))
                #assert abs(t-t1) < EPSILON_t, "Inconsistent divergence time computed for node " + lb
            else:
                stk.append(c)
        node.time = t

    # place the divergence time and mutation rate onto the label
    for node in tree.traverse_postorder():
        #if node.is_leaf():
        #    continue
        lb = node.get_label()
        assert node.time is not None, "Failed to compute divergence time for node " + lb
        if as_date:
            divTime = days_to_date(node.time)
        else:
            divTime = str(node.time) if not bw_time else str(-node.time)
        tag = "[t=" + divTime + ",mu=" + str(node.mu) + "]" if not node.is_root() else "[t=" + divTime + "]"
        lb = lb + tag if lb else tag
        node.set_label(lb)

def init_EM(tree,sampling_time,k=100,s=1000,input_omega=None,refTree=None,eps_tau=EPS_tau,init_rate_distr=None):
    if init_rate_distr:
        omega = init_rate_distr.omega
        phi = init_rate_distr.phi
    elif input_omega:
        omega = input_omega
        phi = [random() for p in range(len(input_omega))]  
        sp = sum(phi)
        phi = [p/sp for p in phi] 
    else:    
        #omega,phi = discrete_lognorm(0.006,0.4,k)
        omega,phi = discrete_exponential(0.006,k)
    
    if refTree is None:
        #mu,tau = run_lsd(tree,sampling_time,s=s,eps_tau=eps_tau)
        N = len(list(tree.traverse_preorder()))-1
        tau = [0]*N
        for node in tree.traverse_preorder():
            if not node.is_root():
                b = node.get_edge_length()
                tmin = b/omega[0]
                tmax = b/omega[-1]
                tau[node.idx] = uniform(tmin,tmax)
    else:
        tau = init_tau_from_refTree(tree,refTree,eps_tau=eps_tau)
    
    #omega = [0.001,0.01]
    #phi = [0.5,0.5]

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
    #density = lognorm.pdf(nu,sigma,0,scale)
    omega = mu*nu
    #phi = density/sum(density)
    phi = [1.0/k]*k

    return omega,phi 

def discrete_exponential(mu,k):
    p = [i/(k+1) for i in range(1,k+1)] 
    omega = expon.ppf(p,scale=mu)
    #density = expon.pdf(omega,scale=mu)
    #phi = density/sum(density)
    phi = [1.0/k]*k

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

def init_tau_from_refTree(my_tree,refTree,eps_tau=EPS_tau):
    BS, bits2idx = get_tree_bitsets(my_tree)
    #refTree = read_tree_newick(ref_tree_file)
    bitset_index(refTree,BS)
    n = len(list(refTree.traverse_leaves()))
    N = 2*n-2
    #tau = np.zeros(N)
    tau = [0]*N
    
    for node in refTree.traverse_postorder():
        if not node.is_root():
            tau[bits2idx[node.bits]] = max(node.edge_length,eps_tau)

    return tau        

def setup_constr(tree,smpl_times,s,root_age=None,eps_tau=EPS_tau,trueTreeFile=None,extra_data={}):
    n = len(list(tree.traverse_leaves()))
    N = 2*n-2

    M = []
    dt = []
    
    idx = 0
    b = [[0]]*N
    lb2idx = {}

    for node in tree.traverse_postorder():
        node.idx = idx
        idx += 1
        if node.is_leaf():
            node.constraint = [0.]*N
            node.constraint[node.idx] = 1
            node.t = smpl_times[node.get_label()]
            b[node.idx] = [node.edge_length]
            name = node.get_label()
            if name in extra_data:
                b[node.idx] += extra_data[name]
            lb2idx[name] = node.idx
        else:
            children = node.child_nodes()      
            m = [x-y for (x,y) in zip(children[0].constraint,children[1].constraint)]
            dt_i = children[0].t - children[1].t
            M.append(m)
            dt.append(dt_i)

            if not node.is_root(): 
                node.constraint = children[0].constraint
                node.constraint[node.idx] = 1
                node.t = children[0].t
                b[node.idx] = [node.edge_length]
                name = node.get_label()
                if name in extra_data:
                    b[node.idx] += extra_data[name]
                lb2idx[name] = node.idx
            elif root_age is not None:
                m = children[0].constraint
                dt_i = children[0].t - root_age
                M.append(m) 
                dt.append(dt_i)  

    if trueTreeFile is not None:
        trueTree = read_tree_newick(trueTreeFile) 
        for node in trueTree.traverse_postorder():
            lb = node.get_label()
            if lb in lb2idx:
                idx = lb2idx[lb]

    return M,dt,b

def log_sum_exp(numlist):
    # using log-trick to compute log(sum(exp(x) for x in numlist))
    # mitigate the problem of underflow
    maxx = max(numlist)
    try:
        minx = min([x for x in numlist if x-maxx > MIN_ll])
    except:
        return log(len(numlist)) + MIN_ll
    #print(min(numlist),max(numlist)) #,exp(max(numlist)-min(numlist)))
    s = sum(exp(x-minx) for x in numlist if x-maxx > MIN_ll)
    result = minx + log(s)  if s > 0 else log(len(numlist)) + MIN_ll
    return result

def run_Estep(b,s,omega,tau,phi,p_eps=EPS_tau):
    N = len(b)
    k = len(omega)
    Q = []

    for b_i,tau_i in zip(b,tau): 
        if b_i is None:
            Q.append(None)
            continue
        lq_i = [0]*k
        for j,(omega_j,phi_j) in enumerate(zip(omega,phi)):
            for x in b_i:
                lq_i[j] += (-(x-omega_j*tau_i)**2*s/2/omega_j/tau_i + log(phi_j) - (log(omega_j)+log(tau_i)-log(s))/2)
        s_lqi = log_sum_exp(lq_i)
        q_i = [exp(x-s_lqi) for x in lq_i]
        s_qi = sum(q_i)
        #Q.append([q/s_qi for q in q_i])
        if s_qi < 1e-10:
            q_i = [1.0/k]*k
        else:
            q_i = [x/s_qi for x in q_i]
        Q.append(q_i)
    return Q

def run_Estep_naive(b,s,omega,tau,phi,p_eps=EPS_tau):
    N = len(b)
    k = len(omega)
    Q = []

    for b_i,tau_i in zip(b,tau): 
        if b_i is None:
            Q.append(None)
            continue
        q_i = [0]*k
        for j,(omega_j,phi_j) in enumerate(zip(omega,phi)):
            var_ij = omega_j*tau_i/s
            q_i[j] = norm.pdf(b_i,omega_j*tau_i,sqrt(var_ij))*phi_j
        s_qi = sum(q_i)
        Q.append([q_ij/s_qi for q_ij in q_i])
    return Q

def run_Mstep(b,s,omega,tau,phi,Q,M,dt,eps_tau=EPS_tau,fixed_phi=False,fixed_tau=False):
    #phi_star = compute_phi_star(Q) if not fixed_phi else phi
    phi_star = compute_phi_star_cvxpy(Q) if not fixed_phi else phi
    tau_star = compute_tau_star_cvxpy(tau,omega,Q,b,s,M,dt,eps_tau=EPS_tau) if not fixed_tau else tau
    #tau_star = compute_tau_star_scipy(tau,omega,Q,b,s,M,dt,eps_tau=EPS_tau) if not fixed_tau else tau
    
    return phi_star, tau_star, omega

def run_MMstep(b,b_avg,b_sq,s,omega,tau,phi,Q,M,dt,eps_tau=EPS_tau,fixed_phi=False,fixed_tau=False,do_sca=True):
    phi_star = compute_phi_star(Q) if not fixed_phi else phi
    tau = compute_tau_star_cvxpy(tau,omega,Q,b_avg,s,M,dt,eps_tau=EPS_tau) if not fixed_tau else tau
    if do_sca:
        tau = compute_tau_star_sca(tau,omega,Q,b_sq,b,s,M,dt,eps_tau=EPS_tau) if not fixed_tau else tau
    for i in range(100):
        if do_sca:
            omega_star = compute_omega_star_sca(tau,omega,Q,b_sq,b,s)
            tau_star = compute_tau_star_sca(tau,omega_star,Q,b_sq,b,s,M,dt,eps_tau=EPS_tau) if not fixed_tau else tau
        else:
            omega_star = compute_omega_star_cvxpy(tau,omega,Q,b_avg)
            tau_star = compute_tau_star_cvxpy(tau,omega_star,Q,b_avg,s,M,dt,eps_tau=EPS_tau) if not fixed_tau else tau
        if sqrt(sum([(x-y)**2 for (x,y) in zip(omega,omega_star)])/len(omega)) < 1e-5:
            break
        if sqrt(sum([(x-y)**2 for (x,y) in zip(tau,tau_star)])/len(tau)) < 1e-5:
            break
        omega = omega_star
        tau = tau_star    
    return phi_star, tau_star, omega_star
    
def f_ll(b,s,tau,omega,phi):
    ll = 0
    k = len(phi)
    for (tau_i,b_i) in zip(tau,b):
        if b_i is None:
            continue
        ll_i = [0]*k
        for j,(omega_j,phi_j) in enumerate(zip(omega,phi)):
            var_ij = tau_i*omega_j/s
            for x in b_i:
                ll_i[j] += (-log(sqrt(2*pi))-(log(omega_j)+log(tau_i)-log(s))/2-(x-tau_i*omega_j)**2/2/var_ij + log(phi_j))
        result = log_sum_exp(ll_i)
        ll += result
    return ll

def f_ll_naive(b,s,tau,omega,phi):
    ll = 0
    for b_i,tau_i in zip(b,tau): 
        lli = 0
        for j,(omega_j,phi_j) in enumerate(zip(omega,phi)):
            var_ij = omega_j*tau_i/s
            lli += norm.pdf(b_i,omega_j*tau_i,sqrt(var_ij))*phi_j
        ll += log(lli)
    return ll

def compute_phi_star(Q):
    k = len(Q[0])
    phi = [0]*k
    N = 0
    for Qi in Q:
        if Qi is not None:
            phi = [x+y for (x,y) in zip(phi,Qi)]
            N += 1
    return [x/N for x in phi]    

def compute_phi_star_cvxpy(Q,gamma=10):
    k = len(Q[0])
    S = [0]*k
    for Qi in Q:
        S = [x+y for (x,y) in zip(S,Qi)]

    var_phi = cp.Variable(k)
       
    objective = cp.Maximize(np.array(S).T @ cp.log(var_phi) + gamma*cp.sum(cp.entr(var_phi)) ) 
    #objective = cp.Maximize( cp.sum(cp.entr(var_phi)))
    constraints = [ (np.zeros(k)+1).T @ var_phi == 1, var_phi >= 0 ]

    prob = cp.Problem(objective,constraints)
    f_star = prob.solve(verbose=False)
    phi_star = var_phi.value

    return phi_star

def compute_tau_star_scipy(tau,omega,Q,b,s,M,dt,eps_tau=EPS_tau):
    N = len(b)
    k = len(omega)
    
    def f(x,*args):
    # objective function
        Q,omega,b = args
        return sum(Q[i][j]*s/omega[j]/x[i]*(b[i]-omega[j]*x[i])**2 + Q[i][j]*log(omega[j]*x[i]) for i in range(N) for j in range(k))
    
    def g(x,*args):
    # Jacobian
        Q,omega,b = args
        return np.array([sum(q_i[j]*s*(omega[j]-b_i**2/omega[j]/tau_i**2) + q_i[j]/tau_i for j in range(k)) for (q_i,tau_i,b_i) in zip(Q,x,b)])
    
    def h(x,*args):
    # Hessian
        Q,omega,b = args
        return diags([sum(2*q_i[j]*s*b_i**2/omega[j]/tau_i**3 - q_i[j]/tau_i**2 for j in range(k)) for (q_i,tau_i,b_i) in zip(Q,x,b)])
    
    linear_constraint = LinearConstraint(csr_matrix(M),dt,dt,keep_feasible=False)
    bounds = Bounds(np.zeros(N)+eps_tau,np.zeros(N)+1e5,keep_feasible=False)
    args = (Q,omega,b)

    result = minimize(fun=f,method="trust-constr",x0=tau,args=args,bounds=bounds,constraints=[linear_constraint],options={'disp':True,'verbose':1},jac=g,hess=h)
    tau_star = result.x

    return tau_star

def compute_tau_star_cvxpy(tau,omega,Q,b,s,M,dt,eps_tau=EPS_tau):
    N = len(b)
    k = len(omega)
    Pd = np.zeros(N)
    q = np.zeros(N)

    for i in range(N):
        if b[i] is None:
            continue
        for j in range(k):
            w_ij = omega[j]*tau[i] # weight by the variance multiplied with s; use previous tau to estimate
            #for x in b[i]:
            Pd[i] += Q[i][j]*omega[j]**2/w_ij
            q[i] -= 2*b[i]*Q[i][j]*omega[j]/w_ij
          
    P = np.diag(Pd)        
    var_tau = cp.Variable(N)
       
    objective = cp.Minimize(cp.quad_form(var_tau,P) + q.T @ var_tau)
    upper_bound = np.array([float("inf") if b_i is not None else 1.0/6 for b_i in b])
    constraints = [np.zeros(N)+eps_tau <= var_tau, csr_matrix(M)@var_tau == np.array(dt)]

    prob = cp.Problem(objective,constraints)
    f_star = prob.solve(verbose=False,max_iter=100000)#,solver=cp.MOSEK)
    tau_star = var_tau.value

    return tau_star

def compute_omega_star_cvxpy(tau,omega,Q,b,eps_omg=0.0001):
    N = len(b)
    k = len(omega)
    Pd = np.zeros(k)
    q = np.zeros(k)

    for j in range(k):
        for i in range(N):
            w_ij = omega[j]*tau[i] # weight by the variance multiplied with s; use previous omega to estimate
            #for x in b[i]:
            Pd[j] += Q[i][j]*tau[i]**2/w_ij
            q[j] -= 2*b[i]*Q[i][j]*tau[i]/w_ij
          
    P = np.diag(Pd)        
    var_omega = cp.Variable(k)
       
    objective = cp.Minimize(cp.quad_form(var_omega,P) + q.T @ var_omega)
    constraints = [np.zeros(k)+eps_omg <= var_omega]

    prob = cp.Problem(objective,constraints)
    f_star = prob.solve(verbose=False)
    omega_star = var_omega.value

    return omega_star

def compute_f_MM(tau,omega,Q,b,s):
    N = len(tau)
    k = len(omega)
    m = len(b[0])

    return sum(s*Q[i][j]*(b[i][l]-omega[j]*tau[i])**2/omega[j]/tau[i] + Q[i][j]*log(omega[j]*tau[i]) for i in range(N) for j in range(k) for l in range(m))

def compute_tau_star_sca(tau,omega,Q,b_sq,b,s,M,dt,eps_tau=EPS_tau):
    N = len(b_sq)
    k = len(omega)

    O = [sum(Q[i][j]/omega[j] for j in range(k)) for i in range(N)]

   
    tau0 = [x for x in tau]
    f_star_pre = compute_f_MM(tau0,omega,Q,b,s)
    for i in range(100):
        print("tau",i,f_star_pre)
        var_tau = cp.Variable(N,pos=True)
        tau_inv = cp.vstack(cp.inv_pos(t) for t in var_tau)
        constraints = [np.zeros(N)+eps_tau <= var_tau, csr_matrix(M)@var_tau == np.array(dt)]
        tau0_inv = cp.vstack(1/t for t in tau0)
           
        objective = cp.Minimize(s*cp.multiply(b_sq,O).T @ tau_inv + 
                                s*var_tau.T @ csr_matrix(Q) @ omega + 
                                tau0_inv.T @ var_tau)

        prob = cp.Problem(objective,constraints)
        prob.solve(verbose=False,solver=cp.MOSEK)
        tau0 = [x for x in var_tau.value]
        f_star = compute_f_MM(tau0,omega,Q,b,s)
        if abs(f_star_pre-f_star) < 1e-3:
            break
        f_star_pre = f_star    
    return tau0

def compute_omega_star_sca(tau,omega,Q,b_sq,b,s,eps_omg=0.0001):
    N = len(b_sq)
    k = len(omega)

    T = cp.vstack(sum(Q[i][j]*b_sq[i]/tau[i] for i in range(N)) for j in range(k))

    omega0 = [x for x in omega]
    f_star_pre = compute_f_MM(tau,omega0,Q,b,s)
    
    for i in range(100):
        print("omega",i,f_star_pre)
        var_omega = cp.Variable(k,pos=True)
        omega_inv = cp.vstack(cp.inv_pos(w) for w in var_omega)
        tau_transpose = cp.vstack(t for t in tau).T
        omega0_inv = cp.multiply(cp.vstack(sum(q[j] for q in Q) for j in range(k)),cp.vstack(1/w for w in omega0))

        objective = cp.Minimize(s*T.T @ omega_inv + 
                                s*tau_transpose @ csr_matrix(Q) @ var_omega +
                                omega0_inv.T @ var_omega)
        constraints = [np.zeros(k)+eps_omg <= var_omega]

        prob = cp.Problem(objective,constraints)
        prob.solve(verbose=False,solver=cp.MOSEK)
        omega0 = [x for x in var_omega.value]
        f_star = compute_f_MM(tau,omega0,Q,b,s)
        if abs(f_star_pre-f_star) < 1e-3:
            break
        f_star_pre = f_star
    return omega0

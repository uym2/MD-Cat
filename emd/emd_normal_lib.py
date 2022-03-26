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
from random import seed,uniform, random, randrange
from simulator.multinomial import *
import time

EPS_tau=1e-3
EPSILON=5e-4
INF = float("inf")
PSEUDO = 1.0
MIN_b=0
MIN_ll = -700 # minimum log-likelihood; due to overflow/underflow issue
MIN_q = 1e-5

def EM_date_random_init(tree,smpl_times,input_omega=None,init_rate_distr=None,s=1000,k=100,nrep=100,maxIter=100,refTree=None,fixed_tau=False,verbose=False,mu_avg=None,fixed_omega=False,randseed=None,pseudo=0):
    best_llh = -float("inf")
    best_tree = None
    best_phi = None
    best_omega = None

    if type(randseed) == list:
        if len(randseed) == nrep:
            rseeds = randseed        
        else:
            print("WARNING: the number of random seeds does not match the number of replicates. The input random seeds are ignored")   
            rseeds = [ randrange(0,10000) for i in range(nrep) ]
    else:
        if type(randseed) == int:
            seed(a=randseed)    
        elif randseed is not None:
            print("WARNING: the input random seeds are invalid and will be ignored")
        rseeds = [ randrange(0,10000) for i in range(nrep) ]
    
    for r in range(nrep):
        seed(a=rseeds[r])
        print("Solving EM with init point + " + str(r+1))
        print("Random seed: " + str(rseeds[r]))
        new_tree = read_tree_newick(tree.newick())
        #try:
        tau,omega,phi,llh = EM_date(new_tree,smpl_times,s=s,input_omega=input_omega,init_rate_distr=init_rate_distr,maxIter=maxIter,refTree=refTree,fixed_tau=fixed_tau,verbose=verbose,mu_avg=mu_avg,fixed_omega=fixed_omega,pseudo=pseudo)
        new_ref = new_tree
        new_tree = read_tree_newick(tree.newick())
        omega_adjusted = [o for o,p in zip(omega,phi) if p > 1e-6]
        phi_adjusted = [p for p in phi if p > 1e-6]
        sum_phi = sum(phi_adjusted)
        phi_adjusted = [p/sum_phi for p in phi_adjusted]
        tau,omega,phi,llh = EM_date(new_tree,smpl_times,s=s,init_rate_distr=multinomial(omega_adjusted,phi_adjusted),maxIter=maxIter,refTree=new_ref,fixed_tau=fixed_tau,verbose=verbose,mu_avg=None,fixed_omega=fixed_omega,pseudo=pseudo) 
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

def EM_date(tree,smpl_times,root_age=None,refTree=None,trueTreeFile=None,s=1000,k=100,input_omega=None,df=5e-4,maxIter=100,eps_tau=EPS_tau,fixed_tau=False,init_rate_distr=None,verbose=False,mu_avg=None,fixed_omega=False,pseudo=0):
    M, dt, b = setup_constr(tree,smpl_times,s,root_age=root_age,eps_tau=eps_tau,trueTreeFile=trueTreeFile,pseudo=pseudo)
    #b_avg = [sum(b_i)/len(b_i) for b_i in b] 
    #b_sq = [sum(x*x for x in b_i)/len(b_i) for b_i in b] 
    tau, phi, omega = init_EM(tree,smpl_times,k=k,input_omega=input_omega,s=s,refTree=refTree,init_rate_distr=init_rate_distr)
    if verbose:
        print("Initialized EM")
    pre_llh = f_ll(b,s,tau,omega,phi,var_apprx=True)
    if verbose:
        print("Initial likelihood: " + str(pre_llh))
    for i in range(1,maxIter+1):
        if verbose:
            print("EM iteration " + str(i))
            print("Estep ...")
        Q = run_Estep(b,s,omega,tau,phi,var_apprx=True)
        if verbose:
            print("Mstep ...")   
        next_phi,next_tau,next_omega = run_MMstep(tree,smpl_times,b,s,omega,tau,phi,Q,M,dt,eps_tau=eps_tau,fixed_phi=True,fixed_tau=fixed_tau,fixed_omega=fixed_omega,mu_avg=mu_avg)
        llh = f_ll(b,s,next_tau,next_omega,next_phi,var_apprx=True)
        if verbose:
            print("Current llh: " + str(llh))
        curr_df = None if pre_llh is None else llh - pre_llh
        if verbose:
            print("Current df: " + str(curr_df))
        if curr_df is not None and abs(curr_df) < df:
            break
        phi = next_phi
        tau = next_tau    
        omega = next_omega
        pre_llh = llh    

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

def compute_divergence_time(tree,sampling_time,bw_time=False,as_date=False,place_mu=True):
# compute and place the divergence time onto the node label of the tree
# must have at least one sampling time. Assumming the tree branches have been
# converted to time unit and are consistent with the given sampling_time
# if place_mu is true, also place the mutation rate of that branch ( assuming
# each node already has the attribute node.mu

    calibrated = []
    for node in tree.traverse_postorder():
        node.time = None
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
        lb = node.get_label()
        assert node.time is not None, "Failed to compute divergence time for node " + lb
        if as_date:
            divTime = days_to_date(node.time)
        else:
            divTime = str(node.time) if not bw_time else str(-node.time)
        if place_mu and not node.is_root():
            tag = "[t=" + divTime + ",mu=" + str(node.mu) + "]"
        else:
            tag = "[t=" + divTime + "]"
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
        omega,phi = discrete_exponential(0.006,k)
    
    if refTree is None:
        N = len(list(tree.traverse_preorder()))-1
        tau = [0]*N
        for node in tree.traverse_preorder():
            if not node.is_root():
                b = node.get_edge_length()
                #tmin = b/omega[0]
                #tmax = b/omega[-1]
                #tau[node.idx] = uniform(tmin,tmax)
                tau[node.idx] = b/omega[randrange(len(omega))]
    else:
        tau = init_tau_from_refTree(tree,refTree,eps_tau=eps_tau)
    
    return tau,phi,omega

def get_tree_bitsets(tree):
    BS = bitset_from_tree(tree)
    bitset_index(tree,BS)
    bits2idx = {}
    for node in tree.traverse_postorder():
        bits2idx[node.bits] = node.idx    
    return BS,bits2idx

def init_tau_from_refTree(my_tree,refTree,eps_tau=EPS_tau):
    BS, bits2idx = get_tree_bitsets(my_tree)
    bitset_index(refTree,BS)
    n = len(list(refTree.traverse_leaves()))
    N = 2*n-2
    tau = [0]*N
    
    for node in refTree.traverse_postorder():
        if not node.is_root():
            tau[bits2idx[node.bits]] = max(node.edge_length,eps_tau)
    return tau        

def setup_constr(tree,smpl_times,s,root_age=None,eps_tau=EPS_tau,trueTreeFile=None,pseudo=0):
    n = len(list(tree.traverse_leaves()))
    N = 2*n-2

    M = []
    dt = []
    
    idx = 0
    b = [0]*N
    lb2idx = {}

    for node in tree.traverse_postorder():
        node.idx = idx
        idx += 1
        if node.is_leaf():
            node.constraint = [0.]*N
            node.constraint[node.idx] = 1
            node.t = smpl_times[node.get_label()]
            b[node.idx] = node.edge_length+pseudo/s
            name = node.get_label()
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
                b[node.idx] = node.edge_length+pseudo/s
                name = node.get_label()
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

def run_Estep(b,s,omega,tau,phi,p_eps=EPS_tau,var_apprx=True):
    N = len(b)
    k = len(omega)
    Q = []

    for b_i,tau_i in zip(b,tau): 
        if b_i is None:
            Q.append(None)
            continue
        lq_i = [0]*k
        for j,(omega_j,phi_j) in enumerate(zip(omega,phi)):
            var_ij = omega_j*tau_i/s if not var_apprx else b_i/s
            lq_i[j] += (-(b_i-omega_j*tau_i)**2/2/var_ij + log(phi_j) - log(var_ij)/2)
        s_lqi = log_sum_exp(lq_i)
        q_i = [exp(x-s_lqi) for x in lq_i]
        q_i = [x if x>MIN_q else 0 for x in q_i]
        s_qi = sum(q_i)
        if s_qi < 1e-10:
            q_i = [1.0/k]*k
        else:
            q_i = [x/s_qi for x in q_i]
        Q.append(q_i)
    return Q

def run_MMstep(tree,smplTimes,b,s,omega,tau,phi,Q,M,dt,eps_tau=EPS_tau,fixed_phi=True,fixed_tau=False,fixed_omega=False,mu_avg=None):
    phi_star = compute_phi_star_cvxpy(Q,omega,mu_avg=mu_avg) if not fixed_phi else phi
    for i in range(100):
        tau_star = compute_tau_star(tree,smplTimes,omega,Q,b,s,M,dt,eps_tau=EPS_tau) if not fixed_tau else tau
        omega_star = compute_omega_star(tau_star,Q,b,phi_star,mu_avg=mu_avg) if not fixed_omega else omega
        if sqrt(sum([(x-y)**2 for (x,y) in zip(omega,omega_star)])/len(omega)) < 1e-5:
            break
        if sqrt(sum([(x-y)**2 for (x,y) in zip(tau,tau_star)])/len(tau)) < 1e-5:
            break
        omega = omega_star
        tau = tau_star    
    return phi_star, tau_star, omega_star
    
def f_ll(b,s,tau,omega,phi,var_apprx=True):
    ll = 0
    k = len(phi)
    for (tau_i,b_i) in zip(tau,b):
        if b_i is None:
            continue
        ll_i = [0]*k
        for j,(omega_j,phi_j) in enumerate(zip(omega,phi)):
            var_ij = tau_i*omega_j/s if not var_apprx else b_i/s
            ll_i[j] += (-log(sqrt(2*pi))-(log(var_ij))/2-(b_i-tau_i*omega_j)**2/2/var_ij + log(phi_j))
        result = log_sum_exp(ll_i)
        ll += result
    return ll

def compute_phi_star(Q,eps_phi=1e-7):
    k = len(Q[0])
    phi = [0]*k
    N = 0
    for Qi in Q:
        if Qi is not None:
            phi = [x+y for (x,y) in zip(phi,Qi)]
            N += 1
    return [max(x/N,eps_phi) for x in phi]    

def compute_tau_star(tree,smplTimes,omega,Q,b,s,M,dt,eps_tau=EPS_tau,maxIter=5000):
# IMPORTANT: only works with var_apprx. Never use this function
# when var_apprx if False
# A: the current active set
    N = len(b)
    k = len(omega)
    alpha = [sum(2*Q[i][j]*omega[j]*omega[j]/b[i] for j in range(k)) for i in range(N)]
    beta = [-sum(2*Q[i][j]*omega[j] for j in range(k)) for i in range(N)]
    
    def __changeVar__(param_yx,param_xt):
        a1,b1 = param_yx # equation: y = a1*x + b1
        a2,b2 = param_xt # equation: x = a2*t + b2
        # compute y by t: y = a1*(a2*t+b2)+b1 = a1*a2*t + a1*b2+b1
        return (a1*a2,a1*b2+b1)

    def __sumParam__(params):
    # params is a list of 2-entry tuples
        return (sum(p[0] for p in params),sum(p[1] for p in params))

    def __compute__(param_yx,x):
        a,b = param_yx
        return a*x+b    
    
    def __solve__(param_yx,y=0):
        a,b = param_yx
        return (y-b)/a

    def __solve_Lagrange__(A):
        lambdaMap = {}
        for u in tree.traverse_postorder():
            if u.is_leaf():
                u.ld = u.label
                u.sld = (1,0) # sld: sum of all lambdas below this node
                u.t = smplTimes[u.label]
                u.as_leaf = True
            else:
                u.as_leaf = False
                v1 = None
                for v in u.children:
                    if v.as_leaf and v.idx in A:
                        v1 = v
                if v1 is None:        
                    v1 = u.children[0]
                else:
                    u.as_leaf = True    
                u.ld = v1.ld
                u.t = v1.t
                g1,e1 = v1.path
                u.sld = (0,0)
                for v2 in u.children:
                    if v2 is v1:
                        g,e = (1,0)
                    else:    
                        g2,e2 = v2.path
                        g = g1/g2
                        e = (e1-e2+v2.t-v1.t)/g2 
                    if v1.ld not in lambdaMap:
                        lambdaMap[v1.ld] = {v2.ld:(g,e)}
                    else:
                        lambdaMap[v1.ld][v2.ld] = (g,e)
                    u.sld = __sumParam__([u.sld,__changeVar__(v2.sld,(g,e))])
            if not u.is_root():
                if u.idx in A:
                    u.tau = (0,eps_tau) # i.e. tau = eps_tau
                    u.nu = __changeVar__((-1/alpha[u.idx],beta[u.idx]/alpha[u.idx]+eps_tau),u.sld)
                else:
                    u.tau = __changeVar__((1/alpha[u.idx],-beta[u.idx]/alpha[u.idx]),u.sld)
                    u.nu = None
                u.path = u.tau if u.is_leaf() else __sumParam__([v1.path,u.tau])
        
        ld0 = tree.root.ld
        if ld0 not in lambdaMap:
            lambdaMap[ld0] = {ld0:(1,0)} # should never happens (unless the tree has unifurcation at the root?)
        else:    
            lambdaMap[ld0][ld0] = (1,0)                            
        for u in tree.traverse_preorder():
            if u.is_leaf():
                continue
            ld1 = u.ld
            for v in u.children:
                ld2 = v.ld
                lambdaMap[ld0][ld2] = __changeVar__(lambdaMap[ld1][ld2],lambdaMap[ld0][ld1])            
        ld0_value = -sum(lambdaMap[ld0][ld1][1] for ld1 in lambdaMap[ld0])/sum(lambdaMap[ld0][ld1][0] for ld1 in lambdaMap[ld0])
        lambdaValues = {ld:__compute__(lambdaMap[ld0][ld],ld0_value) for ld in lambdaMap[ld0]}
        tau = [eps_tau]*N
        nu = {}
        for u in tree.traverse_postorder():
            if not u.is_root():
                ldVal = lambdaValues[u.ld]
                if u.idx in A:
                    nu[u.idx] = __compute__(u.nu,ldVal)
                else:
                    tau[u.idx] = __compute__(u.tau,ldVal)
        return tau,nu        

    def __violated_set__(tau):
        return [i for i in range(N) if tau[i]-eps_tau < -1e-8]
         
    def __make_feasible__(tau):         
        A = set() # active set  
        for u in tree.traverse_postorder():
            if u.is_leaf():
                u.as_leaf = True
                u.t = smplTimes[u.label]
            else:                    
                min_t = float("inf")                    
                for v in u.children:
                    if tau[v.idx] < eps_tau:
                        tau[v.idx] = eps_tau
                    min_t = min(min_t,v.t-tau[v.idx])
                u.t = min_t
                u.as_leaf = False
                minChild = None # the closest child as leaf having tau == eps_tau
                min_t = float("inf")
                for v in u.children:
                    tau[v.idx] = v.t-u.t
                    if abs(tau[v.idx]-eps_tau) < 1e-10:
                        if v.as_leaf:
                            if v.t < min_t:
                                min_t = v.t
                                minChild = v 
                        else:
                            A.add(v.idx)    
                if minChild is not None:
                    A.add(minChild.idx)
                    u.as_leaf = True
        return tau,A
    
    A = set()
    tau = None
    for r in range(40):
        tau_star,nu = __solve_Lagrange__(A)
        V = __violated_set__(tau_star) 
        if len(V) == 0:
            tau = tau_star
            min_nu = float("inf")
            min_idx = None
            for i in nu:
                if nu[i] < min_nu:
                    min_nu = nu[i]
                    min_idx = i
            if min_nu >= 0:
                return tau       
            else:
                A.remove(min_idx)     
        elif tau is None:
            tau,A = __make_feasible__(tau_star)                                
        else:
            flag = False
            for i in V:
                if abs(tau[i]-eps_tau) < 1e-10:
                    A.add(i)
                    flag = True 
            if not flag:
                delta = 0
                minD = float("inf")
                minIdx = None
                for i in V:
                    #if abs(tau[i]-eps_tau) < 1e-10:
                    #    continue
                    delta = (tau_star[i]-eps_tau)/(tau[i]-eps_tau)
                    if delta < minD:
                        minD = delta
                        minIdx = i
                a = 1/(1-minD)
                A.add(minIdx)
                tau = [t+a*(ts-t) for t,ts in zip(tau,tau_star)]       
    return tau            

def compute_omega_star(tau,Q,b,phi,eps_omg=0.0001,mu_avg=None,maxIter=100):
# IMPORTANT: only works with var_apprx. Never call this function
# when var_apprx if False
    N = len(tau)
    k = len(Q[0])
    a = [2*sum(Q[i][j]*tau[i] for i in range(N)) for j in range(k)]
    c = [2*sum(Q[i][j]*tau[i]*tau[i]/b[i] for i in range(N)) for j in range(k)]
    
    def __solve_Lagrange__(A):
        omega_star = [eps_omg]*k
        lambda_star = {}
        if mu_avg is None:
            ld0 = 0
        else:
            u = mu_avg # numerator
            v = 0 # denominator
            for j in range(k):
                if j in A:
                    u -= eps_omg*phi[j]
                else:
                    u -= a[j]/c[j]*phi[j]
                    v += phi[j]*phi[j]/c[j]
            ld0 = u/v
        for j in range(k):
            if j in A:
                lambda_star[j] = eps_omg*c[j]-ld0*phi[j]-a[j]
            else: 
                omega_star[j] = a[j]/c[j]+ld0*phi[j]/c[j]
        return omega_star,lambda_star

    A = set({})
    omega = [mu_avg]*k if mu_avg is not None else None # feasible omega
    for i in range(maxIter):
        omega_star,lambda_star = __solve_Lagrange__(A)
        V = [j for j in range(k) if omega_star[j] < eps_omg]
        if len(V) == 0: # omega_star is feasible
            min_ld = float("inf")
            min_idx = None
            for j in lambda_star:
                if lambda_star[j] < min_ld:
                    min_ld = lambda_star[j]
                    min_idx = j
            if min_ld >= 0:
                return omega_star # feasible and optimal
            else:
                A.remove(min_idx) # remove one constraint in the active set
        elif omega is None:
            omega = [max(omg,eps_omg) for omg in omega_star] # feasible omega
            A = V
        else:    
            flag = False
            for i in V:
                if abs(omega[i]-eps_omg) < 1e-10:
                    A.add(i)
                    flag = True 
            if not flag:
                delta = 0
                minD = float("inf")
                minIdx = None
                for i in V:
                    delta = (omega_star[i]-eps_omg)/(omega[i]-eps_omg)
                    if delta < minD:
                        minD = delta
                        minIdx = i
                alpha = 1/(1-minD)
                A.add(minIdx)
                omega = [m+alpha*(ms-m) for m,ms in zip(omega,omega_star)] # feasible omega       
    return omega                    

def compute_f_MM(tau,omega,Q,b,s,var_apprx=True):
    N = len(tau)
    k = len(omega)
    F = 0
    for i in range(N):
        for j in range(k):
            w_ij = b[i] if var_apprx else omega[j]*tau[i]
            F += s*Q[i][j]*(b[i]-omega[j]*tau[i])**2/w_ij + Q[i][j]*log(w_ij)       
    return F

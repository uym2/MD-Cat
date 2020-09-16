from treeswift import *
import numpy as np
from math import exp,log, sqrt
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
from emd.quadprog_solvers import *

EPS_tau=0
EPSILON=1e-4
INF = float("inf")

lsd_exec=normpath(join(dirname(realpath(__file__)),"../lsd-0.2/src/lsd")) # temporary solution

def EM_date(tree,smpl_times,root_age=None,refTreeFile=None,s=1000,k=100,df=0.01,maxIter=500,eps_tau=EPS_tau,fixed_phi=False,fixed_tau=False,init_rate_distr=None):
    M, dt, b = setup_constr(tree,smpl_times,s,root_age=root_age,eps_tau=eps_tau)
    tau, phi, omega = init_EM(tree,smpl_times,k,s=s,refTreeFile=refTreeFile,init_rate_distr=init_rate_distr)
    
    print("Initialized EM")
    pre_llh = f_ll(b,s,tau,omega,phi)
    print("Initial likelihood: " + str(pre_llh))

    for i in range(1,maxIter+1):
        print("EM iteration " + str(i))
        print("Estep ...")
        Q = run_Estep(b,s,omega,tau,phi)
        print("Mstep ...")
        phi,tau = run_Mstep(b,s,omega,tau,phi,Q,M,dt,eps_tau=eps_tau,fixed_phi=fixed_phi,fixed_tau=fixed_tau)
        llh = f_ll(b,s,tau,omega,phi)
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

    # compute divergence times
    compute_divergence_time(tree,smpl_times)

    return tau,omega,phi

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
                    logger.warning("Inconsistent divergence time computed for node " + lb + ". Violate by " + str(abs(t-t1)))
                #assert abs(t-t1) < EPSILON_t, "Inconsistent divergence time computed for node " + lb
            else:
                stk.append(c)
        node.time = t

    # place the divergence time and mutation rate onto the label
    for node in tree.traverse_postorder():
        if node.is_leaf():
            continue
        lb = node.get_label()
        assert node.time is not None, "Failed to compute divergence time for node " + lb
        if as_date:
            divTime = days_to_date(node.time)
        else:
            divTime = str(node.time) if not bw_time else str(-node.time)
        tag = "[t=" + divTime + "]"
        lb = lb + tag if lb else tag
        node.set_label(lb)

def init_EM(tree,sampling_time,k,s=1000,refTreeFile=None,eps_tau=EPS_tau,init_rate_distr=None):
    if refTreeFile is None:
        mu,tau = run_lsd(tree,sampling_time,s=s,eps_tau=eps_tau)
    else:
        tau = init_tau_from_refTree(tree,refTreeFile,eps_tau=eps_tau)

    #omega,phi = discretize(mu,k)
    #omega,phi = discretize_uniform(k)
    if init_rate_distr:
        omega = init_rate_distr.omega
        phi = init_rate_distr.phi
    else:    
        omega,phi = discrete_lognorm(0.006,0.4,k)
        #omega,phi = discrete_exponential(0.006,k)
    
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
    #tau = np.zeros(N)
    tau = [0]*N
    
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
    #b = np.zeros(N)
    b = [0]*N

    for node in tree.traverse_postorder():
        node.idx = idx
        idx += 1
        if node.is_leaf():
            #node.constraint = np.zeros(N)
            node.constraint = [0.]*N
            node.constraint[node.idx] = 1
            node.t = smpl_times[node.get_label()]
            b[node.idx] = node.edge_length
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
                b[node.idx] = node.edge_length
            elif root_age is not None:
                m = children[0].constraint
                dt_i = children[0].t - root_age
                M.append(m) 
                dt.append(dt_i)  

    return M,dt,b

def log_sum_exp(numlist):
    # using log-trick to compute log(sum(exp(x) for x in numlist))
    # mitigate the problem of underflow
    try:
        minx = min([x for x in numlist if x != -INF])
    except:
        return -INF    
    s = sum(exp(x-minx) for x in numlist)
    return minx + log(s) if s > 0 else -INF

def run_Estep(b,s,omega,tau,phi,p_eps=EPS_tau):
    N = len(b)
    k = len(omega)
   
    #Q = np.zeros((N,k))
    Q = []

    for b_i,tau_i in zip(b,tau): 
        #b_i = b[i]
        #tau_i = tau[i]
        lq_i = [0]*k
        var_i = b_i/s+1.0/s/s
        for j,(omega_j,phi_j) in enumerate(zip(omega,phi)):
            lq_i[j] = -(b_i-omega_j*tau_i)**2/2/var_i + log(phi_j)
            #omega_j = omega[j]
            #phi_j = phi[j]
            #q_i[j] = norm.pdf(b_i,omega_j*tau_i,sigma_i)*phi_j
            #q_i[j] = (omega_j**x_i)*exp(-s*omega_j*tau_i)*phi_j
        
        #Q[i] = q_i/sum(q_i)
        #s_qi = sum(q_i)
        s_lqi = log_sum_exp(lq_i)
        Q.append([exp(x-s_lqi) for x in lq_i])
        #Q.append([x/s_qi for x in q_i])
        
    #return np.matrix(Q)
    return Q

def run_Mstep(b,s,omega,tau,phi,Q,M,dt,eps_tau=EPS_tau,fixed_phi=False,fixed_tau=False):
    phi_star = compute_phi_star(Q) if not fixed_phi else phi
    tau_star = compute_tau_star_cvxpy(tau,omega,Q,b,s,M,dt,eps_tau=EPS_tau) if not fixed_tau else tau

    return phi_star, tau_star
    
def f_ll(b,s,tau,omega,phi):
    ll = 0
    k = len(phi)
    for (tau_i,b_i) in zip(tau,b):
        #ll_i = 0
        ll_i = [0]*k
        #tau_i = tau[i]
        #b_i = b[i]
        #sigma_i = sqrt(b_i/s+1.0/s/s)
        var_i = b_i/s+1.0/s/s
        #for j in range(len(omega)):
        for j,(omega_j,phi_j) in enumerate(zip(omega,phi)):
            #omega_j = omega[j]
            #phi_j = phi[j]
            #ll_i += norm.pdf(b_i,omega_j*tau_i,sigma_i)*phi_j
            ll_i[j] = log(1/sqrt(2*pi*var_i))-(b_i-tau_i*omega_j)**2/2/var_i + log(phi_j)
        #ll += log(ll_i)    
        ll += log_sum_exp(ll_i)

    return ll

def compute_phi_star(Q):
    #return np.array(np.mean(Q,axis=0,dtype=float))[0]
    N = len(Q)
    k = len(Q[0])
    phi = [0]*k
    for Qi in Q:
        phi = [x+y for (x,y) in zip(phi,Qi)]
    return [x/N for x in phi]    

def compute_tau_star_cvxpy(tau,omega,Q,b,s,M,dt,eps_tau=EPS_tau):
    N = len(b)
    k = len(omega)
    Pd = np.zeros(N)
    q = np.zeros(N)

    for i in range(N):
        for j in range(k):
            Pd[i] += Q[i][j]*omega[j]**2
            q[i] -= Q[i][j]*omega[j]
        Pd[i] /= (b[i] + 1.0/s)
        q[i] *= (2*b[i]/(b[i]+1.0/s))
          
    P = diag(Pd)        
    #param_1 = cp.Parameter(q.shape,nonneg=True,value=q)
    #param_2 = cp.Parameter(P.shape,nonneg=True,value=P)
    N = len(tau)
    var_tau = cp.Variable(N)
       
    objective = cp.Minimize(cp.quad_form(var_tau,P) + q.T @ var_tau)
    constraints = [np.zeros(N)+eps_tau <= var_tau, csr_matrix(M)*var_tau == np.array(dt)]

    prob = cp.Problem(objective,constraints)
    f_star = prob.solve(verbose=True)
    tau_star = var_tau.value

    return tau_star

def compute_tau_star_cvxopt(tau,omega,Q,b,s,M,dt,eps_tau=EPS_tau):
    N = len(b)
    k = len(omega)
    Pd = np.zeros(N)
    q = np.zeros(N)

    for i in range(N):
        for j in range(k):
            Pd[i] += Q[i][j]*omega[j]**2
            q[i] -= Q[i][j]*omega[j]
        Pd[i] /= (b[i] + 1.0/s)
        q[i] *= (2*b[i]/(b[i]+1.0/s))
          
    P = diag(Pd)        
    G = -np.identity(N)  
    h = np.zeros(N) + eps_tau
    
    #tau_star = cvxopt_solve_qp(P,q)
    tau_star = quadprog_solve_qp(P,q,G,h,np.matrix(M),dt)
    #tau_star = cvxopt_solve_qp(P,q,G,h,np.matrix(M),dt)
    
    print(np.matmul(np.matmul(np.transpose(tau),P),tau)+np.matmul(np.transpose(q),tau))
    print(np.matmul(np.matmul(np.transpose(tau_star),P),tau_star)+np.matmul(np.transpose(q),tau_star))
    #print(np.matmul(np.matrix(M),tau)-np.array(dt))

    return tau_star 

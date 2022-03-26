from math import log,pi,exp,sqrt
from scipy.stats import norm
from scipy.sparse import diags,csr_matrix
from scipy.optimize import minimize, LinearConstraint,Bounds
import numpy as np
from emd.emd_normal_lib import setup_constr, init_EM,compute_divergence_time 
from simulator.multinomial import multinomial
from treeswift import *

EPS_tau = 1e-3
EPS_phi = 1e-5
MAX_x = 37

def compute_delta_G(omega,tau,b,s):
    k = len(omega)
    N = len(tau)
    dG = [[0]*k for i in range(N)]
    for i in range(N):
        for j in range(1,k):
            x = (omega[j]*tau[i]-b[i])/sqrt(b[i]/s)
            xr = min(x,MAX_x) if x >0 else max(x,-MAX_x) 
            xl = xr-(omega[j]-omega[j-1])*tau[i]/sqrt(b[i]/s)
            if xr > 0:
                dG[i][j] = norm.sf(xl) - norm.sf(xr)
            else:
                dG[i][j] = norm.cdf(xr) - norm.cdf(xl)    
    return dG        

def compute_delta_g(omega,tau,b,s):
    k = len(omega)
    N = len(tau)
    dg = [[0]*k for i in range(N)]
    for i in range(N):
        sigma = sqrt(b[i]/s)
        beta = b[i]/sigma
        for j in range(1,k):
            alpha_l = omega[j-1]/sigma
            alpha_r = omega[j]/sigma
            x = alpha_r*tau[i]-beta
            xr = min(x,MAX_x) if x >0 else max(x,-MAX_x)
            xl = xr-(omega[j]-omega[j-1])*tau[i]/sqrt(b[i]/s)
            dg[i][j] = alpha_r*norm.pdf(xr)-alpha_l*norm.pdf(xl)
    return dg

def compute_delta_gp(omega,tau,b,s):
# gp: derivative of g
    k = len(omega)
    N = len(tau)
    dgp = [[0]*k for i in range(N)] # gp: g prime
    for i in range(N):
        sigma = sqrt(b[i]/s)
        beta = b[i]/sigma
        for j in range(1,k):
            alpha_l = omega[j-1]/sigma
            alpha_r = omega[j]/sigma
            x = alpha_r*tau[i]-beta
            xr = min(x,MAX_x) if x >0 else max(x,-MAX_x)
            xl = xr-(omega[j]-omega[j-1])*tau[i]/sqrt(b[i]/s)
            gp_r = -alpha_r*alpha_r*xr/sqrt(2*pi)*exp(-xr*xr/2)
            gp_l = -alpha_l*alpha_r*xl/sqrt(2*pi)*exp(-xl*xl/2)
            dgp[i][j] = gp_r-gp_l
    return dgp

def F_obj(tau,*args):
    omega = args[0]
    Q = args[1]
    b = args[2]
    s = args[3]            
    N = len(tau)
    k = len(omega)

    # compute dG
    dG = compute_delta_G(omega,tau,b,s)
    
    # compute F
    F = 0
    for i in range(N):
        for j in range(1,k):
            F -= Q[i][j]*log(dG[i][j])
        F += log(tau[i])   
    #print("F",F)     
    return F    

def J_obj(tau,*args):
# compute the Jacobian vector   
    omega = args[0]
    Q = args[1]
    b = args[2]
    s = args[3]            
    N = len(tau)
    k = len(omega)

    # compute dG
    dG = compute_delta_G(omega,tau,b,s)

    # compute dg
    dg = compute_delta_g(omega,tau,b,s)
   
    # compute J
    J = [0]*N
    for i in range(N):
        for j in range(1,k):
            J[i] -= Q[i][j]*dg[i][j]/dG[i][j]
        J[i] += 1/tau[i]   
    #print("J",J)     
    return np.array(J)

def H_obj(tau,*args):
# compute the Hessian matrix
    omega = args[0]
    Q = args[1]
    b = args[2]
    s = args[3]            
    N = len(tau)
    k = len(omega)

    # compute dG
    dG = compute_delta_G(omega,tau,b,s)
    
    # compute g
    dg = compute_delta_g(omega,tau,b,s)
    
    # compute dgp
    dgp = compute_delta_gp(omega,tau,b,s)
    
    # compute diagonal entries
    Hd = [0]*N
    for i in range(N):    
        for j in range(1,k):
            v_j = dG[i][j]
            u_j = dg[i][j]
            up_j = dgp[i][j] # derivative of u_j 
            Hd[i] -= Q[i][j]*(up_j*v_j-u_j*u_j)/(v_j*v_j)
        Hd[i] -= 1/(tau[i]*tau[i])
    #print("H",Hd)    
    return diags(Hd)

def f_ll_umix(b,s,tau,omega,phi):
    F = 0
    N = len(tau)
    k = len(omega)

    dG = compute_delta_G(omega,tau,b,s)

    for i in range(N):
        f_i = 0
        for j in range(1,k):
            f_i += phi[j]*sqrt(b[i]/s)/(omega[j]-omega[j-1])/tau[i]*dG[i][j]
        F += log(f_i)   
    return F     

def run_Estep_umix(b,s,omega,tau,phi):
    N = len(tau)
    k = len(omega)
    dG = compute_delta_G(omega,tau,b,s)
    
    Q = []
    for i in range(N):
        q_i = [0]*k
        for j in range(1,k):
            q_i[j] = phi[j]*sqrt(b[i]/s)/(omega[j]-omega[j-1])/tau[i]*dG[i][j] 
        sq_i = sum(q_i)    
        Q.append([x/sq_i for x in q_i])
    return Q    

def compute_phi_star(Q,eps_phi=EPS_phi):
    k = len(Q[0])
    phi = [0]*k
    N = 0
    for Qi in Q:
        if Qi is not None:
            phi = [x+y for (x,y) in zip(phi,Qi)]
            N += 1
    return [max(x/N,eps_phi) for x in phi]    
    
def run_Mstep_umix(b,s,omega,tau,phi,Q,M,dt,eps_tau=EPS_tau,eps_phi=EPS_phi):
    phi_star = compute_phi_star(Q,eps_phi=eps_phi)
    tau_star = compute_tau_star(b,s,omega,tau,Q,M,dt,eps_tau=EPS_tau)
    return phi_star,tau_star

def compute_tau_star(b,s,omega,tau,Q,M,dt,eps_tau=EPS_tau):
    N = len(tau)
    linear_constraint = LinearConstraint(csr_matrix(M),dt,dt,keep_feasible=False)
    bounds = Bounds(np.zeros(N)+eps_tau,np.inf,keep_feasible=True)
    args = (omega,Q,b,s)
            
    result = minimize(fun=F_obj,method="trust-constr",x0=tau,args=args,bounds=bounds,constraints=[linear_constraint],options={'disp':True,'verbose':3,'maxiter':1000},jac=J_obj)#,hess=H_obj)
    print(F_obj(tau,*args))
    #for i in range(3):
    #    result = minimize(fun=F_obj,method="SLSQP",x0=tau,args=args,bounds=bounds,constraints=[linear_constraint],options={'disp':True,'maxiter':3},jac=J_obj)#,hess=H_obj)
    #    tau = result.x
    tau_star = result.x    
    return tau_star 

def EM_date_umix(tree,smpl_times,root_age=None,refTree=None,trueTreeFile=None,s=1000,k=100,input_omega=None,df=5e-4,maxIter=100,eps_tau=EPS_tau,init_rate_distr=None,verbose=False):
    M, dt, b = setup_constr(tree,smpl_times,s,root_age=root_age,eps_tau=eps_tau)
    b_avg = [sum(b_i)/len(b_i) for b_i in b] 
    tau, phi, omega = init_EM(tree,smpl_times,k=k,input_omega=input_omega,s=s,refTree=refTree,init_rate_distr=init_rate_distr)
    if verbose:
        print("Initialized EM")
    pre_llh = f_ll_umix(b_avg,s,tau,omega,phi)
    if verbose:
        print("Initial likelihood: " + str(pre_llh))
    for i in range(1,maxIter+1):
        if verbose:
            print("EM iteration " + str(i))
            print("Estep ...")
        Q = run_Estep_umix(b_avg,s,omega,tau,phi)
        if verbose:
            print("Mstep ...")   
        next_phi,next_tau = run_Mstep_umix(b_avg,s,omega,tau,phi,Q,M,dt,eps_tau=eps_tau)
        llh = f_ll_umix(b_avg,s,next_tau,omega,next_phi)
        if verbose:
            print("Current llh: " + str(llh))
        curr_df = None if pre_llh is None else llh - pre_llh
        if verbose:
            print("Current df: " + str(curr_df))
        if curr_df is not None and abs(curr_df) < df:
            break
        print(max(abs(t1-t2) for t1,t2 in zip(tau,next_tau)))
        #print([p1-p2 for p1,p2 in zip(phi,next_phi)])
        phi = next_phi
        tau = next_tau    
        pre_llh = llh    

    # convert branch length to time unit and compute mu for each branch
    for node in tree.traverse_postorder():
        if not node.is_root():
            node.set_edge_length(tau[node.idx])
            node.mu = sum(o*p for (o,p) in zip(omega,Q[node.idx]))

    # compute divergence times
    compute_divergence_time(tree,smpl_times)

    return tau,omega,phi,llh

if __name__ == "__main__":
    from sys import argv

    inTree = read_tree_newick(argv[1])
    startTree = read_tree_newick(argv[2])
    k = 50
    omega_max = 0.03
    omega = [ i/k*omega_max for i in range(k+1) ]
    phi = [0] + [1/k]*k
    init_rate_distr = multinomial(omega,phi)
             
    smpl_times = {}             
    with open(argv[3],"r") as fin:
        for line in fin:
            name,age = line.split()
            smpl_times[name] = float(age)
    
    tau,omega,phi,llh = EM_date_umix(inTree,smpl_times,refTree=startTree,s=1000,df=5e-3,maxIter=20,eps_tau=EPS_tau,init_rate_distr=init_rate_distr,verbose=True)
    inTree.write_tree_newick(argv[4])    
    with open(argv[5],'w') as fout:
        for (o,p) in zip(omega[1:],phi[1:]):
            fout.write(str(o) + " " + str(p) + "\n")

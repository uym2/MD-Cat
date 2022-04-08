lsd_exec="/calab_data/mirarab/home/umai/my_gits/EM_Date/Software/lsd-0.2/bin/lsd.exe"

def run_lsd(tree,sampling_time,s=1000,eps_tau=EPS_tau):
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
    tau = init_tau_from_refTree(tree,result_tree_file,eps_tau=eps_tau)
    return mu,tau

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
    phi_star = compute_phi_star_cvxpy(Q) if not fixed_phi else phi
    tau_star = compute_tau_star_cvxpy(tau,omega,Q,b,s,M,dt,eps_tau=EPS_tau) if not fixed_tau else tau
    
    return phi_star, tau_star, omega

def f_ll_naive(b,s,tau,omega,phi):
    ll = 0
    for b_i,tau_i in zip(b,tau): 
        lli = 0
        for j,(omega_j,phi_j) in enumerate(zip(omega,phi)):
            var_ij = omega_j*tau_i/s
            lli += norm.pdf(b_i,omega_j*tau_i,sqrt(var_ij))*phi_j
        ll += log(lli)
    return ll

def compute_phi_star_cvxpy(Q,omega,mu_avg=None,eps_phi=1e-7):
    k = len(Q[0])
    S = [0]*k
    for Qi in Q:
        S = [x+y for (x,y) in zip(S,Qi)]

    var_phi = cp.Variable(k,pos=True)
       
    objective = cp.Maximize(np.array(S).T @ cp.log(var_phi))
    if mu_avg is not None:
        constraints = [ (np.zeros(k)+1).T @ var_phi == 1, var_phi.T @ omega == mu_avg, np.zeros(k)+eps_phi <= var_phi ]
    else:
        constraints = [ (np.zeros(k)+1).T @ var_phi == 1, np.zeros(k)+eps_phi <= var_phi ]

    prob = cp.Problem(objective,constraints)
    f_star = prob.solve(verbose=False,solver=cp.MOSEK)
    phi_star = var_phi.value

    return phi_star

def compute_tau_star_cvxpy(tau,omega,Q,b,s,M,dt,eps_tau=EPS_tau,var_apprx=False):
    N = len(b)
    k = len(omega)
    Pd = np.zeros(N)
    q = np.zeros(N)

    for i in range(N):
        if b[i] is None:
            continue
        for j in range(k):
            if not var_apprx:
                #w_ij = tau[i]
                w_ij = omega[j]*tau[i] # weight by the variance multiplied with s; use previous tau to estimate
            else:
                w_ij = b[i]    
            #for x in b[i]:
            Pd[i] += Q[i][j]*omega[j]**2/w_ij
            q[i] -= 2*b[i]*Q[i][j]*omega[j]/w_ij
          
    P = np.diag(Pd)        
    var_tau = cp.Variable(N)
       
    objective = cp.Minimize(cp.quad_form(var_tau,P) + q.T @ var_tau)
    upper_bound = np.array([float("inf") if b_i is not None else 1.0/6 for b_i in b])
    constraints = [np.zeros(N)+eps_tau <= var_tau, csr_matrix(M)@var_tau == np.array(dt)]

    prob = cp.Problem(objective,constraints)
    f_star = prob.solve(verbose=False,solver=cp.MOSEK)
    tau_star = var_tau.value

    return tau_star

def compute_omega_star_cvxpy(tau,omega,Q,b,phi,eps_omg=0.0001,var_apprx=False,mu_avg=None):
    N = len(b)
    k = len(omega)
    Pd = np.zeros(k)
    q = np.zeros(k)

    for j in range(k):
        for i in range(N):
            if not var_apprx:
                w_ij = omega[j]*tau[i] # weight by the variance multiplied with s; use previous omega to estimate
            else:
                w_ij = b[i]    
            Pd[j] += Q[i][j]*tau[i]**2/w_ij
            q[j] -= 2*b[i]*Q[i][j]*tau[i]/w_ij
          
    P = np.diag(Pd)        
    var_omega = cp.Variable(k,pos=True)
    
    objective = cp.Minimize(cp.quad_form(var_omega,P) + q.T @ var_omega)
    if mu_avg is None:
        constraints = [np.zeros(k)+eps_omg <= var_omega]
    else:
        constraints = [np.zeros(k)+eps_omg <= var_omega, var_omega.T @ phi == mu_avg]
    prob = cp.Problem(objective,constraints)
    f_star = prob.solve(verbose=False,solver=cp.MOSEK)
            
    omega_star = var_omega.value

    return omega_star

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

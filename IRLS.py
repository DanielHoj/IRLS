import numpy as np

# Binomial: 
    
def BinomialLinkInv(x):
    return 1 / (1 + np.exp(-x))

def BinomialMuEta(x):
    return np.exp(x)/(np.exp(x)+1)**2

def BinomialVariance(x):
    return x*(1-x)

# Poisson

def PoissonLinkInv(x):
    return np.exp(x)

def PoissonMuEta(x):
    return np.exp(x)

def PoissonVariance(x):
    return x


def IRLS(dummy, y, dist, it_max = 40, tol_max = 10e-6):
    
    # Get family functions
    if dist in ['poisson']:
        MuEta = PoissonMuEta
        LinkInv = PoissonLinkInv
        Variance = PoissonVariance
        
    elif dist in ['binom']:
        MuEta = BinomialMuEta
        LinkInv = BinomialLinkInv
        Variance = BinomialVariance
    
    
    # Rows and columns of dummy matrix
    if len(dummy.shape) == 1:
        n_cols = 1
        n_rows = dummy.shape[0]
    else:
        n_cols = dummy.shape[1]
        n_rows = dummy.shape[0]
    
    # Make initial guess for beta
    beta = np.zeros(n_cols)
    
    weights = np.repeat(1, n_rows)
    
    beta1 = []
    converge = False
    
    for k in range(it_max):
        
        # Linear predictor
        eta = np.dot(dummy, beta)
        
        
        # Mu
        mu = LinkInv(eta)
        
        # Variance of Mu
        varmu = Variance(mu)
        
        # Derivative of the inverse-link function with respect to the linear predictor
        mu_eta_val = MuEta((eta))
        
        # Making sure no zero weights or division by zero
        good = ((varmu > 1e-15)) & (mu_eta_val != 0)
        if not any(good):
            break
        
        # Working variable
        z = eta[good] + (y - mu)[good]/mu_eta_val[good]
        
        # Weight matrix
        w = np.diagflat((weights[good] * mu_eta_val[good]**2)/varmu[good])
        #v = (v/np.sum(v)) + 1e-9
        
        # Set beta(i-1)
        beta_old = beta
        
        try:
            beta = np.linalg.inv(dummy[good].T@w@dummy[good]) @ dummy[good].T @ w @ z
        except:
            break
        #weights = np.diag(w)
        
        beta1.append(beta)
        
        if np.linalg.norm(beta_old - beta)/(1e-9 + np.linalg.norm(beta_old)) < tol_max:
            converge = True
            break

    return beta, np.array(beta1), converge

    
    
    
    
    
    
    
    
    
    
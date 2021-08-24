import numpy as np

def delta_r_estimator(survey_range, lmax, cl_sys):
    """ Get a estimated systematic bias i.e. delta_r
    
    Parameters
    ----------
    survey_range: array
        Enter the range of delta_r to calculate the likelihood function.
        ex.) survey_range = np.linspace(1e-6, 1e-3, 1000)
    lmax: int
        In most cases, 3*nside-1 is fine.
    cl_sys: array
        Enter the systematic bias of B-mode spectra.
        
    Returns
    -------
    delta_r: froat
        The estimated bias.
    likelihood: array
        Likelihood function as array.
    """
    path = "../data/ClBB_PTEPini.npz"
    cl_models = np.load(path)
    cl_lens = cl_models["lens"]
    cl_tens = cl_models["tensor"]
    l_array = np.arange(2, lmax+1) # It starts at 2 to exclude the unipolar and bipolar components.
    N_r = len(survey_range)
    N_l = len(l_array)
    array0 = np.zeros(N_r)

    for i in range(0, N_r):
        #The Cl in loop starts at 2 to remove the monopolar and dipolar components.
        Cl_hat = cl_sys[2:N_l+2] + cl_lens[2:N_l+2]
        Cl = survey_range[i] * cl_tens[2:N_l+2] + cl_lens[2:N_l+2]
        logPl = ((-0.5) * (2*l_array + 1)*((Cl_hat / Cl) + np.log(Cl) - ((2*l_array - 1)/(2*l_array + 1)) * np.log(Cl_hat))).sum()
        array0[i] = logPl

    likelihood = np.exp(array0-max(array0)) #  This calculation is standardized to avoid calculating an exponential function that is too large.
    r_max_id = np.argmax(likelihood) # surching r that make peakposition of likelihood function.
    delta_r = survey_range[r_max_id]
    return (delta_r, likelihood)

def Cl2Dl(cl):
    """ Convert C_\ell to D_\ell
    
    Parameter
    ---------
    cl: array
        Power spectrum
    
    Return
    ------
        D_\ell: l(l+1)Cl/2π
    """
    lmax = len(cl)-1
    l = np.arange(lmax+1)
    lp1 = np.arange(1,lmax+2)
    return (cl*l*lp1)/(2.*np.pi)
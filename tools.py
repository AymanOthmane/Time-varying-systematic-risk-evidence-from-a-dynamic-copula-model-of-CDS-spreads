import numpy as np
from scipy.optimize import minimize


def nines(*args):
    """
    Returns a matrix or array filled with -999.99.

    Parameters:
        *args: Variable length argument list. Can be a sequence of dimensions or an array-like to specify the shape.
    
    Returns:
        np.ndarray: An array of -999.99s with the specified shape.
    """

    # Handle the case where the first argument is array-like or scalar
    if len(args) == 1:
        if np.isscalar(args[0]):
            # Single scalar, assume a square matrix of that size
            return -999.99 * np.ones((args[0], args[0]))
        else:
            # Array-like, create an array of the same shape
            return -999.99 * np.ones(np.shape(args[0]))
    
    # Handle the case with two or more dimensions specified
    elif len(args) > 1:
        shape = tuple(arg if np.isscalar(arg) else len(arg) for arg in args)
        return -999.99 * np.ones(shape)
    
    # Default case if no arguments are provided
    else:
        return np.array([-999.99])




def rhobar2betabar(rhobar):
    N = rhobar.shape[0]
    if N < 3:
        raise ValueError("This mapping requires there to be at least 3 variables.")
    
    theta0 = np.ones((N, 1))

    def rho2theta(rho):
        k = rho.shape[1]
        out1 = -999.99 * np.ones((k * (k - 1) // 2, 1))
        #out1 = nines(np.ones((k * (k - 1) // 2, 1)))
        
        counter = 0
        for ii in range(k):
            for jj in range(ii + 1, k):
                out1[counter] = rho[ii, jj]
                counter += 1
        return out1

    def rhobar2betabar_calc(beta, rhobar):
        Nb = len(beta)
        rho = np.full((Nb, Nb), np.nan)
        for ii in range(Nb):
            for jj in range(ii + 1, Nb):
                rho[ii, jj] = beta[ii] * beta[jj] / np.sqrt((1 + beta[ii]**2) * (1 + beta[jj]**2))
                rho[jj, ii] = rho[ii, jj]
        return np.sum((rho2theta(rho) - rho2theta(rhobar))**2)
    
    # Flatten theta0 to make it 1D as required by `minimize`
    theta0_flat = theta0.flatten()

    # Optimization settings
    options = {'disp': False, 'gtol': 1e-6}
    result = minimize(lambda beta: rhobar2betabar_calc(beta, rhobar), theta0_flat, method='BFGS', options=options)
    if not result.success:
        raise RuntimeError("Optimization did not converge: " + result.message)
    
    return result.x

# Example use case
rhobar = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]])
betabar = rhobar2betabar(rhobar)
print("Estimated Loadings on the Common Factor:")
print(betabar)

import numpy as np
from scipy.stats import t
from scipy.special import gamma
from scipy import sqrt

def skewt_cdf(x, nu, lambda_):
    """
    Calculate the cumulative distribution function (CDF) for Hansen's skewed t-distribution.
    
    Parameters:
    x : array_like
        Data points where the CDF is evaluated. Can be a matrix, vector, or scalar.
    nu : array_like
        Degrees of freedom parameter, can be a matrix, vector, or scalar.
    lambda_ : array_like
        Skewness parameter, can be a matrix, vector, or scalar.

    Returns:
    np.ndarray
        A matrix of CDF values at each element of x.
    
    Notes:
    The function adapts the CDF calculation to account for skewness in the distribution. 
    It adjusts for different dimensions of `nu` and `lambda_` to match `x` if necessary.
    """
    # Convert input to numpy arrays to facilitate mathematical operations
    x = np.asarray(x)
    nu = np.atleast_1d(nu)
    lambda_ = np.atleast_1d(lambda_)

    # Check dimensions and broadcast nu and lambda if necessary to match the dimensions of x
    T, k = x.shape
    if nu.size < T:
        nu = np.full(T, nu[0])
    if lambda_.size < T:
        lambda_ = np.full(T, lambda_[0])

    # Compute constants used in the skewed t-distribution formula
    c = gamma((nu+1)/2) / (sqrt(np.pi * (nu-2)) * gamma(nu/2))
    a = 4 * lambda_ * c * ((nu-2) / (nu-1))
    b = sqrt(1 + 3 * lambda_**2 - a**2)

    # Transform x for the calculation of CDF values
    y1 = (b * x + a) / (1 - lambda_) * sqrt(nu / (nu-2))
    y2 = (b * x + a) / (1 + lambda_) * sqrt(nu / (nu-2))

    # Calculate the CDF using the regular t-distribution's CDF function
    # Conditionally calculate based on the skewness adjusted regions
    cdf = (1 - lambda_) * t.cdf(y1, nu) * (x < -a / b)
    cdf += (x >= -a / b) * ((1 - lambda_) / 2 + (1 + lambda_) * (t.cdf(y2, nu) - 0.5))
    
    return cdf


def skewt_inv_cdf(u, nu, lambda_):
    """
    Returns the inverse CDF (quantiles) for Hansen's skewed t-distribution at given probabilities.
    
    Parameters:
    u : array_like
        Probabilities at which to evaluate the inverse CDF (quantiles). Should be in the unit interval (0, 1).
    nu : array_like
        Degrees of freedom parameter, can be a matrix or scalar.
    lambda_ : array_like
        Skewness parameter, can be a matrix or scalar.

    Returns:
    np.ndarray
        An array of quantiles corresponding to each probability in u.
    """
    u = np.asarray(u)
    nu = np.asarray(nu)
    lambda_ = np.asarray(lambda_)

    T, k = u.shape
    if nu.size < T:
        nu = np.full((T, 1), nu[0])
    if lambda_.size < T:
        lambda_ = np.full((T, 1), lambda_[0])

    c = gamma((nu + 1) / 2) / (sqrt(np.pi * (nu - 2)) * gamma(nu / 2))
    a = 4 * lambda_ * c * ((nu - 2) / (nu - 1))
    b = sqrt(1 + 3 * lambda_**2 - a**2)

    f1 = u < (1 - lambda_) / 2
    f2 = u >= (1 - lambda_) / 2

    inv1 = ((1 - lambda_[f1]) / b[f1] * sqrt((nu[f1] - 2) / nu[f1]) *
            t.ppf(u[f1] / (1 - lambda_[f1]), nu[f1]) - a[f1] / b[f1])
    inv2 = ((1 + lambda_[f2]) / b[f2] * sqrt((nu[f2] - 2) / nu[f2]) *
            t.ppf(0.5 + 1 / (1 + lambda_[f2]) * (u[f2] - (1 - lambda_[f2]) / 2), nu[f2]) - a[f2] / b[f2])

    inv = -999.99 * np.ones((T, 1))
    inv[f1] = inv1
    inv[f2] = inv2

    return inv


def skewt_pdf(x, nu, lambda_):
    """
    Returns the probability density function (PDF) of Hansen's skewed t-distribution.

    Parameters:
    x : array_like
        Data points where the PDF is evaluated, can be a matrix, vector, or scalar.
    nu : array_like
        Degrees of freedom parameter, can be a matrix, vector, or scalar.
    lambda_ : array_like
        Skewness parameter, can be a matrix, vector, or scalar.

    Returns:
    np.ndarray
        A matrix of PDF values at each element of x.
    """
    x = np.asarray(x)
    nu = np.asarray(nu)
    lambda_ = np.asarray(lambda_)

    T, k = x.shape
    if nu.size < T:
        nu = np.full((T, 1), nu[0])
    if lambda_.size < T:
        lambda_ = np.full((T, 1), lambda_[0])

    c = gamma((nu + 1) / 2) / (sqrt(np.pi * (nu - 2)) * gamma(nu / 2))
    a = 4 * lambda_ * c * ((nu - 2) / (nu - 1))
    b = sqrt(1 + 3 * lambda_**2 - a**2)

    pdf1 = b * c * (1 + 1 / (nu - 2) * ((b * x + a) / (1 - lambda_))**2)**(-(nu + 1) / 2)
    pdf2 = b * c * (1 + 1 / (nu - 2) * ((b * x + a) / (1 + lambda_))**2)**(-(nu + 1) / 2)
    pdf = pdf1 * (x < (-a / b)) + pdf2 * (x >= (-a / b))

    return pdf


import numpy as np
from scipy.stats import t, gamma
from math import sqrt, pi


class SkewedTDistribution:
    def __init__(self, nu, _lambda):
        self.nu = nu  # Kurtosis Parameter (>2 to inf)
        self._lambda = _lambda #  Assymetry Parameter (-1 to 1)

    def cdf(self, z):
        """_summary_
          Calculates the CDF for the skewed Student-t. Hansen (1994) version.
        Args:
            z : Random Variable value (Between -inf and inf)
        Returns
            u : random Variable (between 0 and 1)
        """
        z  = np.atleast_1d(z)
    
        c = gamma((self.nu+1)/2)/(sqrt(pi*(self.nu-2))*gamma(self.nu/2))
        a = 4*self._lambda*c*((self.nu-2)/(self.nu-1))
        b = sqrt(1 + 3 * self._lambda**2 - a**2)
        
        limit_variable = -a/b
        lt = z < limit_variable
        gt = z >= limit_variable
        
        y_1 = (b*z+a) / (1-self._lambda) * sqrt(self.nu/(self.nu-2))
        y_2 = (b*z+a) / (1+self._lambda) * sqrt(self.nu/(self.nu-2))
        
        pdf1 = (1-self._lambda) * t.cdf(y_1, self.nu)
        pdf2 = (1-self._lambda)/2 + (1+self._lambda) * (t.cdf(y_2, self.nu)-0.5)

        u = z.copy()

        u[lt] = pdf1[lt]
        u[gt] = pdf2[gt]

        return u
    
    def pdf(self, z):
        """_summary_
          Calculates the PDF for the skewed Student-t. Hansen (1994) version.
        Args:
            z : Random Variable value (Between -inf and inf)

        Returns:
            u : random Variable > 0
        """
        z  = np.atleast_1d(z)

        c = gamma((self.nu+1)/2)/(sqrt(pi*(self.nu-2))*gamma(self.nu/2))
        a = 4*self._lambda*c*((self.nu-2)/(self.nu-1))
        b = sqrt(1 + 3 * self._lambda**2 - a**2)

        limit_variable = -a/b
        lt = z < limit_variable
        gt = z >= limit_variable

        y_1 = b * c * (1 + (1/(self.nu-2)) * ((b*z+a) / (1-self._lambda))**2 )**-((self.nu+1)/2)
        y_2 = b * c * (1 + (1/(self.nu-2)) * ((b*z+a) / (1+self._lambda))**2 )**-((self.nu+1)/2)

        u = z.copy()

        u[lt] = y_1[lt]
        u[gt] = y_2[gt]
        return u
    
    def inv_cdf(self, u):
        """_summary_
          Calculates the inverse CDF for the skewed Student-t. Hansen (1994) version.
        Args:
            u : Cdf value (Between 0 and 1)

        Returns:
            z : random Variable (between -inf and inf)
        """
        u  = np.atleast_1d(u)
    
        c = gamma((self.nu+1)/2)/(sqrt(pi*(self.nu-2))*gamma(self.nu/2))
        a = 4*self._lambda*c*((self.nu-2)/(self.nu-1))
        b = sqrt(1 + 3 * self._lambda**2 - a**2)
        
        inv1 = (1-self._lambda)/b * sqrt((self.nu-2) / self.nu) * t.ppf(u / (1-self._lambda), self.nu) - a/b
        inv2 = (1+self._lambda)/b * sqrt((self.nu-2) / self.nu) * t.ppf(0.5 + (1 / (1+self._lambda)) * (u - (1-self._lambda)/2) , self.nu) - a/b
        
        limit_variable = (1-self._lambda)/2
        
        z = u.copy()
        
        lt = u < limit_variable
        gt = u >= limit_variable
        
        z[lt] = inv1[lt]
        z[gt] = inv2[gt]

        return z

    
    
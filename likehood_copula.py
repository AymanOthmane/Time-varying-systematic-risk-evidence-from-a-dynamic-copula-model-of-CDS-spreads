from skew_t import *

from scipy.special import roots_legendre
import numpy as np

def factor_cop_Gcdf_calc1_Skewtt(u, x, theta):
    """
    Calcule la valeur de l'argument de l'intégrale pour une distribution t asymétrique.
    
    Args:
        u (np.ndarray): Un vecteur Kx1 de valeurs de u utilisé par la fonction d'intégration numérique.
        x (float): La valeur de x pour laquelle évaluer G.
        theta (list): Liste contenant les paramètres [sig2z, nuinv1, nuinv2, lam] de Fz et Feps.

    Returns:
        np.ndarray: Un vecteur Kx1, la valeur de l'argument de l'intégrale à chaque valeur de u.
    """
    sig2z, nuinv1, nuinv2, lam = theta  # Décomposition des paramètres
    
    # Assure que u est un vecteur colonne
    if u.ndim > 1 and u.shape[1] > u.shape[0]:
        u = u.T

    # Calcul de l'inverse de la distribution t asymétrique
    # skewt_inv_cdf doit être définie ailleurs
    z_inv = skewt_inv_cdf(u, 1/nuinv1, lam) * np.sqrt(sig2z)
    
    # Calcul de la fonction de répartition cumulative de la distribution t asymétrique
    # skewtdis_cdf doit être définie ailleurs
    out1 = skewt_cdf(x - z_inv, 1/nuinv2, 0)
    
    return out1




def GLquad(f_name, a=-1, b=1, n=10, *args):
    """
    Effectue une quadrature de Gauss-Legendre sur une fonction univariée sur l'intervalle [a, b].
    
    Args:
        f_name (function): La fonction à intégrer, qui prend un vecteur de valeurs x.
        a (float): La borne inférieure de l'intervalle. Par défaut à -1.
        b (float): La borne supérieure de l'intervalle. Par défaut à 1.
        n (int): Le nombre de "noeuds" à utiliser dans la quadrature. Par défaut à 10.
        args: Arguments supplémentaires à passer à la fonction f_name.

    Returns:
        float: L'intégrale estimée de la fonction f_name sur l'intervalle [a, b].
    """
    # Obtention des abscisses (x) et des poids (w) pour la quadrature de Gauss-Legendre
    x, w = roots_legendre(n)
    
    # Transformation des points x pour correspondre à l'intervalle [a, b]
    x_mapped = (x + 1) * (b - a) / 2 + a
    
    # Évaluation de la fonction sur les points mappés
    f_values = f_name(x_mapped, *args)
    
    # Calcul de l'intégrale en utilisant les poids de la quadrature
    integral = np.sum(w * f_values * (b - a) / 2)
    
    return integral



import numpy as np
from scipy.integrate import quad

def factor_cop_FXpdf_GL_Skewtt(x, theta, GLweight, group_code, epsi):
    """
    Calcule la densité jointe de [X1, ..., XN] associée à un modèle de facteurs t-skew t
    avec d'autres densités jointes évaluées à des charges factorielles modifiées.
    
    Args:
        x (numpy.ndarray): Matrice Nx2, les valeurs de x pour évaluer g(x1,...,xN).
        theta (list): Paramètres [factor loading, nuinv_z, nuinv_esp, psi_z].
        GLweight (numpy.ndarray): Nœuds et poids pour la quadrature de Gauss-Legendre.
        group_code (numpy.ndarray): Vecteur Nx1 des codes de groupe.
        epsi (float): Taille de pas pour la dérivée numérique.
        
    Returns:
        numpy.ndarray: Vecteur de densité jointe évaluée.
    """

    def integrand(u, x, theta, group_code, epsi):
        Ngroup = np.max(group_code)
        if len(theta) - 3 != Ngroup:
            raise ValueError('N_group is not equal to N_theta')

        nuinv_z = theta[-3]
        nuinv_eps = theta[-2]
        psi_z = theta[-1]
        
        N = x.shape[0]
        Nnodes = len(u)
        u = np.reshape(u, (Nnodes, 1))

        Fz_inv_u = skewt_inv_cdf(u, 1/nuinv_z, psi_z)

        xxx = np.empty((Nnodes * N, 1))
        for ii in range(N):
            inx = group_code[ii]
            xxx[Nnodes * ii: Nnodes * (ii + 1), 0] = x[ii, 0] - np.sqrt(theta[inx]) * Fz_inv_u.squeeze()

        xxx = np.tile(xxx, (1, Ngroup + 1))

        for ii in range(N):
            inx = group_code[ii]
            xxx[Nnodes * ii: Nnodes * (ii + 1), inx + 1] = x[ii, 1] - np.sqrt(theta[inx] + epsi) * Fz_inv_u.squeeze()

        xxx = xxx.ravel()

        out_temp = skewt_pdf(xxx, 1/nuinv_eps, 0)
        out_temp = out_temp.reshape(Nnodes * N, Ngroup + 1)

        out = np.empty((Nnodes, Ngroup + 1))
        for ii in range(Ngroup + 1):
            temp = out_temp[:, ii].reshape(Nnodes, N)
            out[:, ii] = np.prod(temp, axis=1)

        return out

    # Intégration utilisant la quadrature Gauss-Legendre
    nodes = GLweight[:, 0]
    weights = GLweight[:, 1]
    result = sum([w * integrand(u, x, theta, group_code, epsi) for u, w in zip(nodes, weights)])
    
    return result




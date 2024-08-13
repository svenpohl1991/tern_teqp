import teqp
import numpy as np
from scipy.optimize import root
from ternary_teqp.derivs import *


class HybrjFunctorMixVLETp:
    """
    A class to evaluate and optimize a function related to pressure and its derivatives
    based on two component vectors.

    Attributes:
        model: An object that provides the method `get_R`.
        T: A scalar temperature value.
        p: A scalar reference pressure value.
    """

    def __init__(self, model, T, p):
        """
        Initializes the HybrjFunctorMixVLETp instance.

        Args:
            model: An object with a method `get_R`.
            T: Temperature value (scalar).
            p: Reference pressure value (scalar).
        """
        self.model = model
        self.T = T
        self.p = p

    def __call__(self, x):
        """
        Evaluates the residuals based on the input vector `x`.

        Args:
            x: A vector containing concatenated component densities for L and V phases.

        Returns:
            A vector of residuals based on the pressure calculations and their gradients.
        """
        n = len(x) // 2  # Number of components
        rhovecL = x[:n]
        rhovecV = x[n:]

        # Compute the RT value
        rhoL_sum = np.sum(rhovecL)
        RT = self.model.get_R(rhovecL / rhoL_sum) * self.T

        # Compute Psi and its gradients
        PsirL, PsirgradL, _ = build_Psir_fgradHessian_autodiff(self.model, self.T, rhovecL)
        PsirV, PsirgradV, _ = build_Psir_fgradHessian_autodiff(self.model, self.T, rhovecV)

        # Sum of densities for L and V phases
        rhoL = rhoL_sum
        rhoV = np.sum(rhovecV)

        # Compute pressures for L and V phases
        pL = rhoL * RT - PsirL + np.sum(rhovecL * PsirgradL)
        pV = rhoV * RT - PsirV + np.sum(rhovecV * PsirgradV)

        # Initialize residual vector
        r = np.zeros(2 * n)

        # Compute residuals for each component
        for i in range(n):
            if rhovecL[i] > 0 and rhovecV[i] > 0:
                r[i] = PsirgradL[i] + RT * np.log(rhovecL[i]) - (PsirgradV[i] + RT * np.log(rhovecV[i]))
            else:
                r[i] = PsirgradL[i] - PsirgradV[i]

        # Compute residuals for pressure constraints
        r[n] = (pV - self.p) / self.p
        r[n + 1] = (pL - self.p) / self.p

        return r

    def jacobian(self, x):
        """
        Computes the Jacobian matrix of the residuals with respect to the input vector `x`.

        Args:
            x: A vector containing concatenated component densities for L and V phases.

        Returns:
            A Jacobian matrix of shape (2 * n, 2 * n) where n is the number of components.
        """
        n = len(x) // 2
        rhovecL = x[:n]
        rhovecV = x[n:]

        # Initialize the Jacobian matrix
        J = np.zeros((2 * n, 2 * n))

        # Compute the RT value
        rhoL_sum = np.sum(rhovecL)
        RT = self.model.get_R(rhovecL / rhoL_sum) * self.T

        # Compute Psi and its gradients and Hessians
        _, PsirgradL, hessianL = build_Psir_fgradHessian_autodiff(self.model, self.T, rhovecL)
        _, PsirgradV, hessianV = build_Psir_fgradHessian_autodiff(self.model, self.T, rhovecV)

        # Compute dp/drho for L and V phases
        dpdrhovecL = RT + np.dot(hessianL, rhovecL)
        dpdrhovecV = RT + np.dot(hessianV, rhovecV)

        # Fill in the Jacobian matrix
        for i in range(n):
            # Diagonal terms
            J[i, i] = hessianL[i, i] + (RT / rhovecL[i] if rhovecL[i] > 0 else 0)
            # Off-diagonal terms
            for j in range(n):
                if i != j:
                    J[i, j] = hessianL[i, j]
            J[i, n + i] = -(hessianV[i, i] + (RT / rhovecV[i] if rhovecV[i] > 0 else 0))
            for j in range(n):
                if i != j:
                    J[i, n + j] = -hessianV[i, j]

        # Pressure constraint Jacobian terms
        J[n, n:] = dpdrhovecV / self.p
        J[n + 1, :n] = dpdrhovecL / self.p

        return J

def mix_VLE_Tp(model, T, pgiven, rhovecL0, rhovecV0, maxiter=100):
    functor = HybrjFunctorMixVLETp(model, T, pgiven)
    
    x0 = np.concatenate([rhovecL0, rhovecV0])
    r = functor(x0)
    J = functor.jacobian(x0)
    sol = root(functor, x0, jac=functor.jacobian, method='hybr', options={'maxfev': maxiter, 'xtol' : 1E-10})
    
    rhovecL = sol.x[:len(rhovecL0)]
    rhovecV = sol.x[len(rhovecL0):]
    
    return {
        'rhovecL': rhovecL,
        'rhovecV': rhovecV,
        'success': sol.success,
        'message': sol.message,
        'nfev': sol.nfev,
        'njev': sol.njev,
        'r': sol.fun
    }
    
    
class InvalidArgument(Exception):
    pass

class IterationFailure(Exception):
    pass

class TVLEOptions:
    def __init__(self, abs_err=1e-8, rel_err=1e-6, init_c=100.0, init_dt=30.0, max_steps=200, max_dt=1e0, integration_order=5, calc_criticality=False, crit_termination=1e-10, p_termination=1e8, polish=False, polish_reltol_rho=1e-5, polish_exception_on_fail=False, verbosity=0, revision=1):
        self.abs_err = abs_err
        self.rel_err = rel_err
        self.init_c = init_c
        self.init_dt = init_dt
        self.max_steps = max_steps
        self.max_dt = max_dt
        self.integration_order = integration_order
        self.calc_criticality = calc_criticality
        self.crit_termination = crit_termination
        self.p_termination = p_termination
        self.polish = polish
        self.polish_reltol_rho = polish_reltol_rho
        self.polish_exception_on_fail = polish_exception_on_fail
        self.verbosity = verbosity
        self.revision = revision
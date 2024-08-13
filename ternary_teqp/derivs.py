
from scipy.linalg import solve as linsolve
import numpy as np

def get_drhovecdp_Tsat(model, T, rhovecL, rhovecV):
    Hliq = model.build_Psi_Hessian_autodiff(T, rhovecL)
    Hvap = model.build_Psi_Hessian_autodiff(T, rhovecV)
    
    Hvap[np.isinf(Hvap)] = 1E12
    Hliq[np.isinf(Hliq)] = 1E12
    
    N = rhovecL.size
    A = np.zeros((N, N))
    b = np.ones((N, 1))
    b[-1] = 0.0
    id_x_constant = 0
    x_liq = rhovecL/rhovecL.sum()
    if np.all(rhovecL != 0) and np.all(rhovecV != 0):
        # Normal treatment for all concentrations not equal to zero
        A[0, 0] = np.dot(Hliq[0, :], rhovecV)
        A[0, 1] = np.dot(Hliq[1, :], rhovecV)
        A[0, 2] = np.dot(Hliq[2, :], rhovecV)
        A[1, 0] = np.dot(Hliq[0, :], rhovecL)
        A[1, 1] = np.dot(Hliq[1, :], rhovecL)
        A[1, 2] = np.dot(Hliq[2, :], rhovecL)
        for i in range(N):
            if(i==id_x_constant):
                A[-1,i] = 1.0  - x_liq[i]
            else:
                A[-1,i] = x_liq[i]
        drhodp_liq = linsolve(A, b)
        drhodp_vap = linsolve(Hvap, np.dot(Hliq, drhodp_liq))
    else:
        # Special treatment for infinite dilution
        murL = model.build_Psir_gradient_autodiff(T, rhovecL)
        murV = model.build_Psir_gradient_autodiff(T, rhovecV)
        RL = model.get_R(rhovecL / rhovecL.sum())
        RV = model.get_R(rhovecV / rhovecV.sum())

        # First, for the liquid part
        for i in range(N-1):
            for j in range(N):
                if rhovecL[j] == 0:
                    # Initial values
                    Aij = (Hliq[j, :] * (rhovecV if i == 0 else rhovecL)).flatten()
                    # Apply correction to the j term
                    is_liq = (i == 1)
                    Aij[j] = RL * T if is_liq else RL * T * np.exp(-(murV[j] - murL[j]) / (RL * T))
                    A[i, j] = Aij.sum()
                else:
                    # Normal
                    A[i, j] = np.dot(Hliq[j, :], rhovecV if i == 0 else rhovecL)
    
        for i in range(N):
            if(i==id_x_constant):
                A[-1,i] = 1.0  - x_liq[i]
            else:
                A[-1,i] = x_liq[i]

        drhodp_liq = linsolve(A, b)

        Hvap[np.isinf(Hvap)] = 1E6
        Hliq[np.isinf(Hliq)] = 1E6

        # Special treatment for the vapor part
        diagrhovecL = np.diag(rhovecL)
        PSIVstar = np.dot(diagrhovecL, Hvap)
        PSILstar = np.dot(diagrhovecL, Hliq)

        for j in range(N):
            if rhovecL[j] == 0:
                PSILstar[j, j] = RL * T
                PSIVstar[j, j] = RV * T / np.exp(-(murV[j] - murL[j]) / (RV * T))

        drhodp_vap = linsolve(PSIVstar, np.dot(PSILstar, drhodp_liq))
    
    return drhodp_liq, drhodp_vap

def psir(model,rhoVec,T):
    rho = sum(rhoVec)
    x   = rhoVec/rho
    return model.get_Ar00(T,rho,x) * model.get_R(x) * T * rho

def build_Psir_fgradHessian_autodiff(model,T,rhoVec):
    f      = psir(model,rhoVec,T)
    f_grad = model.build_Psir_gradient_autodiff(T,rhoVec)
    f_hess = model.build_Psir_Hessian_autodiff(T,rhoVec)
    return f,f_grad,f_hess

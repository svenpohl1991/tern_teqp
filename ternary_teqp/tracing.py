import numpy as np
from ternary_teqp.derivs      import *
from ternary_teqp.vle_solvers import *
from scipy.integrate import solve_ivp


class InvalidArgument(Exception):
    pass

class IterationFailure(Exception):
    pass

class TVLEOptions:
    def __init__(self, abs_err=1e-8, rel_err=1e-6, init_c=1.0, init_dt=1.0, max_steps=500, max_dt=5e3, integration_order=5, calc_criticality=False, crit_termination=1e-10, p_termination=1e8, polish=False, polish_reltol_rho=1e-5, polish_exception_on_fail=False, verbosity=0, revision=1):
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
        
class ode_system:
    def __init__(self,model,T,N,opt):
        self.model = model
        self.T     = T
        self.N     = N
        self.set_previous_drhodt  = None
        self.opt = opt
        self.c     = opt.init_c
        self.dt    = opt.init_dt
        self.JSONdata = []
        
    def set_previous_drhodt(self ,X , t):
        self.set_previous_drhodt = self.xprime(self , t ,X)
    
    
    def xprime(self , t ,X):
        rhovecL = np.array(X[:self.N])
        rhovecV = np.array(X[self.N:])
        [drhovecdTL_sat, drhovecdTV_sat] = get_drhovecdp_Tsat(self.model, self.T,rhovecL, rhovecV)
        dpdt = 1.0 / np.sqrt(np.linalg.norm(drhovecdTL_sat) + np.linalg.norm(drhovecdTV_sat))
        drhovecdtL = self.c * drhovecdTL_sat * dpdt
        drhovecdtV = self.c * drhovecdTV_sat * dpdt
        Xprime = np.concatenate((drhovecdtL.flatten(), drhovecdtV.flatten()))
        if self.set_previous_drhodt is not None:
            if np.dot(Xprime[:], self.set_previous_drhodt) < 0:
                Xprime[:] *= -1
        return Xprime
    
    def est_initial_direction(self,t,X):
        dxdt = self.xprime(t, X)
        step = X + dxdt * self.dt
        
        # Flip the sign if the first step would yield any negative concentrations
        if np.any(step < 0):
            self.c *= -1

    def store_point(self,t,X):
        self.last_d_rho_dt = self.xprime(t,X)
        rhovecL = np.array(X[:self.N ])
        rhovecV = np.array(X[self.N:])
        pL = rhovecL.sum() * self.model.get_R(rhovecL / rhovecL.sum()) * self.T + self.model.get_pr(self.T, rhovecL)
        pV = rhovecV.sum() * self.model.get_R(rhovecV / rhovecV.sum()) * self.T + self.model.get_pr(self.T, rhovecV)

        point = {
            "t": t,
            "T / K": self.T,
            "pL / Pa": pL,
            "pV / Pa": pV,
            "c": self.c,
            "rhoL / mol/m^3": rhovecL.tolist(),
            "rhoV / mol/m^3": rhovecV.tolist(),
            "xL1": rhovecL[0] / rhovecL.sum(),
            "xV1": rhovecV[0] / rhovecV.sum(),
            "xL2": rhovecL[1] / rhovecL.sum(),
            "xV2": rhovecV[1] / rhovecV.sum(),
            "xL3": rhovecL[2] / rhovecL.sum(),
            "xV3": rhovecV[2] / rhovecV.sum()
        }

        self.JSONdata.append(point)

    def stop_requested(self,X):
        
        rhovecL = X[:self.N]
        rhovecV = X[self.N:]
        x = rhovecL / rhovecL.sum()
        y = rhovecV / rhovecV.sum()
        p = rhovecL.sum() * self.model.get_R(x) * self.T + self.model.get_pr(self.T, rhovecL)

        if self.opt.calc_criticality:
            condsL = self.model.get_criticality_conditions(self.T, rhovecL)
            condsV = self.model.get_criticality_conditions(self.T, rhovecV)
            if condsL[0] < self.opt.crit_termination or condsV[0] < self.opt.crit_termination:
                return True
            
        if p > self.opt.p_termination:
            return True
        if (x < 0).any() or (x > 1).any() or (y < 0).any() or (y > 1).any() or not np.isfinite(
                rhovecL).all() or not np.isfinite(rhovecV).all():
            return True
        else:
            return False
        

def set_init_state(rhovecL0,rhovecV0):
    N = len(rhovecL0)
    X = np.zeros(2*N)
    X[:N] = rhovecL0
    X[N:] = rhovecV0
    return X


def trace_VLE_isotherm_ternary(model,pgiven, T, rhovecL0, rhovecV0, options=None):
    
    # Get the options, or the default values if not provided
    opt = options or TVLEOptions()
    N   = rhovecL0.size
    x0  = np.concatenate((rhovecL0.flatten(), rhovecV0.flatten()))
    
    o_sys           = ode_system(model,T,len(rhovecL0),opt)

    o_sys.est_initial_direction(0.0,x0)

    x0 = set_init_state(rhovecL0,rhovecV0)
    
    fail_step = 0
    t= 0
    dt = o_sys.opt.init_dt
    
    for istep in range(opt.max_steps):

        if istep == 0:
            o_sys.store_point(t,x0)

        # Solve the ODE using solve_ivp
        try:
            sol = solve_ivp(o_sys.xprime, [t, t + dt], x0, method='RK45', rtol=1E-8, atol=1E-8)
            fail_step = 0
            dt = min(2*dt,o_sys.opt.max_dt)
        except:
            fail_step += 1
            istep -= 1
            dt = dt/2.0
        
        if(fail_step>10):
            break
        
        if(fail_step == 0):

            if not sol.success: 
                break

            x0 = sol.y[:, -1]
            t = sol.t[-1]

            if o_sys.stop_requested(x0):
                break

            rhovecL = x0[:N].copy()
            rhovecV = x0[N:].copy()
            #T = x0[0]
            x = rhovecL/sum(rhovecL)
            p = rhovecL.sum() * model.get_R(x) * T + model.get_pr(T, rhovecL)

            res = mix_VLE_Tp(model, T, pgiven, rhovecL, rhovecV)

            x0[:N] = res['rhovecL']
            x0[N:] = res['rhovecV']


            try:
                o_sys.store_point(t,x0)
                o_sys.set_previous_drhodt  = o_sys.last_d_rho_dt
            except:
                break
        
    return o_sys.JSONdata
import numpy as np
import primpy.potentials as pp
from primpy.events import UntilNEvent, InflationEvent, SlowRowEvent
from primpy.initialconditions import InflationStartIC, ISIC_Nt
from primpy.time.inflation import InflationEquationsT as InflationEquations
from primpy.solver import solve


class SlowRowIC(object):
    """Deep slow row initial conditions given backgound."""
    def __init__(self, background, equations):
        self.background = background
        self.equations = equations

    def __call__(self, y0, **ivp_kwargs):
    
        # #########################################################################
        # Set background equations of inflation for `N`, `phi` and `dphi`.
        # #########################################################################
        SR_key = 'SlowRow_dir1_term0'
        self.x_ini = self.background.t_events[SR_key][0]
        self.x_end = self.background.t[0]

        SR_phi, SR_dphidt, SR_N, SR_eta = self.background.y_events[SR_key][0]
        SR_V = self.background.potential.V(SR_phi)
        SR_dNdt = np.sqrt((SR_dphidt**2 /2 + SR_V) /3 - self.background.K * np.exp(-2*SR_N))
        y0[self.equations.idx['phi']] = SR_phi
        y0[self.equations.idx['dphidt']] = SR_dphidt
        y0[self.equations.idx['N']] = SR_N
        y0[self.equations.idx['Nb']] = SR_N
        y0[self.equations.idx['dNbdt']] = SR_dNdt


class IC_RST_b(object):
    def __init__(self, backgound, cs=1):
        self.backgound = backgound  # the equation should have track_b=True
        self.V = self.backgound.potential.V
        self.dV = self.backgound.potential.dV
        self.K = self.backgound.K
        self.cs = cs

        equations = InflationEquations(K=self.K, potential=self.backgound.potential, track_eta=False, track_b=True)
        """get initial condition of Nb and dNbdt"""
        ic_SR = SlowRowIC(self.backgound, equations)
        ev = [InflationEvent(equations, -1, terminal=True)]   # end at inflation start
        backwards = solve(ic=ic_SR, events=ev)
        self.Nb_i = backwards.Nb[-1]
        self.dNbdt_i = backwards.dNbdt[-1]


    # #########################################################################
    # Functions needed for defining IC of R and dRdt
    # #########################################################################
    def kappa2(self, k, K):
        return k**2 + k * K * (K + 1) - 3 * K

    def epsilon(self, a, da, dphi):
        return 0.5* (a* dphi/da)**2

    def phi_dot_IC(self, phi): # when start of inflation, V = phi_prime**2
        return - np.sqrt(self.V(phi))

    def N_dot_IC(self, K, phi, N): # when start of inflation
        return np.sqrt( 1.0/2.0 * self.phi_dot_IC(phi)**2 - K* np.exp(-2.0*N) )

    def z_g(self, dphi, b, db):
        return dphi * b**2 / self.cs / db

    def z_g_dot(self, dphi, ddphi, b, db, ddb):
        return (ddphi*b**2 + 2.*b*db*dphi)/(self.cs*db) - (self.cs*ddb*dphi*b**2)/(self.cs*db)**2

    def g(self, a, da, b, db):
        return (a/b)**2 * (db/da)

    def dg(self, a, a_dot, a_2dot, b, b_dot, b_2dot):
        return 2.*(b_dot/a_dot) * (a*a_dot*b**2-b*b_dot*a**2)/ b**4 + (a/b)**2 * (b_2dot*a_dot-b_dot*a_2dot)/a_dot**2

    def f(self, K, a, da, dda, b, db, ddb):
        return a* da* self.dg(a, da, dda, b, db, ddb) / K + self.g(a, da, b, db)

    def zeta_IC(self, dphi, b, db, k, K):
        return 1./ ( 2*self.cs*self.z_g(dphi, b, db)**2 * self.kappa2(k, K)**0.5 )**0.5

    def zeta_dot_IC(self, dphi, ddphi, a, da, b, db, ddb, k, K):
        return  (- 1.j*self.kappa2(k,K)**0.5/a + da/a - self.z_g_dot(dphi, ddphi, b, db, ddb)/self.z_g(dphi, b, db) ) * self.zeta_IC(dphi, b, db, k, K)

    def R_PPS(self, k, K, zeta, dzeta, a, da, dda, b, db, ddb):
        return zeta/ self.g(a, da, b, db) - K* (a/da)* self.f(K, a, da, dda, b, db, ddb)*dzeta/ ( self.g(a, da, b, db)**2 *self.cs**2 *self.kappa2(k,K) )

    def R_dot_PPS(self, k, K, zeta, dzeta, dphi, a, da, dda, b, db, ddb):
        return 1./ ( self.g(a, da, b, db)*self.kappa2(k, K)) *( -K**2* self.f(K, a, da, dda, b, db, ddb)/(self.g(a, da, b, db)*self.cs**2*da**2) + (self.kappa2(k,K)+K*self.epsilon(a, da, dphi)) ) * dzeta + K*zeta/(a*da*self.g(a, da, b, db))
    
    def get_R_IC(self, k):
        # Return IC of R
        K, phi, N, Nb, dNb = self.K, self.backgound.phi[0], self.backgound.N[0], self.Nb_i, self.dNbdt_i 
        b = np.exp(Nb)
        db = b* dNb
        ddN = -1.0/2.0 * self.phi_dot_IC(phi)**2 + K* np.exp(-2*N)
        ddNb = dNb**2 + ddN - self.N_dot_IC(K, phi, N)**2 - K*np.exp(-2*N)
        ddb = b * (ddNb + (db/b)**2)
        dda = np.exp(N) * (ddN + self.N_dot_IC(K, phi, N)**2)

        zeta_IC = self.zeta_IC(self.phi_dot_IC(phi), b, db, k, K)
        zeta_dot_IC = self.zeta_dot_IC(self.phi_dot_IC(phi), - 3.0*self.N_dot_IC(K, phi, N)*self.phi_dot_IC(phi) - self.dV(phi), np.exp(N), np.exp(N)*self.N_dot_IC(K, phi, N), b, db, ddb, k, K)
        R_IC = self.R_PPS(k, K, zeta_IC, zeta_dot_IC, np.exp(N), np.exp(N)*self.N_dot_IC(K, phi, N), dda, b, db, ddb)
        R_dot_IC =  self.R_dot_PPS(k, K, zeta_IC, zeta_dot_IC, self.phi_dot_IC(phi), np.exp(N), np.exp(N)*self.N_dot_IC(K, phi, N), dda, b, db, ddb)
        
        return [R_IC, R_dot_IC]
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import optimize
from astropy import units as u
from astropy.constants import c
import PPS_models

# Planck constant
t_p = 1.0 # second
l_p = 1.0 # km
m_p = 1.0 # reduced planck mass = \sqrt(c*hbar/8/pi/G) ??

# constants
sigma = 1.1424624667273704e-05
phi_0 = 5.815354804140279
V0 = m_p**4
N_0 = 12.0 + np.log(sigma)  # N_paper = ln(a/lp) = 10.0
Nb_0 = 0.6971946729724704
Nb_dot_0 = 0.7957397827643071
b_0 = np.log(Nb_0)
b_dot_0 = Nb_dot_0 * b_0

# Cross horizon (k=0.05 Mpc^-1)
H0 = 70 * u.km/u.s/u.Mpc
K = -1.0
if K>0: Omega_K = - 0.01
else: Omega_K = 0.01
a0 = c * np.sqrt(-K/Omega_K)/H0


#### Define functions for ODEs

def potential(phi):
    #return V0 * phi**(4./3.)
    return V0 * (1.0 - np.exp(- math.sqrt(2.0/3.0)* phi/m_p ) )**2

def potential_prime(phi):
    #return V0 * 4./3. * phi**(1./3.)
    return 2.0* math.sqrt(2.0/3.0) * V0/m_p *(1.0- np.exp(- math.sqrt(2.0/3.0)* phi/m_p)) *np.exp(- math.sqrt(2.0/3.0)* phi/m_p)

def phi_dot_IC(phi): # when start of inflation, V = phi_prime**2
    return - math.sqrt(potential(phi))

def b_dot(N, N_dot, X, X_dot):
    return N_dot* np.exp(N)* X + np.exp(N) * X_dot

def N_dot_IC(phi, N): # when start of inflation
    return math.sqrt( 1.0/(2.0*m_p**2) * phi_dot_IC(phi)**2 - K* np.exp(-2.0*N) )

def P_R(phi_dot, N_dot): # N_dot = H
    return ( N_dot**2/2.0/np.pi/phi_dot )**2

def X_dot(N, N_dot, b, b_prime):
    return b_prime/np.exp(N)**2 - N_dot*b/np.exp(N)


"""
# Define Nb_0 and Nb_dot_0 with R=zeta at t0
g0 = 1
f0=0
N_2dot_0 = -1.0/(2.0*m_p**2) * phi_dot_IC(phi_0)**2 + K* np.exp(-2*N_0)
b_0 = np.exp(N_0)* (N_2dot_0- K/np.exp(N_0)**2)/ (K*(f0-g0)/np.exp(N_0)**2+g0*N_2dot_0)
b_dot_0 = g0*N_dot_IC(phi_0, N_0)*b_0**2/np.exp(N_0)
Nb_0 = np.log(b_0)
Nb_dot_0 = b_dot_0/b_0
"""

# define ODEs
def odes(t, y):
    phi = y[0]
    N = y[1]
    dphidt = y[2]
    dNdt = y[3]
    tau = y[4]
    dtaudt = 1./np.exp(N)
    #Nb = y[5]
    #dNbdt = y[6]
    R = y[5]
    dRdt = y[6]

    # define each ODE
    d2phidt = - 3.0*dNdt*dphidt - potential_prime(phi)
    d2Ndt = -1.0/(2.0*m_p**2) * dphidt**2 + K* np.exp(-2*N) 
    #d2Nbdt = dNbdt**2 + d2Ndt - dNdt**2 - K*np.exp(-2*N)  

    # Define R
    a = np.exp(N)
    dadt = np.exp(N)*dNdt
    a_2dot = a* (d2Ndt + dNdt**2)
    z_PPS = PPS_models.z_PPS(a, dadt, dphidt)
    z_dot_PPS = PPS_models.z_dot_PPS(a, dadt, a_2dot, dphidt, d2phidt)
    epsilon = PPS_models.epsilon(a, dadt, dphidt)
    #d2Rdt = ( -2*a**2 * ( z_dot_PPS/z_PPS*PPS_models.keppaD_2(k_R,K) + K*dNdt*epsilon )* dRdt \
     #   - ( K*(1.+epsilon-2./dNdt*z_dot_PPS/z_PPS)*PPS_models.keppaD_2(k_R,K) - K**2 *epsilon +PPS_models.keppaD_2(k_R,K)**2 )*R ) \
      #      / ( a**2 * (PPS_models.keppaD_2(k_R,K)+ K* epsilon) )
    d2Rdt = - ( ( (dNdt+ 2.*z_dot_PPS/z_PPS)*PPS_models.keppaD_2(k_R,K) + 3*K*dNdt*epsilon )*dRdt + ( K*(1.+epsilon-2.*z_dot_PPS/z_PPS/dNdt)*PPS_models.keppaD_2(k_R,K) - K**2*epsilon +PPS_models.keppaD_2(k_R,K)**2 ) * R/a**2  ) / (PPS_models.keppaD_2(k_R,K)+ K*epsilon)

    return [dphidt, dNdt, d2phidt, d2Ndt, dtaudt, dRdt, d2Rdt]


### Define events 

# inflating event
def inflating(t,y):
    phi = y[0]
    dphidt = y[2]
    return dphidt**2 - potential(phi)

inflating.terminal = True
inflating.direction = 1

# find the time when Big Bang started
def BBstart(t, y):
    N = y[1]
    a = np.exp(N)
    return a - 1.e-6

BBstart.terminal = True
BBstart.direction = -1

P_R_list = []  # scalar PPS
#k_array = np.logspace(-5, -1, base=10, num=300) # k=[1.e-4~1.e-1] is the observable range.
k_array = [1.e-1]

for k in k_array:

    # solve R
    k_R_physical = k/u.Mpc
    k_R = (k_R_physical*a0).si.value
    kappa_R_2 = k_R**2 + k_R*K * (K+1) - 3*K 
    D_2 = -k_R**2 + 3*K

    # solve ODEs to get solution of BG variables
    d2Ndt_IC = -1.0/(2.0*m_p**2) * phi_dot_IC(phi_0)**2 + K* np.exp(-2*N_0)
    d2Nbdt_IC = Nb_dot_0**2 + d2Ndt_IC - N_dot_IC(phi_0, N_0)**2 - K*np.exp(-2*N_0)
    d2bdt_0 = b_0 * (d2Nbdt_IC+ (b_dot_0/b_0)**2)
    d2adt_0 = np.exp(N_0) * (d2Ndt_IC+ N_dot_IC(phi_0, N_0)**2)

    zeta_IC = PPS_models.zeta_IC(phi_dot_IC(phi_0), b_0, b_dot_0, k_R, K)
    zeta_dot_IC = PPS_models.zeta_dot_IC(phi_dot_IC(phi_0), - 3.0*N_dot_IC(phi_0, N_0)*phi_dot_IC(phi_0) - potential_prime(phi_0), np.exp(N_0), np.exp(N_0)*N_dot_IC(phi_0, N_0), b_0, b_dot_0, d2bdt_0, k_R, K)
    R_IC = PPS_models.R_PPS(k_R, K, zeta_IC, zeta_dot_IC, np.exp(N_0), np.exp(N_0)*N_dot_IC(phi_0, N_0), d2adt_0, b_0, b_dot_0, d2bdt_0)
    R_dot_IC =  PPS_models.R_dot_PPS(k_R, K, zeta_IC, zeta_dot_IC, phi_dot_IC(phi_0), np.exp(N_0), np.exp(N_0)*N_dot_IC(phi_0, N_0), d2adt_0, b_0, b_dot_0, d2bdt_0)

    y0 = [phi_0, N_0, phi_dot_IC(phi_0), N_dot_IC(phi_0, N_0), 0.0, R_IC, R_dot_IC]
    sol_inf = solve_ivp(odes, [0, 1.e8], y0, method='RK45', events=inflating) 
    sol_KD = solve_ivp(odes, [0, -1.e5], y0, method='RK45', events=BBstart)

    # Combine solutions of KD and inflation together
    t_tot = np.concatenate((np.flipud(sol_KD.t), sol_inf.t), axis=0)
    sol_tot = np.concatenate((np.fliplr(sol_KD.y), sol_inf.y), axis=1)

    # Shift t and tau such that the universe start from t=0 and tau=0
    for i in range(len(t_tot)):
        t_tot[i] = t_tot[i] - sol_KD.t[-1]
        sol_tot[4][i] = sol_tot[4][i] - sol_KD.y[4][-1]

    # calculate equation of state (w = p/rho)
    w_phi = []
    for i in range(len(sol_tot[0])):
        p_phi = 0.5* sol_tot[2][i]**2 - potential(sol_tot[0][i]) 
        rho_phi = 0.5* sol_tot[2][i]**2 + potential(sol_tot[0][i])
        w_phi.append(p_phi/rho_phi)

    # calculate Hubble Horizon
    Hubble_Horizon = []
    for i in range(len(sol_tot[0])):
        Hubble_Horizon.append( a0.to(u.Mpc).value / (sol_tot[3][i]*np.exp(sol_tot[1][i])) )
    
    # Scaling all variables
    for i in range(len(t_tot)):
        t_tot[i] = t_tot[i] / sigma
        sol_tot[1][i] = sol_tot[1][i] - np.log(sigma)  # e-folding (N)
        sol_tot[2][i] = sol_tot[2][i] * sigma          # phi_dot
        sol_tot[3][i] = sol_tot[3][i] * sigma          # N_dot
        sol_tot[4][i] = sol_tot[4][i]                  # tau
        #sol_tot[5][i] = sol_tot[5][i] - np.log(sigma)  # Nb
        #sol_tot[6][i] = sol_tot[6][i] * sigma          # Nb_dot
        sol_tot[5][i] = sol_tot[5][i] * sigma          # R
        sol_tot[6][i] = sol_tot[6][i] * sigma**2       # R_dot
    
    # calculate P_R:
    P_R = k_R**3/ (2.*np.pi**2) * abs(sol_tot[5][-1])**2
    P_R_list.append(np.log(1.e10*P_R))
    #print('t_end='+str(t_tot[-1]))
    #print('k, P_R='+str(k)+','+str(P_R))
    """
    # Plot variables
    #plt.plot(t_tot, sol_tot[1], label='a')
    plt.plot(t_tot, sol_tot[5], label='R')
    plt.xscale('log')
    #plt.yscale('log')
    #plt.xlim([0, 1])
    #plt.ylim([-1.e-1, 2.e-5])
    plt.xlabel('N')
    plt.ylabel('phi_dot')
    plt.title('N - phi_dot')
    plt.legend()
    plt.show()
    """
    
"""
plt.plot(k_array, P_R_list)
plt.xscale('log')
plt.xlabel('k(Mpc^-1)')
plt.ylabel('log(10^10*P_R(k))')
plt.title('k - P_R(k)')
plt.show()
"""
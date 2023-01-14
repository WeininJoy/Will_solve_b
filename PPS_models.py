import numpy as np

cs = 1

def keppaD_2(k, K):
    if K==0 or K==-1:
        return k**2 - 3*K
    if K==1:
        return k*(k+2) - 3*K  ## k > 2, k\in Z
    else:
        raise ValueError("K should be 0, 1 or -1.")

def epsilon(a, a_dot, phi_dot):
    return 0.5* (a* phi_dot/a_dot)**2

def z_PPS(a, a_dot, phi_dot):
    return a**2 * phi_dot/ a_dot
    
def z_dot_PPS(a, a_dot, a_2dot, phi_dot, phi_2dot):
    return 2*a*phi_dot + a**2*phi_2dot/a_dot - (a/a_dot)**2 * a_2dot*phi_dot

def z_g(phi_dot, b, b_dot):
    return phi_dot * b**2 / cs / b_dot

def z_g_dot(phi_dot, phi_2dot, b, b_dot, b_2dot):
    return (phi_2dot*b**2 + 2.*b*b_dot*phi_dot)/(cs*b_dot) - (cs*b_2dot*phi_dot*b**2)/(cs*b_dot)**2

def g(a, a_dot, b, b_dot):
    return (a/b)**2 * (b_dot/a_dot)

def g_dot(a, a_dot, a_2dot, b, b_dot, b_2dot):
    return 2.*(b_dot/a_dot) * (a*a_dot*b**2-b*b_dot*a**2)/ b**4 + (a/b)**2 * (b_2dot*a_dot-b_dot*a_2dot)/a_dot**2

def f(K, a, a_dot, a_2dot, b, b_dot, b_2dot):
    return a* a_dot* g_dot(a, a_dot, a_2dot, b, b_dot, b_2dot) / K + g(a, a_dot, b, b_dot)

def zeta_IC(phi_dot, b, b_dot, k, K):
    return 1./ ( 2*cs*z_g(phi_dot, b, b_dot)**2 * keppaD_2(k, K)**0.5 )**0.5

def zeta_dot_IC(phi_dot, phi_2dot, a, a_dot, b, b_dot, b_2dot, k, K):
    return  (- 1.j*keppaD_2(k,K)**0.5/a + a_dot/a - z_g_dot(phi_dot, phi_2dot, b, b_dot, b_2dot)/z_g(phi_dot, b, b_dot) ) * zeta_IC(phi_dot, b, b_dot, k, K)

def R_PPS(k, K, zeta, zeta_dot, a, a_dot, a_2dot, b, b_dot, b_2dot):
    return zeta/ g(a, a_dot, b, b_dot) - K* (a/a_dot)* f(K, a, a_dot, a_2dot, b, b_dot, b_2dot)*zeta_dot/ ( g(a, a_dot, b, b_dot)**2 *cs**2 *keppaD_2(k,K) )

def R_dot_PPS(k, K, zeta, zeta_dot, phi_dot, a, a_dot, a_2dot, b, b_dot, b_2dot):
    return 1./ ( g(a, a_dot, b, b_dot)*keppaD_2(k, K)) *( -K**2* f(K, a, a_dot, a_2dot, b, b_dot, b_2dot)/(g(a, a_dot, b, b_dot)*cs**2*a_dot**2) + (keppaD_2(k,K)+K*epsilon(a, a_dot, phi_dot)) ) * zeta_dot + K*zeta/(a*a_dot*g(a, a_dot, b, b_dot))
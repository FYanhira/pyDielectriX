import numpy as np
import matplotlib.pyplot as plt

# ========================
# Modelos dieléctricos
# ========================

def debye(f, eps_inf, eps_s, tau):
    w = 2 * np.pi * f
    denom = 1 + (w * tau)**2
    eps_real = eps_inf + (eps_s - eps_inf) / denom
    eps_imag = (eps_s - eps_inf) * w * tau / denom
    return eps_real, eps_imag

def cole_cole(f, eps_inf, eps_s, tau, alpha):
    w = 2 * np.pi * f
    delta_eps = eps_s - eps_inf
    A1 = (w * tau)**(1 - alpha) * np.cos((1 - alpha) * np.pi / 2)
    A2 = (w * tau)**(1 - alpha) * np.sin((1 - alpha) * np.pi / 2)
    denom = (1 + A1)**2 + A2**2
    eps_real = eps_inf + delta_eps * (1 + A1) / denom
    eps_imag = delta_eps * A2 / denom
    return eps_real, eps_imag

def cole_davidson(f, eps_inf, eps_s, tau, alpha):
    w = 2 * np.pi * f
    delta_eps = eps_s - eps_inf
    wtau = w * tau
    mag = (1 + wtau**2) ** (-alpha / 2)
    theta = np.arctan(wtau)
    eps_real = eps_s - delta_eps * mag * np.cos(alpha * theta)
    eps_imag = delta_eps * mag * np.sin(alpha * theta)
    return eps_real, eps_imag

def havriliak_negami(f, eps_inf, eps_s, tau, alpha, beta):
    w = 2 * np.pi * f
    delta_eps = eps_s - eps_inf
    wtau_alpha = (w * tau)**alpha
    
    R = 1 + wtau_alpha * np.cos(np.pi * alpha / 2)
    I = wtau_alpha * np.sin(np.pi * alpha / 2)
    
    M = np.sqrt(R**2 + I**2)
    theta = np.arctan2(I, R)  # más seguro que arctan(I/R)
    
    eps_real = eps_inf + delta_eps * (M**(-beta)) * np.cos(beta * theta)
    eps_imag = delta_eps * (M**(-beta)) * np.sin(beta * theta)
    
    return eps_real, eps_imag


def fractional_1(f, eps_inf, eps_s, tau, alpha):
    w = 2 * np.pi * f
    A1 = (w * tau) ** (-alpha) * np.cos(alpha * np.pi / 2)
    A2 = (w * tau) ** (-alpha) * np.sin(alpha * np.pi / 2)
    denom = (1 + A1)**2 + A2**2
    eps_real = eps_s - (eps_s - eps_inf) * (1 + A1) / denom
    eps_imag = (eps_s - eps_inf) * A2 / denom
    return eps_real, eps_imag

def fractional_2(f, eps_inf, eps_s, tau_alpha, tau_beta, alpha, beta):
    w = 2 * np.pi * f
    B1 = (w * tau_alpha) ** (-alpha) * np.cos(alpha * np.pi / 2) + \
         (w * tau_beta) ** (-beta) * np.cos(beta * np.pi / 2)
    B2 = (w * tau_alpha) ** (-alpha) * np.sin(alpha * np.pi / 2) + \
         (w * tau_beta) ** (-beta) * np.sin(beta * np.pi / 2)
    denom = (1 + B1)**2 + B2**2
    eps_real = eps_s - ((eps_s - eps_inf) * (1 + B1)) / denom
    eps_imag = ((eps_s - eps_inf) * B2) / denom
    return eps_real, eps_imag

# ========================
# Graficar Cole-Cole Plot
# ========================

# Frecuencia logarítmica
f = np.logspace(0, 6, 500)  # 1 Hz a 1 MHz

# Parámetros teóricos de ejemplo
eps_inf, eps_s, tau = 2.0, 10.0, 1e-3
alpha, beta = 0.4, 0.75

# Calcular cada modelo
epsr1, epsi1 = debye(f, eps_inf, eps_s, tau)
epsr2, epsi2 = cole_cole(f, eps_inf, eps_s, tau, alpha)
epsr3, epsi3 = cole_davidson(f, eps_inf, eps_s, tau, beta)
epsr4, epsi4 = havriliak_negami(f, eps_inf, eps_s, tau, alpha, beta)
epsr5, epsi5 = fractional_1(f, eps_inf, eps_s, tau, alpha)
epsr6, epsi6 = fractional_2(f, eps_inf, eps_s, tau, tau*2, alpha, beta)

# Plot
plt.figure(figsize=(7,6))
plt.plot(epsr1, epsi1, label="Debye")
plt.plot(epsr2, epsi2, label="Cole-Cole")
plt.plot(epsr3, epsi3, label="Cole-Davidson")
plt.plot(epsr4, epsi4, label="Havriliak-Negami")
plt.plot(epsr5, epsi5, label="Fractional-1 RC")
plt.plot(epsr6, epsi6, label="Fractional-2 RC")

plt.xlabel("ε′ (Real)")
plt.ylabel("ε″ (Imag)")
plt.title("Cole–Cole Plot of Dielectric Models")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parámetros base
eps_inf = 2.0
eps_s   = 10.0
delta_eps = eps_s - eps_inf
tau = 1e-4   # tiempo característico (ajustado)
alpha = 0.4  # para cole-cole y HN
beta  = 0.75  # para cole-davidson y HN

# Frecuencia (log)
freq = np.logspace(1, 8, 400)   # ampliado rango de 10 Hz a 100 MHz
w = 2 * np.pi * freq
jw_tau = 1j * w * tau

# ---- Modelos ----
# 1. Debye
eps1 = eps_inf + delta_eps / (1 + jw_tau)

# 2. Cole-Cole
eps2 = eps_inf + delta_eps / (1 + (jw_tau)**(1-alpha))

# 3. Cole-Davidson
eps3 = eps_inf + delta_eps / (1 + jw_tau)**beta

# 4. Havriliak-Negami
eps4 = eps_inf + delta_eps / (1 + (jw_tau)**(1-alpha))**beta

# 5. Fractional-1 RC 
alpha1 = 0.4
eps5 = (eps_s + (eps_inf * (jw_tau)**alpha1)) / (1 + (jw_tau)**alpha1)

# 6. Fractional-2 RC 
alpha2, beta2 = 0.4, 0.75
eps6 = (eps_inf + (eps_s * ((jw_tau)**(-alpha2) + (jw_tau)**(-beta2)))) / (1 + ((jw_tau)**(-alpha2) + (jw_tau)**(-beta2)))

# ---- Partes reales e imaginarias ----
def split(eps):
    return np.real(eps), -np.imag(eps)  # signo negativo para que e'' >= 0

epsr1, epsi1 = split(eps1)
epsr2, epsi2 = split(eps2)
epsr3, epsi3 = split(eps3)
epsr4, epsi4 = split(eps4)
epsr5, epsi5 = split(eps5)
epsr6, epsi6 = split(eps6)

# ---- Gráfica (a) Parte real ----
plt.figure(figsize=(6,5))
plt.semilogx(freq, epsr1, label="Debye", linewidth=2)
plt.semilogx(freq, epsr2, label="Cole-Cole", linewidth=2)
plt.semilogx(freq, epsr3, label="Cole-Davidson", linewidth=2)
plt.semilogx(freq, epsr4, label="Havriliak-Negami", linewidth=2)
plt.semilogx(freq, epsr5, label="Fractional_1CR", linewidth=2)
plt.semilogx(freq, epsr6, label="Fractional_2CR", linewidth=2)
plt.xlabel("Frequency [Hz]", fontsize=16, fontweight="bold")
plt.ylabel("ε' (Real part)", fontsize=16, fontweight="bold")
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(prop={'weight': 'bold', 'size': 12})
plt.grid(True)
plt.tight_layout()
plt.savefig("real_part.png", dpi=300)
plt.show()

# ---- Gráfica (b) Parte imaginaria ----
plt.figure(figsize=(6,5))
plt.semilogx(freq, epsi1, label="Debye", linewidth=2)
plt.semilogx(freq, epsi2, label="Cole-Cole", linewidth=2)
plt.semilogx(freq, epsi3, label="Cole-Davidson", linewidth=2)
plt.semilogx(freq, epsi4, label="Havriliak-Negami", linewidth=2)
plt.semilogx(freq, epsi5, label="Fractional_1CR", linewidth=2)
plt.semilogx(freq, epsi6, label="Fractional_2CR", linewidth=2)
plt.xlabel("Frequency [Hz]", fontsize=16, fontweight="bold")
plt.ylabel("ε'' (Imaginary part)",fontsize=16, fontweight="bold")
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(prop={'weight': 'bold', 'size': 12})
plt.grid(True)
plt.tight_layout()
plt.savefig("imag_part.png", dpi=300)
plt.show()

# ---- Gráfica (c) Tan delta ----
plt.figure(figsize=(6,5))
plt.semilogx(freq, epsi1/epsr1, label="Debye", linewidth=2)
plt.semilogx(freq, epsi2/epsr2, label="Cole-Cole", linewidth=2)
plt.semilogx(freq, epsi3/epsr3, label="Cole-Davidson", linewidth=2)
plt.semilogx(freq, epsi4/epsr4, label="Havriliak-Negami", linewidth=2)
plt.semilogx(freq, epsi5/epsr5, label="Fractional_1CR", linewidth=2)
plt.semilogx(freq, epsi6/epsr6, label="Fractional_2CR", linewidth=2)
plt.xlabel("Frequency [Hz]", fontsize=16, fontweight="bold")
plt.ylabel("Tan δ = ε''/ε'",fontsize=16, fontweight="bold")
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(prop={'weight': 'bold', 'size': 12})
plt.grid(True)
plt.tight_layout()
plt.savefig("tan_delta.png", dpi=300)
plt.show()

# ---- Gráfica (d) Cole–Cole ----
plt.figure(figsize=(6,5))
plt.plot(epsr1, epsi1, label="Debye", linewidth=2)
plt.plot(epsr2, epsi2, label="Cole-Cole", linewidth=2)
plt.plot(epsr3, epsi3, label="Cole-Davidson", linewidth=2)
plt.plot(epsr4, epsi4, label="Havriliak-Negami", linewidth=2)
plt.plot(epsr5, epsi5, label="Fractional_1CR", linewidth=2)
plt.plot(epsr6, epsi6, label="Fractional_2CR", linewidth=2)
plt.xlabel("ε'", fontsize=16, fontweight="bold")
plt.ylabel("ε''", fontsize=16, fontweight="bold")
plt.ylim(-0.5, None)
plt.axis('equal')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(prop={'weight': 'bold', 'size': 12})
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig("cole_cole.png", dpi=300)
plt.show()


### models/IonicModulus.py

from .base_model import BaseModel
import numpy as np
from lmfit import Model, Parameters

class IonicModulusModel(BaseModel):
    def __init__(self):
        super().__init__("Ionic_Modulus")
        self.params_init = {
            'eps_inf1': (None, None, None),
            'eps_s1': (None, None, None),
            'eps_inf2': (None, None, None),
            'eps_s2': (None, None, None),
            'tau_alpha': (1e-4, 1e-10, 1),
            'tau_beta': (1e-4, 1e-10, 1),
            'tau_gamma': (1e-4, 1e-10, 1),
            'alpha': (0.5, 0.01, 1.0),
            'beta': (0.5, 0.01, 1.0),
            'gamma': (0.5, 0.01, 1.0),
        }

    def get_params(self):
        return self.params_init

    def set_auto_params_from_data(self, eps_real, n_points=5):
        flex_factor = 10

        avg_low_freq = np.mean(eps_real[-n_points:])
        avg_mid_low = np.mean(eps_real[-n_points:])
        avg_mid_high = np.mean(eps_real[-n_points:])
        avg_high_freq = np.mean(eps_real[-n_points:])

        self.params_init['eps_s1'] = (avg_low_freq, avg_low_freq/flex_factor, avg_low_freq*flex_factor)
        self.params_init['eps_s2'] = (avg_mid_low, avg_mid_low/flex_factor, avg_mid_low*flex_factor)
        self.params_init['eps_inf1'] = (avg_mid_high, avg_mid_high/flex_factor, avg_mid_high*flex_factor)
        self.params_init['eps_inf2'] = (avg_high_freq, avg_high_freq/flex_factor, avg_high_freq*flex_factor)

    def model_function(self, f, eps_inf1, eps_s1, eps_inf2, eps_s2,
                       tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma):
        """
        Devuelve el módulo eléctrico complejo M* = 1/ε*
        """
        w = 2*np.pi*f

        A1 = ((w*tau_alpha)**(-alpha))*np.cos(alpha*np.pi/2) + ((w*tau_beta)**(-beta))*np.cos(beta*np.pi/2)
        A2 = ((w*tau_alpha)**(-alpha))*np.sin(alpha*np.pi/2) + ((w*tau_beta)**(-beta))*np.sin(beta*np.pi/2)
        A3 = ((w*tau_gamma)**(-gamma))*np.cos(gamma*np.pi/2)
        A4 = ((w*tau_gamma)**(-gamma))*np.sin(gamma*np.pi/2)

        # ε′ y ε″
        eps_real = eps_s1 - ((eps_s1 - eps_inf1)*(1+A1))/((1+A1)**2 + A2**2) + (eps_s2 - eps_inf2)*A3
        eps_imag = ((eps_s1 - eps_inf1)*A2)/((1+A1)**2 + A2**2) + (eps_s2 - eps_inf2)*A4

        # Módulo eléctrico M* = 1/ε*
        denom = eps_real**2 + eps_imag**2
        M_real = eps_real / denom
        M_imag = eps_imag / denom

        return M_real + 1j*M_imag

    def fit(self, f, M_real_exp, M_imag_exp, user_params=None):
        def model_real(f, eps_inf1, eps_s1, eps_inf2, eps_s2,
                       tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma):
            return np.real(self.model_function(f, eps_inf1, eps_s1, eps_inf2, eps_s2,
                                               tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma))

        def model_imag(f, eps_inf1, eps_s1, eps_inf2, eps_s2,
                       tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma):
            return np.imag(self.model_function(f, eps_inf1, eps_s1, eps_inf2, eps_s2,
                                               tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma))

        model_real_fit = Model(model_real)
        model_imag_fit = Model(model_imag)

        params = Parameters()

        if user_params:
            for key in self.params_init:
                _, minval, maxval = self.params_init[key]
                val_str = user_params[key]['val']
                val = float(val_str) if val_str not in ('', None) else None
                if val is None:
                    val = self.params_init[key][0]
                params.add(key, value=val, min=minval, max=maxval)
        else:
            for key, (val, minval, maxval) in self.params_init.items():
                if val is None:
                    raise ValueError(f"Missing automatic value for parameter '{key}'")
                params.add(key, value=val, min=minval, max=maxval)

        result_real = model_real_fit.fit(M_real_exp, f=f, params=params)
        result_imag = model_imag_fit.fit(M_imag_exp, f=f, params=result_real.params)

        self.params = result_imag.params
        return result_real, result_imag

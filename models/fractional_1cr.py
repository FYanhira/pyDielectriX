### models/F1CR.py

from .base_model import BaseModel
import numpy as np
from lmfit import Model, Parameters

class Fractional1CRModel(BaseModel):
    def __init__(self):
        super().__init__("Fracc-1CR")
        self.params_init = {
            'eps_inf': (None, None, None),
            'eps_s': (None, None, None),
            'tau': (1e-4, 1e-6, 1e-1),
            'gamma': (0.5, 0.01, 1.0),
        }

    def get_params(self):
        return self.params_init

    def set_auto_params_from_data(self, eps_real, n_points=5):
        """
        Calcula eps_s y eps_inf automáticos y sus rangos en base a datos.
        """
        flex_factor = 2

        avg_low_freq = np.mean(eps_real[:n_points])   # baja frecuencia
        avg_high_freq = np.mean(eps_real[-n_points:]) # alta frecuencia

        # Definir valores y rangos calculados dinámicamente
        self.params_init['eps_s'] = (
            avg_low_freq,
            avg_low_freq / flex_factor,
            avg_low_freq * flex_factor
        )
        self.params_init['eps_inf'] = (
            avg_high_freq,
            avg_high_freq / flex_factor,
            avg_high_freq * flex_factor
        )

    def model_function(self, f, eps_inf, eps_s, tau, gamma):
            w = 2 * np.pi * f
            A1 = (w * tau) ** (-gamma) * np.cos(gamma * np.pi / 2)
            A2 = (w * tau) ** (-gamma) * np.sin(gamma * np.pi / 2)

            denom = (1 + A1)**2 + A2**2

            eps_real = eps_s - (eps_s - eps_inf) * (1 + A1) / denom
            eps_imag = (eps_s - eps_inf) * A2 / denom

            return eps_real + 1j * eps_imag

    def fit(self, f, eps_real, eps_imag, user_params=None):
        def model_real(f, eps_inf, eps_s, tau, gamma):
            return np.real(self.model_function(f, eps_inf, eps_s, tau, gamma))

        def model_imag(f, eps_inf, eps_s, tau, gamma):
            return np.imag(self.model_function(f, eps_inf, eps_s, tau, gamma))

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

        result_real = model_real_fit.fit(eps_real, f=f, params=params)
        result_imag = model_imag_fit.fit(eps_imag, f=f, params=result_real.params)

        self.params = result_imag.params
        return result_real, result_imag

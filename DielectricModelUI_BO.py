# --- Lanzador principal ---
# (Debe ir al final del archivo, después de los imports y definiciones)

# (El encabezado de imports y definiciones previas se mantienen igual)
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext  # <<--- agregado
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image
import io

# ---- Optimización Bayesiana auténtica ----
from skopt import gp_minimize
from skopt.space import Real

# ---- Models (as provided by you) ----
from models.debye import DebyeModel
from models.colecole import ColeColeModel
from models.coledavidson import ColeDavidsonModel
from models.havriliak_negami import HavriliakNegamiModel
from models.fractional_1cr import Fractional1CRModel
from models.fractional_2cr import Fractional2CRModel
from models.Ionic_conductivity import IonicModel

from models.Debye_modulus import DebyeModulusModel
from models.Colecole_modulus import ColeColeModulusModel
from models.ColeDavidson_modulus import ColeDavidsonModulusModel
from models.HN_modulus import HNModulusModel
from models.fract_1cr_modulus import Fract1crModulusModel
from models.fract_2cr_modulus import Fract2crModulusModel
from models.Ionic_modulus import IonicModulusModel
from models.MWS_Relax_modulus import MWSModulusModel

# ================== metrics ==================
def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_fit_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    aad = mean_absolute_deviation(y_true, y_pred)
    return r2, mse, aad

# ================== GUI ==================
class DielectricGUI_BO:
    def __init__(self, root):
        self.root = root
        self.root.title("Fractional Dielectric Model Fitter (BO auténtico)")
        self.data = None
        self.freq = None

        # --- state ---
        self.param_inputs = {}
        self.result_texts = {}
        self.model_metrics = {}
        self.normal_fit_params = {}
        self.bayes_fit_params = {}
        self.last_fit_bayes = False
        self.current_canvas = None

        # --- LM error history (for analysis, not user display) ---
        self._lm_error_history = {}  # {model_name: [error_per_iter]}

        # --- models ---
        self.registered_models = {
            'Debye': DebyeModel(),
            'Cole-Cole': ColeColeModel(),
            'Cole-Davidson': ColeDavidsonModel(),
            'Havriliak-Negami': HavriliakNegamiModel(),
            'Fractional-1CR': Fractional1CRModel(),
            'Fractional-2CR': Fractional2CRModel(),
            'Ionic': IonicModel(),
            'Debye_Modulus': DebyeModulusModel(),
            'Cole-Cole_Modulus': ColeColeModulusModel(),
            'Cole-Davidson_Modulus': ColeDavidsonModulusModel(),
            'Havriliak-Negami_Modulus': HNModulusModel(),
            'Fractional-1CR_Modulus': Fract1crModulusModel(),
            'Fractional-2CR_Modulus': Fract2crModulusModel(),
            'Ionic_modulus': IonicModulusModel(),
            'MWS_Modulus': MWSModulusModel(),
        }
        self.permittivity_models = {k:self.registered_models[k] for k in list(self.registered_models.keys())[:7]}
        self.modulus_models      = {k:self.registered_models[k] for k in list(self.registered_models.keys())[7:]}

        # --- layout ---
        self.top_control_frame = tk.Frame(root)
        self.top_control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.load_button = tk.Button(self.top_control_frame, text="Load CSV", command=self.load_csv)
        self.load_button.grid(row=0, column=0, padx=5)

        self.domain_var = tk.StringVar(value="Permittivity")
        self.domain_combo = ttk.Combobox(self.top_control_frame, textvariable=self.domain_var, state="readonly")
        self.domain_combo['values'] = ("Permittivity", "Modulus")
        self.domain_combo.grid(row=0, column=1, padx=10)
        self.domain_combo.bind("<<ComboboxSelected>>", self.update_model_list)

        self.model_frame = tk.LabelFrame(self.top_control_frame, text="Select Models to Fit")
        self.model_frame.grid(row=0, column=2, sticky='nsw', padx=5)
        self.model_checks = {}

        self.params_container = tk.LabelFrame(self.top_control_frame, text="Model Parameters")
        self.params_container.grid(row=0, column=3, sticky='nsew', padx=5)
        self.params_canvas = tk.Canvas(self.params_container, height=220)
        self.params_scrollbar = tk.Scrollbar(self.params_container, orient="vertical", command=self.params_canvas.yview)
        self.params_frame = tk.Frame(self.params_canvas)
        self.params_frame.bind("<Configure>", lambda e: self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all")))
        self.params_canvas.create_window((0,0), window=self.params_frame, anchor="nw")
        self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)
        self.params_canvas.pack(side="left", fill="both", expand=True)
        self.params_scrollbar.pack(side="right", fill="y")

        self.fit_button = tk.Button(self.top_control_frame, text="Fit LM (Levenberg-Marquardt)", command=self.run_fit)
        self.fit_button.grid(row=1, column=0, pady=(5,0), sticky='w')


        # Campo para elegir número de iteraciones de BO
        tk.Label(self.top_control_frame, text="BO Iterations:").grid(row=1, column=1, sticky='e')
        self.bo_iter_var = tk.IntVar(value=30)
        self.bo_iter_entry = tk.Entry(self.top_control_frame, textvariable=self.bo_iter_var, width=5)
        self.bo_iter_entry.grid(row=1, column=2, sticky='w')

        self.fit_bayes_button = tk.Button(self.top_control_frame, text="Bayesian Optimize + Fit", command=self.run_fit_bayes)
        self.fit_bayes_button.grid(row=1, column=3, pady=(5,0), sticky='w')

        self.reset_button = tk.Button(self.top_control_frame, text="Reset to Normal Fit", command=self.reset_to_normal)
        self.reset_button.grid(row=1, column=4, pady=(5,0), sticky='w')





        # Barra de botones secundarios (justo después de los controles principales)
        self.bottom_button_frame = tk.Frame(root)
        self.bottom_button_frame.pack(fill=tk.X, padx=10, pady=(0,5))

        # (Botones secundarios ya están definidos una sola vez más abajo)

        # Área de resultados
        self.result_container = tk.LabelFrame(self.top_control_frame, text="Fit Results")
        self.result_container.grid(row=0, column=5, rowspan=2, sticky='nsew', padx=5)
        self.result_text = scrolledtext.ScrolledText(self.result_container, height=16, width=70, wrap=tk.WORD)
        self.result_text.pack(fill='both', expand=True)

        # Área de gráficas
        self.plot_frame = tk.Frame(root, height=800)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        self.export_button = tk.Button(self.bottom_button_frame, text="Export Results", command=self.export_results)
        self.export_button.pack(side="left", padx=5)

        self.bic_button = tk.Button(self.bottom_button_frame, text="Rank Models by BIC", command=self.run_bic_eval)
        self.bic_button.pack(side="left", padx=5)

        self.fbn_button = tk.Button(self.bottom_button_frame, text="Evaluate Models with FBN", command=self.run_fbn)
        self.fbn_button.pack(side="left", padx=5)

        # Botón para maximizar/restaurar área de gráficas
        self.maximized = False
        self.max_plot_button = tk.Button(self.bottom_button_frame, text="Maximize Plot Area", command=self.toggle_maximize_plot)
        self.max_plot_button.pack(side="right", padx=5)
    def toggle_maximize_plot(self):
        if not self.maximized:
            self.top_control_frame.pack_forget()
            self.bottom_button_frame.pack_forget()
            self.plot_frame.pack_forget()
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            self.max_plot_button.config(text="Restore Controls")
            self.bottom_button_frame.pack(fill=tk.X, padx=10, pady=(0,5))
            self.maximized = True
        else:
            self.plot_frame.pack_forget()
            self.top_control_frame.pack(fill=tk.X, padx=10, pady=5)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            self.max_plot_button.config(text="Maximize Plot Area")
            self.maximized = False


        # (No actualizar lista de modelos al maximizar/restaurar para no perder selección)

    # Métodos utilitarios y de interfaz copiados del original
    def _clear_plots_and_results(self):
        for w in self.plot_frame.winfo_children():
            w.destroy()
        self.result_text.delete("1.0", tk.END)
        self.current_canvas = None

    def _selected_models(self):
        return [name for name, var in self.model_checks.items() if var.get()]

    def _inputs_to_param_dict(self, model_name):
        d = {}
        for key, cells in self.param_inputs[model_name].items():
            try:
                v  = float(cells['val'].get())
                vmin = float(cells['min'].get())
                vmax = float(cells['max'].get())
            except Exception:
                v, vmin, vmax = 0.0, -np.inf, np.inf
            d[key] = {'val': v, 'min': vmin, 'max': vmax}
        return d

    def _apply_params_to_entries(self, model_name, param_map):
        for k, v in param_map.items():
            if model_name in self.param_inputs and k in self.param_inputs[model_name]:
                e = self.param_inputs[model_name][k]['val']
                e.delete(0, tk.END)
                e.insert(0, f"{float(v):.6g}")

    def _safe_fit(self, model_instance, freq, eps_real, eps_imag, param_dict, model_name=None, track_lm_error=False):
        """
        Ajusta el modelo y, si track_lm_error=True y el modelo lo soporta, guarda el historial de error LM en self._lm_error_history[model_name].
        """
        try:
            if hasattr(model_instance, 'params'):
                for k, spec in param_dict.items():
                    try:
                        model_instance.params[k].set(value=spec['val'], min=spec['min'], max=spec['max'])
                    except Exception:
                        pass
            # Solo trackear error LM si se solicita, se da model_name y el modelo tiene fit_real/fit_imag
            if (
                track_lm_error and model_name is not None and
                hasattr(model_instance, 'fit_real') and hasattr(model_instance, 'fit_imag')
            ):
                error_hist_real = []
                error_hist_imag = []
                def cb_real(params, iter, resid, *args, **kwargs):
                    mse = np.mean(resid**2)
                    error_hist_real.append(mse)
                def cb_imag(params, iter, resid, *args, **kwargs):
                    mse = np.mean(resid**2)
                    error_hist_imag.append(mse)
                res_real = model_instance.fit_real(freq, eps_real, param_dict, iter_cb=cb_real)
                res_imag = model_instance.fit_imag(freq, eps_imag, param_dict, iter_cb=cb_imag)
                self._lm_error_history[model_name] = {
                    'real': error_hist_real,
                    'imag': error_hist_imag
                }
                return res_real, res_imag
            else:
                # Si no soporta tracking, solo fit normal (sin historial)
                return model_instance.fit(freq, eps_real, eps_imag, param_dict)
        except Exception as e:
            print(f"Fit error in {type(model_instance).__name__}: {e}")
            return None, None

    def update_model_list(self, event=None):
        for w in self.model_frame.winfo_children():
            w.destroy()
        self.model_checks.clear()
        models_to_show = self.permittivity_models if self.domain_var.get() == "Permittivity" else self.modulus_models
        for model_name in models_to_show:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self.model_frame, text=model_name, variable=var, command=self.update_params)
            chk.pack(anchor='w')
            self.model_checks[model_name] = var
        self.update_params()

    def update_params(self):
        for w in self.params_frame.winfo_children():
            w.destroy()
        self.param_inputs.clear()
        for model_name, var in self.model_checks.items():
            if var.get():
                model_instance = self.registered_models[model_name]
                params = model_instance.get_params()
                tk.Label(self.params_frame, text=f"--- {model_name} Parameters ---", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(6,2))
                self.param_inputs[model_name] = {}
                for key, (val, vmin, vmax) in params.items():
                    f = tk.Frame(self.params_frame)
                    f.pack(anchor='w')
                    tk.Label(f, text=f"{key}:").grid(row=0, column=0, padx=(0,4))
                    e_val = tk.Entry(f, width=10)
                    e_val.insert(0, f"{0.0 if val is None else float(val):.6g}")
                    e_val.grid(row=0, column=1)
                    e_min = tk.Entry(f, width=10)
                    e_min.insert(0, str(vmin))
                    e_min.grid(row=0, column=2)
                    e_max = tk.Entry(f, width=10)
                    e_max.insert(0, str(vmax))
                    e_max.grid(row=0, column=3)
                    self.param_inputs[model_name][key] = {'val': e_val, 'min': e_min, 'max': e_max}

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
        if not file_path:
            return
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        try:
            self.freq = df['frequency'].values
            real = df['real'].values
            imag = df['imag'].values
            self.data = real + 1j*imag
            messagebox.showinfo("Success","CSV loaded successfully.")
            for _name, m in self.registered_models.items():
                if hasattr(m, 'set_auto_params_from_data'):
                    try:
                        m.set_auto_params_from_data(np.real(self.data))
                    except Exception:
                        pass
            self.update_params()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def run_fit(self):
        self._fit_core(use_bayes=False)

    def reset_to_normal(self):
        if not self.normal_fit_params:
            messagebox.showwarning("No normal fit stored","Run a normal fit first.")
            return
        restored_any = False
        for model_name in self._selected_models():
            if model_name in self.normal_fit_params:
                self._apply_params_to_entries(model_name, self.normal_fit_params[model_name])
                restored_any = True
        if not restored_any:
            messagebox.showinfo("Reset","No selected model has a stored normal fit yet.")
            return
        self.last_fit_bayes = False
        self.run_fit()

    def save_plot(self, event=None):
        file_path=filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", ".png"),("All Files",".*")])
        if file_path:
            self.fig.savefig(file_path)

    def export_results(self):
        if not self.model_metrics:
            messagebox.showwarning("No Data", "Run fitting first!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return

        params_source = self.bayes_fit_params if self.last_fit_bayes else self.normal_fit_params

        rows_params = []
        for model_name, metrics in self.model_metrics.items():
            row = {"Model": model_name}
            if model_name in params_source:
                for key, val in params_source[model_name].items():
                    row[key] = float(val)
            if 'rss' in metrics:
                row['RSS_total'] = metrics['rss']
            if 'bic' in metrics:
                row['BIC'] = metrics['bic']
            row.update(metrics)
            rows_params.append(row)
        df_params = pd.DataFrame(rows_params)

        rows_curves = []
        for model_name in params_source:
            model_instance = self.registered_models[model_name]
            param_dict = {k: {'val': v, 'min': -np.inf, 'max': np.inf}
                          for k, v in params_source[model_name].items()}
            res_real, res_imag = self._safe_fit(model_instance, self.freq,
                                                np.real(self.data), np.imag(self.data),
                                                param_dict)
            if res_real and res_imag:
                eps_r_fit = res_real.best_fit
                eps_i_fit = res_imag.best_fit
                with np.errstate(divide='ignore', invalid='ignore'):
                    tan_d_fit = eps_i_fit / eps_r_fit
                for f, r, i, t in zip(self.freq, eps_r_fit, eps_i_fit, tan_d_fit):
                    rows_curves.append({
                        "Model": model_name,
                        "Frequency": f,
                        "Real_fit": r,
                        "Imag_fit": i,
                        "Tan_delta_fit": t
                    })
        df_curves = pd.DataFrame(rows_curves)

        with open(file_path, "w", newline="") as f:
            f.write("=== Parameters and Metrics ===\n")
            df_params.to_csv(f, index=False)
            f.write("\n")
            f.write("=== Fitted Curves ===\n")
            df_curves.to_csv(f, index=False)

        messagebox.showinfo("Export", "Parameters and curves exported in one file.")

    def run_fbn(self):
        if not self.model_metrics:
            messagebox.showwarning("No Metrics","Run fitting first!")
            return
        scores = {}
        for model, m in self.model_metrics.items():
            score_real = m['r2_real'] / (m['mse_real'] + m['aad_real'] + 1e-6)
            score_imag = m['r2_imag'] / (m['mse_imag'] + m['aad_imag'] + 1e-6)
            scores[model] = 0.5 * (score_real + score_imag)
        total = sum(max(s, 0) for s in scores.values())
        if total <= 0:
            messagebox.showinfo("FBN Analysis", "Scores are non-positive; cannot compute probabilities.")
            return
        probs = {k: max(v, 0)/total for k, v in scores.items()}
        sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
        msg = "FBN Model Probabilities:\n"
        for k, v in sorted_probs.items():
            msg += f"{k}: {v*100:.1f}%\n"
        messagebox.showinfo("FBN Analysis", msg)

    def run_bic_eval(self):
        if not self.model_metrics:
            messagebox.showwarning("No Metrics", "Run fitting first!")
            return

        bic_values = []
        for model, metrics in self.model_metrics.items():
            bic = metrics.get('bic', None)
            if bic is not None:
                bic_values.append((model, bic))

        if not bic_values:
            messagebox.showinfo("BIC Ranking", "No BIC values available. Run fitting first.")
            return

        bic_values.sort(key=lambda x: x[1])
        msg = "BIC Model Ranking (lower = better):\n\n"
        for name, value in bic_values:
            n_params = self.model_metrics[name].get('n_params', '–')
            msg += f"{name}:  BIC={value:.3f}   (params={n_params})\n"
        messagebox.showinfo("BIC Ranking", msg)

    def run_fit_bayes(self):
        self._fit_core(use_bayes=True)

    def _fit_core(self, use_bayes=False):
        if self.data is None or self.freq is None:
            messagebox.showwarning("Missing Data","Please load data first.")
            return

        self._clear_plots_and_results()

        fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
        axs = axs.flatten()
        eps_real = np.real(self.data)
        eps_imag = np.imag(self.data)
        tan_delta = eps_imag / np.where(eps_real==0, np.nan, eps_real)

        if self.domain_var.get() == "Permittivity":
            ylabels = ["Real part ε'", "Imaginary part ε''", "Tan δ = ε''/ε'", ("ε'","ε''")]
        else:
            ylabels = ["Real part M'", "Imaginary part M''", "Tan δ = M''/M'", ("M'","M''")]

        axs[0].semilogx(self.freq, eps_real, 'ko', label='Data', markersize=6 )
        axs[0].set_xlabel("ω (rad/s)", fontsize=12, fontweight="bold")
        axs[0].set_ylabel(ylabels[0], fontsize=12, fontweight="bold")
        axs[1].semilogx(self.freq, eps_imag, 'ko', label='Data', markersize=6)
        axs[1].set_xlabel("ω (rad/s)", fontsize=12, fontweight="bold")
        axs[1].set_ylabel(ylabels[1], fontsize=12, fontweight="bold")
        axs[2].semilogx(self.freq, tan_delta, 'ko', label='Data', markersize=6)
        axs[2].set_xlabel("ω (rad/s)", fontsize=12, fontweight="bold")
        axs[2].set_ylabel(ylabels[2], fontsize=12, fontweight="bold")
        axs[3].plot(eps_real, eps_imag, 'ko', label='Data', markersize=6)
        axs[3].set_xlabel(ylabels[3][0], fontsize=12, fontweight="bold")
        axs[3].set_ylabel(ylabels[3][1], fontsize=12, fontweight="bold")

        self.model_metrics.clear()
        if not use_bayes:
            self.normal_fit_params.clear()
        else:
            self.bayes_fit_params.clear()

        selected = self._selected_models()
        if not selected:
            messagebox.showwarning("No models","Please select at least one model.")
            return


        for model_name in selected:
            model_instance = self.registered_models[model_name]
            param_dict = self._inputs_to_param_dict(model_name)

            tiempo_bo = None
            tiempo_lm = None

            bo_error_history_real = []  # Historial de error BO (real)
            bo_error_history_imag = []  # Historial de error BO (imag)
            bo_error_history_total = []  # Historial de error BO (total)

            if use_bayes:
                import time
                search_space = []
                param_names = []
                for k, spec in param_dict.items():
                    param_names.append(k)
                    search_space.append(Real(float(spec['min']), float(spec['max']), name=k))
                def objective(x):
                    trial = {k: {'val': v, 'min': float(param_dict[k]['min']), 'max': float(param_dict[k]['max'])} for k, v in zip(param_names, x)}
                    resR, resI = self._safe_fit(model_instance, self.freq, eps_real, eps_imag, trial)
                    if resR is None or resI is None:
                        err_real = 1e9
                        err_imag = 1e9
                    else:
                        err_real = mean_squared_error(eps_real, resR.best_fit)
                        err_imag = mean_squared_error(eps_imag, resI.best_fit)
                    bo_error_history_real.append(err_real)
                    bo_error_history_imag.append(err_imag)
                    bo_error_history_total.append(err_real + err_imag)
                    return err_real + err_imag
                n_calls = self.bo_iter_var.get() if self.bo_iter_var.get() > 0 else 30
                t0 = time.time()
                res = gp_minimize(objective, search_space, n_calls=n_calls, random_state=0, acq_func="EI")
                tiempo_bo = time.time() - t0
                best_params = {k: v for k, v in zip(param_names, res.x)}
                self._apply_params_to_entries(model_name, best_params)
                for k in param_dict:
                    if k in best_params:
                        param_dict[k]['val'] = float(best_params[k])

            import time
            t0 = time.time()
            # --- Guardar historial de error LM solo si no es BO ---
            res_real, res_imag = self._safe_fit(
                model_instance, self.freq, eps_real, eps_imag, param_dict,
                model_name=model_name, track_lm_error=(not use_bayes)
            )
            tiempo_lm = time.time() - t0

            if res_real is None or res_imag is None:
                self.result_text.insert(tk.END, f"===== {model_name} =====\nFit failed. Check bounds/initial values.\n\n")
                self.result_text.see(tk.END)
                continue

            eps_r_fit = res_real.best_fit
            eps_i_fit = res_imag.best_fit
            with np.errstate(divide='ignore', invalid='ignore'):
                tan_d_fit = eps_i_fit / eps_r_fit

            label = model_name + (" (BO)" if use_bayes else "")
            axs[0].semilogx(self.freq, eps_r_fit, label=label, linewidth=3)
            axs[1].semilogx(self.freq, eps_i_fit, label=label, linewidth=3)
            axs[2].semilogx(self.freq, tan_d_fit, label=label, linewidth=3)
            axs[3].plot(eps_r_fit, eps_i_fit, label=label, linewidth=3)

            fitted_params = {pname: p.value for pname, p in res_imag.params.items()} if hasattr(res_imag, 'params') else {k: v['val'] for k, v in param_dict.items()}

            if use_bayes:
                self.bayes_fit_params[model_name] = dict(fitted_params)
            else:
                self.normal_fit_params[model_name] = dict(fitted_params)

            self.result_text.insert(tk.END, f"===== {model_name}{' (BO)' if use_bayes else ''} =====\n\n")
            for pname, pval in fitted_params.items():
                self.result_text.insert(tk.END, f"{pname} = {float(pval):.6g}\n")

            r2_real, mse_real, aad_real = calculate_fit_metrics(eps_real, eps_r_fit)
            r2_imag, mse_imag, aad_imag = calculate_fit_metrics(eps_imag, eps_i_fit)
            self.result_text.insert(tk.END, f"\n--- Real part metrics ---\nR²={r2_real:.4f}, MSE={mse_real:.4g}, AAD={aad_real:.4g}\n")
            self.result_text.insert(tk.END, f"\n--- Imag part metrics ---\nR²={r2_imag:.4f}, MSE={mse_imag:.4g}, AAD={aad_imag:.4g}\n\n")
            self.result_text.see(tk.END)

            residuals_real = (eps_real - eps_r_fit)
            residuals_imag = (eps_imag - eps_i_fit)
            rss_real = np.sum(np.square(residuals_real))
            rss_imag = np.sum(np.square(residuals_imag))
            rss_total = float(rss_real + rss_imag) + 1e-20
            n_points = len(self.freq)
            N_total = 2 * n_points
            n_params = max(len(fitted_params), 1)
            bic = float(n_params * np.log(N_total) + N_total * np.log(rss_total / N_total))
            self.result_text.insert(tk.END, f"RSS_total={rss_total:.4g}, n_params={n_params}, BIC={bic:.4g}\n\n")
            if tiempo_bo is not None:
                self.result_text.insert(tk.END, f"Tiempo BO: {tiempo_bo:.3f} s\n")
            if tiempo_lm is not None:
                self.result_text.insert(tk.END, f"Tiempo LM: {tiempo_lm:.3f} s\n")
            self.result_text.see(tk.END)
            self.model_metrics[model_name] = {
                'r2_real': r2_real, 'mse_real': mse_real, 'aad_real': aad_real,
                'r2_imag': r2_imag, 'mse_imag': mse_imag, 'aad_imag': aad_imag,
                'rss': rss_total, 'n_params': n_params, 'bic': bic,
                'tiempo_bo': tiempo_bo, 'tiempo_lm': tiempo_lm
            }

            # --- Imprimir historial de error BO en terminal (solo para análisis interno) ---
            if use_bayes:
                print(f"\n=== Historial de error BO por modelo: {model_name} ===")
                print("  Real:", bo_error_history_real)
                print("  Imag:", bo_error_history_imag)
                print("  Total:", bo_error_history_total)

        for ax in axs:
            ax.legend(prop={'weight': 'bold', 'size': 9})  # tamaño reducido
            ax.grid(True)
        fig.tight_layout()
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.fig=fig
        self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_canvas.get_tk_widget().bind("<Button-3>", self.save_plot)
        self.current_canvas.draw()
        self.last_fit_bayes = use_bayes

        # --- Imprimir historial de error LM en terminal (solo para análisis interno) ---
        if not use_bayes and self._lm_error_history:
            print("\n=== Historial de error LM por modelo ===")
            for model, hist in self._lm_error_history.items():
                print(f"Modelo: {model}")
                print("  Real:", hist['real'])
                print("  Imag:", hist['imag'])

# --- Lanzador principal ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DielectricGUI_BO(root)
    root.mainloop()



# ⚡ Dielectric Model GUI (pyDielectriX)

Python application for fitting dielectric spectroscopy data using classic and fractional models, with a user-friendly graphical interface. The main GUI is **DielectricModelUI_BO.py**, supporting both Levenberg-Marquardt (LM) and Bayesian Optimization (BO) fitting, model comparison, and export of results.

---



## ✨ Features
- 🖥️ **Graphical User Interface (GUI):** Tkinter-based interface for loading CSV data, selecting models, and visualizing fits.
- 🧪 **Multiple Dielectric Models:** Debye, Cole-Cole, Cole-Davidson, Havriliak-Negami, fractional CR models, and ionic models, in both permittivity and modulus domains.
- 🤖 **Bayesian Optimization (BO):** Optional Bayesian parameter optimization for robust fitting.
- ⚡ **Levenberg-Marquardt (LM):** Standard non-linear least squares fitting.
- 📊 **Model Comparison:** Fit and compare multiple models simultaneously; overlay results in a single plot.
- 🏷️ **Compact Legends:** Small font legends for clarity when comparing many models.
- 📈 **Instant Plotting:** Real-time visualization of fits, residuals, and model comparison.
- 💾 **Export Results:** Save fitted parameters, metrics, and fitted curves to CSV.
- 🧩 **Extensible:** Modular Python class design for easy addition of new models.

---

## 🗂️ Project Structure
```
├── DielectricModelUI_BO.py   # Main GUI (LM + BO, recommended)
├── DielectricModelUI.py      # Legacy GUI (LM only)
├── Comparative_modelsc.py    # Script for comparing model responses
├── models/                  # All dielectric and modulus model classes
├── optim/                   # Bayesian optimization starter
├── data/                    # Example experimental datasets (CSV)
├── figures/                 # Output figures and plots
├── assets/                  # GUI images/screenshots
├── requirements.txt         # Python dependencies
├── setup.py                 # Install script
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/FYanhira/pyDielectriX.git
   cd pyDielectriX
   ```
2. **(Recommended) Create a virtual environment:**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage


### 🖥️ Run the Main GUI (Recommended)
```bash
python DielectricModelUI_BO.py
```

- Load your CSV data (columns: `frequency`, `real`, `imag`).
- Select one or more models to fit (Permittivity or Modulus domain).
- Adjust parameter bounds if needed.
- Use **Fit LM (Levenberg-Marquardt)** for standard fitting, or **Bayesian Optimize + Fit** for BO fitting.
- Compare results for all selected models in a single plot (with compact legends).
- Export results and fitted curves as CSV.

### 🧑‍💻 Example: Programmatic Model Fitting
```python
import pandas as pd
import numpy as np
from models.colecole import ColeColeModel

# Load your data
data = pd.read_csv('data/your_data.csv')
freq = data['frequency'].values
eps_real = data['real'].values
eps_imag = data['imag'].values

# Initialize and fit model
model = ColeColeModel()
model.set_auto_params_from_data(eps_real)
res_real, res_imag = model.fit(freq, eps_real, eps_imag)

# Print fitted parameters
for name, param in model.params.items():
    print(f"{name}: {param.value:.4f}")
```

---

## 🧪 Models Implemented
- Debye
- Cole-Cole
- Cole-Davidson
- Havriliak-Negami
- Fractional-1CR, Fractional-2CR
- Ionic conductivity (composite models)
- All above also in electric modulus domain

---


## 📄 Data Format
CSV files should have columns:
- `frequency` (Hz)
- `real` (real part of permittivity or modulus)
- `imag` (imaginary part)

Example:
```
frequency,real,imag
100, 10.5, 2.1
200, 9.8, 2.3
...
```

---

## 📜 License
MIT License — Free to use, modify, and distribute with attribution.

---


## 👩‍🔬 Author
Developed by Flor Yanhira Rentería Baltiérrez
- GitHub: [FYanhira](https://github.com/FYanhira)

---

## 🤝 Contributing
Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request.

---


## 🙏 Acknowledgements
- Built with [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [lmfit](https://lmfit.github.io/lmfit-py/), [scikit-learn](https://scikit-learn.org/), [scikit-optimize](https://scikit-optimize.github.io/), and [Tkinter](https://wiki.python.org/moin/TkInter).
- Includes example datasets for testing and demonstration.

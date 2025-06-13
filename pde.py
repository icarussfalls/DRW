import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

# --- Load data ---
stock = 'HIDCL'
data = pd.read_csv(f'/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/{stock}.csv')
data['date'] = pd.to_datetime(data['date']).dt.date
data.dropna(inplace=True)

train_data = data.iloc[:-30]
S = train_data['close'].values
t = np.arange(len(S))

# --- Smooth the price ---
S_smooth = savgol_filter(S, 11, 3)

# --- Derivative ---
dSdt = np.zeros_like(S_smooth)
dSdt[1:-1] = (S_smooth[2:] - S_smooth[:-2]) / 2
dSdt[0] = S_smooth[1] - S_smooth[0]
dSdt[-1] = S_smooth[-1] - S_smooth[-2]

# --- Build features: S, S^2, sin(t), cos(t), 1 ---
Theta = np.column_stack([
    S_smooth,
    S_smooth**2,
    np.sin(2 * np.pi * t / 30),
    np.cos(2 * np.pi * t / 30),
    np.ones_like(S_smooth)
])
lib_labels = ["S", "S^2", "sin(t)", "cos(t)", "1"]

# --- Scale features ---
scaler = StandardScaler()
Theta_scaled = scaler.fit_transform(Theta)
scales = scaler.scale_

# --- RidgeCV ---
ridge_cv = RidgeCV(alphas=np.logspace(-3, 2, 100), fit_intercept=False)
ridge_cv.fit(Theta_scaled, dSdt)
coeffs_scaled = ridge_cv.coef_
best_alpha = ridge_cv.alpha_

# --- Unscale coefficients ---
coeffs_unscaled = coeffs_scaled / scales

# --- Print final ODE ---
print(f"Best alpha chosen by CV: {best_alpha}")
print("Unscaled Inferred ODE:")
print("dS/dt =", " + ".join([f"({c:.6f})*{label}" for c, label in zip(coeffs_unscaled, lib_labels)]))

# --- Define the ODE with time ---
def ode_with_time(t, S):
    features = np.array([
        S[0],
        S[0]**2,
        np.sin(2 * np.pi * t / 30),
        np.cos(2 * np.pi * t / 30),
        1.0
    ])
    dSdt = np.dot(coeffs_unscaled, features)
    return [dSdt]

# --- Simulate forward ---
t_span = (t[-1], t[-1] + 30)
t_eval = np.arange(t[-1], t[-1] + 31)
S0 = [S[-1]]
sol = solve_ivp(ode_with_time, t_span, S0, t_eval=t_eval)

# --- Plot ---
actual_last_30 = data['close'].values[-30:]
dates_actual = pd.to_datetime(data['date'].values[-30:])

plt.figure(figsize=(12,6))
plt.plot(train_data['date'], S, label="Training Price")
plt.plot(dates_actual, actual_last_30, label="Actual Last 30", color='orange')
plt.plot(dates_actual, sol.y[0][1:], '--', label="Simulated", color='green')
plt.xticks(rotation=45)
plt.title(f"Stock Price Prediction with Time-dependent ODE: {stock}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

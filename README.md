[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/DsXbTGI3)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20926092&assignment_repo_type=AssignmentRepo)

### Course
**CS 418 – Introduction to Data Science (Fall 2025)**  

# CS 418 Project: Predicting Remaining Useful Life of Train Air Systems
# 🚆 RailLife: Predicting Remaining Useful Life (RUL) of Train Air Systems



**Dataset:** [MetroPT-3 (UCI Repository)](https://archive.ics.uci.edu/dataset/933/metropt3)

---

## 📌 Project Overview
RailLife aims to develop a **predictive maintenance system** for metro train **Air Production Units (APU)** by forecasting the **Remaining Useful Life (RUL)** of key subsystems like compressors.  
The goal is to shift from *scheduled maintenance* to *condition-based maintenance* using **multivariate time-series sensor data**.

---

## 🎯 Motivation
Railway maintenance today is mostly time-based — replace parts after a fixed number of days.  
This causes:
- Over-maintenance (wasted effort and cost)  
- Unexpected breakdowns (downtime and safety issues)

Predicting **how much life is left before failure (RUL)** allows operators to:
- Plan maintenance precisely when needed  
- Improve system reliability  
- Save cost and reduce downtime  

---

## 🧩 Dataset Summary — MetroPT-3
**Source:** UCI Machine Learning Repository  
**Records:** 15,169,480 samples (1Hz)  
**Duration:** Feb–Aug 2020  
**Sensors:** 15 (7 analog + 8 digital)  

### Key Features
| Type | Sensor | Description |
|------|---------|-------------|
| Analog | TP2, TP3, H1, DV Pressure, Reservoirs | Air pressure readings at different points |
| Analog | Motor Current | Current in compressor motor (0–9 A) |
| Analog | Oil Temperature | Compressor oil temperature |
| Digital | COMP, DV Electric, Towers, MPG, LPS, Pressure Switch, Oil Level, Caudal Impulse | Compressor valve, air dryer, oil level, and state indicators |

### Failure Events
| ID | Start | End | Fault | Maintenance |
|----|--------|------|--------|-------------|
| 1 | 2020-04-18 00:00 | 2020-04-18 23:59 | Air leak | — |
| 2 | 2020-05-29 23:30 | 2020-05-30 06:00 | Air leak | Apr 30 12:00 |
| 3 | 2020-06-05 10:00 | 2020-06-07 14:30 | Air leak | Jun 8 16:00 |
| 4 | 2020-07-15 14:30 | 2020-07-15 19:00 | Air leak | Jul 16 00:00 |

---

## 🧠 Approach & Methodology

### Phase 1: Data Exploration & Cleaning
- Load `.csv` files using Pandas  
- Convert timestamps to datetime format  
- Visualize key signals (pressure, temperature, current)  
- Handle missing or inconsistent data  
- Segment data into **run-to-failure** cycles based on failure logs

### Phase 2: Feature Engineering
- Compute rolling mean, std, min/max of key analog signals  
- Add rate-of-change features  
- Normalize features (StandardScaler)  
- Compute **Remaining Useful Life (RUL)** for each timestamp:







### Phase 3: Modeling
| Model Type | Algorithms | Purpose |
|-------------|-------------|----------|
| Classical ML | Random Forest, XGBoost | Baseline regression on aggregated features |
| Deep Learning | LSTM / GRU | Sequence-based RUL prediction |
| Anomaly Detection | Isolation Forest / Autoencoder | Detect early warning patterns |

### Phase 4: Evaluation
- Metrics: MAE, RMSE for RUL; F1/AUC for anomaly detection  
- Visualize actual vs predicted RUL degradation curves

### Phase 5: Visualization & Dashboard
- Use **Streamlit** or **Plotly Dash** for an interactive dashboard  
- Show:
- Real-time sensor trends  
- Predicted RUL and risk score  
- Maintenance recommendations  

---

## 📈 Expected Deliverables
| Stage | Output |
|--------|---------|
| Week 1–2 | Clean dataset + EDA notebook |
| Week 3 | Feature-engineered dataset + RUL labeling |
| Week 4–5 | Baseline and deep learning RUL models |
| Week 6 | Evaluation report + visualizations |
| Week 7 | Final dashboard + presentation |

---

## 👥 Team Roles
| Member | Role | Responsibility |
|---------|------|----------------|
| Datta Sai V V N | Lead / Modeling | EDA, feature engineering, LSTM/GRU implementation |
| Member 2 | Data Engineer | Cleaning, RUL labeling, failure segmentation |
| Member 3 | ML Engineer | Baseline models (RF, XGBoost), hyperparameter tuning |
| Member 4 | Visualization | Dashboard (Streamlit / Plotly) and documentation |
| Member 5 | Report Writer | Final report and presentation slides |

---

## 🧰 Tech Stack
- **Python**: Pandas, NumPy, Scikit-learn, PyTorch / TensorFlow  
- **Visualization**: Matplotlib, Seaborn, Plotly  
- **App / Dashboard**: Streamlit / Dash  
- **Version Control**: Git & GitHub  



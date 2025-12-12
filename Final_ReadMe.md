# RailLife: Predictive Maintenance for Railway Systems

**CS 418 Final Project - Team 13**

## Team Members
- James Cook
- Datta Sai VVN
- Heba Syed
- Faaizah Ismail
- Lasya Sahithi

## Project Overview

RailLife is an advanced machine learning system for predicting the Remaining Useful Life (RUL) of train air production units. Traditional railway maintenance relies on fixed schedules, leading to inefficiencies, unexpected breakdowns, and safety issues. Our project enables condition-based maintenance by predicting equipment failures before they occur, with a target accuracy of less than 50 hours of error.

## Problem Statement

Railway maintenance costs can exceed $50,000 per incident when unexpected breakdowns occur. Our goal is to develop a predictive model that:
- Predicts RUL with <50 hours of error
- Enables proactive maintenance scheduling
- Reduces downtime and improves safety
- Optimizes maintenance resource allocation

## Dataset: MetroPT-3

We utilized sensor data from the UCI repository, covering February-September 2020 with four recorded failure events.

### Key Sensors

| Sensor | Description | Degradation Indicator |
|--------|-------------|----------------------|
| Motor_current | Electrical current draw (Amperes) | Increased load from wear |
| Oil_temperature | Lubricant temperature (°C) | Friction, cooling failure |
| TP2, TP3 | Temperature probes at key locations | Localized overheating |
| H1 | Air intake pressure valve reading | Supply degradation |
| DV_pressure | Distributor valve pressure (bar) | Pressure regulation failure |
| Reservoirs | Air reservoir pressure | System capacity decline |
| COMP | Compressor output pressure | Core functionality metric |
| MPG | Main pressure gauge | Overall system health |

### Failure Events

| ID | Start | End | Fault | Maintenance |
|----|-------|-----|-------|-------------|
| 1 | 2020-04-18 00:00 | 2020-04-18 23:59 | Air leak | — |
| 2 | 2020-05-29 23:30 | 2020-05-30 06:00 | Air leak | Apr 30 12:00 |
| 3 | 2020-06-05 10:00 | 2020-06-07 14:30 | Air leak | Jun 8 16:00 |
| 4 | 2020-07-15 14:30 | 2020-07-15 19:00 | Air leak | Jul 16 00:00 |

### Data Preprocessing

- Original sampling rate: 1Hz (1 sample/second)
- Downsampled to: 1-hour intervals (mean aggregation)
- Train/test split: Events 1-2 for training, Events 3-4 for testing
- Training: 3,091 event samples → 3,022 sequences
- Validation: 80/20 split → 2,417 train, 605 validation
- Test: 319 event samples → 296 sequences

## Feature Engineering

We engineered 233 features from 10 raw sensor readings to capture temporal degradation patterns, cumulative stress, and early warning signals.

### Feature Categories

**Base Sensors (10 features)**
- Z-score normalized sensor readings

**EWMAs (30 features)**
- Exponential weighted moving averages with spans of 3h, 12h, 24h
- Smooths noise while emphasizing recent observations

**Delta Features (20 features)**
- First-order differences at 1h and 3h intervals
- Captures rate of change

**Rolling Statistics (150 features)**
- Windows: 6h, 12h, 24h
- Metrics: Mean, Std Dev, Max, Min, Range
- High standard deviation indicates erratic behavior

**Degradation Signatures (15 features)**
- Custom domain-specific features
- Temperature trends, motor-temperature interaction, volatility metrics

**Baseline Deviations (18 features)**
- Established "healthy operation" baselines from first 10% of data
- Deviation measures direct degradation

## Machine Learning Models

### 1. Gradient Boosted Decision Tree Regressor (GBR)

Our best-performing traditional ML model with careful hyperparameter tuning.

**Configuration:**
- Estimators: 200
- Max depth: 5
- Subsample: 0.8
- Learning rate: 0.1

**Performance:**
- MAE: 171.44 hours
- Successfully captured linear decay trends
- Limited by reliance on "time since last failure" feature

**Top 10 Most Important Features:**

| Feature | Importance | Category |
|---------|-----------|----------|
| MPG 24hr EWMA | 0.391222 | Temperature |
| COMP 24hr EWMA | 0.146122 | Pressure |
| Temp trend 720hr | 0.082274 | Temperature |
| DV pressure rolling 24hr min | 0.065527 | Pressure |
| Caudal impulses | 0.056647 | Flow |
| H1 24hr EWMA | 0.024132 | Pressure |
| LPS 24hr EWMA | 0.020982 | Pressure |
| Temp recent 24hr max | 0.018370 | Temperature |
| DV pressure rolling 24hr max | 0.015326 | Pressure |
| Motor 48hr volatility | 0.013633 | Motor |

### 2. Cox Proportional Hazards Model

Survival analysis approach modeling hazard function as a product of baseline hazard and exponential term.

**Performance:**
- MAE: 292.7 hours
- Successfully detected the third failure event
- Better at identifying imminent failures

### 3. Dynamic Ensemble (GBR + Cox)

Combined strengths of both regressors through weighted voting.

**Performance:**
- MAE: 216.0 hours
- Improved over Cox alone but worse than GBR

### 4. LSTM Deep Learning Network ⭐ **BEST MODEL**

Long Short-Term Memory network designed to capture temporal dependencies and sequential degradation patterns.

**Architecture:**
- **Input Layer:** 233 engineered features
- **LSTM Layer 1:** 64 units, return_sequences=True, dropout=0.2
- **Batch Normalization**
- **LSTM Layer 2:** 32 units, return_sequences=False, dropout=0.2
- **Batch Normalization**
- **Dense Layer 1:** 64 units, ReLU activation, dropout=0.3
- **Dense Layer 2:** 32 units, ReLU activation, dropout=0.2
- **Output Layer:** 1 unit (RUL prediction)

**Key Design Decisions:**
- 24-hour lookback window for daily operational cycles
- Dropout (0.2-0.3) to prevent overfitting with limited failure events
- Batch normalization for training stability
- ReLU activation for computational efficiency

**Training Details:**
- Early stopping triggered at epoch 13 (best validation loss: 389,226)
- Training loss: 550,000 → 58,000 (50 epochs)
- Validation loss: 530,859 → 380,000

**Performance:** 🏆
- **MAE: 139.38 hours**
- 18.7% better than GBR
- 35.5% better than Ensemble
- 52.4% better than Cox
- 75.8% better than Baseline

## Results Summary

| Model | MAE (hours) | Improvement vs Baseline |
|-------|-------------|------------------------|
| **LSTM** | **139.38** | **75.8%** |
| GBR | 171.44 | 70.3% |
| Dynamic Ensemble | 216.00 | 62.5% |
| Cox Regression | 292.70 | 49.2% |
| Baseline (Mean) | 576.58 | — |

## Key Findings

### Strengths
✅ LSTM temporal modeling significantly outperforms traditional ML approaches  
✅ Sequential sensor behavior is essential for capturing early degradation signals  
✅ Window-based features (6h-24h) carry the most predictive signal  
✅ Model provides actionable early-warning windows (100+ hours advance notice)  
✅ Feature engineering pipeline successfully captures domain-specific degradation patterns

### Limitations
⚠️ Only 4 failure events severely constrain generalization  
⚠️ Model shows overfitting behavior (training loss ~58k vs validation ~380k)  
⚠️ Censored data creates ambiguity in RUL targets  
⚠️ Synthetic data generation did not improve performance  
⚠️ Classification approach (predicting failure within N hours) was unsuccessful

### Future Improvements
🔮 Uncertainty quantification for confidence intervals  
🔮 Transfer learning from similar industrial equipment  
🔮 Hybrid real-time systems combining LSTM with traditional models  
🔮 Scale to 30+ failure cycles for production-grade reliability (<50h MAE target)

## Technical Stack

- **Python 3.x**
- **scikit-learn** - GBR, preprocessing, metrics
- **TensorFlow/Keras** - LSTM implementation
- **lifelines** - Cox proportional hazards model
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **matplotlib/seaborn** - Visualizations

## Repository Structure

```
├── Final_Models.ipynb          # Main notebook with all models
├── README.md                    # This file
├── data/                        # Dataset (not included - see UCI link)
├── models/                      # Saved model checkpoints
└── visualizations/              # Generated plots and figures
```

## Usage

```python
# Load the trained LSTM model
from tensorflow import keras
model = keras.models.load_model('models/lstm_raillife.h5')

# Prepare 24-hour sequence of 233 features
# sequence shape: (1, 24, 233)
predicted_rul = model.predict(sequence)
print(f"Predicted RUL: {predicted_rul[0][0]:.2f} hours")
```

## Visualizations

Our analysis includes:
- Correlation heatmaps showing sensor relationships
- Boxplots revealing operational modes (idle vs full-load)
- RUL prediction graphs overlaid on actual failure events
- Feature importance rankings
- Training/validation loss curves

## Business Impact

Early warning windows of 100+ hours enable:
- Scheduled maintenance during low-traffic periods
- Reduced emergency repair costs (potentially saving $50k+ per incident)
- Improved passenger safety and service reliability
- Optimized spare parts inventory management

## References

- **Dataset:** UCI Machine Learning Repository - MetroPT-3 Dataset
- **Notebook:** [Project GitHub](https://github.com/cs418-fa25/project-check-in-team-13)

## License

This project was completed as part of CS 418 coursework at the University of Illinois Chicago.

---

*For questions or collaboration opportunities, please contact the team members.*

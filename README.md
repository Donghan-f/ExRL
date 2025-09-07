
**ExRL** is a research project exploring the use of reinforcement learning (RL) to optimize currency exchange decisions over finite horizons.  
The project defines a custom Gymnasium environment (`ExchangeEnv`) that simulates CNY—USD conversion with different forecasting models and reward functions, and trains RL agents to minimize the effective exchange cost.  

Although the experiments did not yield consistent improvements over simple uniform benchmarks, the implementation provides a structured testbed and highlights key challenges of applying RL to foreign exchange.

---

## Problem Setting

Suppose you need to convert $30,000 into CNY before a fixed deadline (e.g., September 1st). Every day you receive a predicted exchange rate for the upcoming days. The question is:

> **How should you split your exchange amount day-by-day to minimize cost under uncertain future conditions?**

We model this as a **finite-horizon MDP** and train RL agents using offline historical data and online forecast models.


---

## Project Structure

exchange_rl_project/
│
├── predictors/               # Exchange rate forecasting modules
│   ├── base.py              # Abstract interface: RatePredictor
│   ├── fx_predictor.py   # Different predictors｜
├── models/                   # Reinforcement learning agents
│   
├── envs/
       ├── gymnasium_env 
               ├── envs
                         ├── exchange_env.py #Fixed-horizon environment
│
├── data/
│   └── historical_rates.csv # Historical exchange rate data (USD/CNY, etc.)
│
├── batch_trainning.py                 # Training different models simultaneously
├── single_trainning.py                 # Training a single model 
├── evaluate.py                         # Evaluate a single model
├── evaluate_batch.py              # Evaluate and compare different models
└── utility.py                 # Helper functions (e.g., data loading, plotting)

##  Environment Design

The custom Gymnasium environment simulates sequential currency exchange  
(USD → CNY) over a finite horizon. The agent decides how much to exchange each day,  
given predictions of future rates, remaining days, and remaining budget.  

---

### Two Environment Modes

#### 1. Fixed-Parameter Environment (Env v1)
- **Horizon (H)** and **Total Amount (M0)** are passed as *environment parameters* at initialization.  
- Example: `ExchangeEnv(horizon=15, total_amount=10_000_000, ...)`  
- These values remain constant across all episodes.  
- Observation only contains *relative states*:
[days_left_frac, money_left_frac, preds_pad]

markdown

- **Pros**: Simple to implement; good for controlled experiments.  
- **Cons**: Each model is tied to its configuration (H, M0); no generalization across different horizons/amounts.  

#### 2. Universal Environment (Env v2)
- **Horizon (H)** and **Total Amount (M0)** are **sampled per episode** within predefined ranges.  
- Example: `H ~ Uniform(H_min, H_max)` and `M0 ~ Uniform(M_min, M_max)`  
- Both values are encoded in the **observation vector**:
[days_left_frac, money_left_frac, preds_pad, mask_pad]

markdown

- **Pros**: One model can adapt to multiple horizons and budgets; closer to real-world usage.  
- **Cons**: Harder to train due to greater variability; reward signal is noisier.  

---

### Reward Modes

Several reward formulations were designed to encourage different agent behaviors:

1. **original**  
 - Per-step reward: `- amount * rate`  
 - At the end: leftover USD is converted at the last-day rate as a penalty.  
 - **Motivation**: Directly minimize CNY cost.  
 - **Issue**: Agent often procrastinates, hoarding all money until the final day.  

2. **relative**  
 - Reward = `(baseline - rate) * amount - α * leftover_ratio`  
 - Baseline: predictor forecast or rolling mean.  
 - **Motivation**: Spend more when today’s rate is cheaper than expected.  
 - **Issue**: Sensitive to predictor quality.  

3. **burn**  
 - Reward = `(baseline - rate) * amount - β * excess_leftover`  
 - Excess leftover: how much the agent lags behind a linear spending schedule.  
 - **Motivation**: Encourage smoother spending across horizon.  
 - **Issue**: May force premature spending even when rates are bad.  

4. **relative_burn**  
 - Reward = `(baseline - rate) * amount - α * leftover_ratio - β * excess_leftover`  
 - **Motivation**: Combine rate advantage with schedule discipline.  
 - **Issue**: Harder to tune (α, β trade-off).  

5. **relative_final**  
 - Reward = `(pred_final - rate) * amount - α * leftover_ratio`  
 - Baseline: predictor’s forecast of the **final-day rate**.  
 - **Motivation**: Compare today’s rate to expected final-day rate.  
 - **Issue**: Relies on long-term prediction accuracy.  

---

###  Observation Padding

The observation vector must have a **fixed dimension** for RL algorithms,  
but horizons vary across episodes. To handle this, we pad predictor outputs:  

- **preds_pad**: Predictor outputs for the next *k* days (`k = H`),  
padded to length `H_max`.  
- **mask_pad**: Binary mask indicating which entries are valid (1) vs padding (0).  

 Padding strategies:  
1. Zero padding (our choice).  
2. Sentinel padding (e.g. -10).  
3. Masking-only.  

We adopt **zero padding** for simplicity, with a mask provided so the agent can learn to ignore padded positions.  

**Example (H=5, H_max=10):**

| Step | Predictor Output | Padded Value | Mask |
|------|------------------|--------------|------|
| 1    | 7.05             | 7.05         | 1    |
| 2    | 7.06             | 7.06         | 1    |
| 3    | 7.07             | 7.07         | 1    |
| 4    | 7.08             | 7.08         | 1    |
| 5    | 7.10             | 7.10         | 1    |
| 6–10 | —                | 0.00         | 0    |

---
## Training Process (Fixed-Parameter Environment, Env v1)

In the fixed-parameter environment, each model is trained for a **specific horizon (H)** and **total amount (M0)**.  
This means one model corresponds to one exact configuration, e.g.  
> “PPO with HoltWinters predictor, H=15, M0=10M, reward=relative_final, alpha=0.001”.

---

###  Training Goal

The agent learns a policy to minimize the total CNY cost of converting USD over a fixed horizon.  
Training explores different RL algorithms, predictors, and reward functions to test which setup yields the best improvement over a uniform baseline strategy.

---

###  Environment Parameters

- **Horizon (H)**: Number of trading days in an episode.  
  - Values used: `5`, `15`, `25`  
- **Total Amount (M0)**: Total USD to exchange during the horizon.  
  - Fixed at: `10,000,000` USD  
- **k (prediction window)**: Set equal to horizon (`k = H`)  
- **History Window**: `250` days (for predictor input)  
- **Max Daily Exchange**: Disabled (no explicit cap)  

---

###  Predictors

Each environment is equipped with one predictor for forecasting future FX rates:  
- **NaiveRW** (Random Walk)  
- **Drift** (trend-based)  
- **HoltWinters** (seasonal smoothing)  
- **Theta** (decomposition-based)  
- **LocalLevel** (state-space model)  


The predictor generates forecasts that shape the observation space and reward signals.

---

### Reward Modes

Reward shaping strategies explored:

1. **original**  
   - Reward = `- amount * rate` each step  
   - Terminal penalty for leftover USD at last-day rate  
   - **Issue**: Agent tends to hoard and spend everything on the last day  

2. **relative**  
   - Reward = `(baseline - rate) * amount - α * leftover_ratio`  
   - Baseline = predictor or rolling mean  
   - **α values tried**: `0.0001`, `0.001`  

3. **burn**  
   - Reward = `(baseline - rate) * amount - β * excess_leftover`  
   - **β values tried**: `0.01`, `0.05`  

4. **relative_burn**  
   - Combination of relative and burn terms  
   - **α values tried**: `0.0001`, `0.001`  
   - **β values tried**: `0.01`, `0.05`  

5. **relative_final**  
   - Reward = `(pred_final - rate) * amount - α * leftover_ratio`  
   - Pred_final = predictor’s forecast of last-day rate  
   - **α values tried**: `0.0001`, `0.001`  

---

###  RL Algorithms

Three standard RL algorithms were tested:

- **PPO (Proximal Policy Optimization)**  
  - Clip range: `0.2`  
  - Gamma: `0.999`  
  - Learning rate: `3e-4`  
  - N-steps: `64`  
  - Entropy coef: `0.0`  

- **A2C (Advantage Actor-Critic)**  
  - Gamma: `0.999`  
  - Learning rate: `7e-4`  
  - N-steps: `5`  
  - Entropy coef: `0.0`  

- **DDPG (Deep Deterministic Policy Gradient)**  
  - Used default hyperparameters from Stable-Baselines3  

---
##  Results (Env v1)

Evaluation across **PPO, A2C, DDPG** with multiple predictors and reward modes showed:

- **Overall improvement negligible**  
  - Most runs achieved <0.05% improvement vs. the uniform baseline.  
  - Several runs performed worse than baseline.

- **Failure modes observed**  
  - `original`: agent often hoarded funds and spent everything on the last day.  
  - `relative` / `relative_burn`: occasionally spread spending, but no consistent gain.  
  - `relative_final`: sometimes positive, but typically left money unspent.  

- **Best runs (by improvement vs. uniform)**  
  - PPO + LocalLevel (original): +0.0091%  
  - DDPG + Drift (original): +0.0077%  
  - PPO + HoltWinters (relative_final): +0.0055%  
  → These values are statistically insignificant compared to noise.

- **Spent ratio patterns**  
  - Many runs ended with 0% or 100% of funds spent.  
  - Partial spending occurred in `relative_burn`, but without performance advantage.

###  Interpretation
- Reward shaping as designed was **insufficient to overcome procrastination bias**.  
- Predictor-driven signals were too weak/noisy for the agent to exploit.  
- No algorithm or predictor consistently outperformed others.  
- Results suggest the **fixed-parameter formulation (Env v1) is unlikely to yield meaningful gains**.

###  Universal Environment (Env v2)

We also repeated the same training setup in the **universal environment** (where horizon and total amount are randomized per episode and included in the state).  
The outcome was consistent: **agents almost always postponed action and converted all funds on the final day**, regardless of algorithm, predictor, or reward mode.  
This indicates that the procrastination bias observed in the fixed setup (Env v1) persists, and reward shaping as currently designed is insufficient to overcome it.






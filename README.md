# Hybrid Behavior Planner — Classical vs Learning vs Hybrid (with Safety)

Scenario-driven evaluation framework for autonomous navigation that compares:
- **Classical FSM**
- **Imitation Learning (Behavior Cloning)**
- **Reinforcement Learning (PPO)**
- **PPO + Safety Shield**
- **Hybrid Planner** (PPO+Shield primary + FSM fallback)

Includes reproducible benchmarking on **unseen scenarios**, metric reporting, and **cinematic side-by-side visualizations**.

---

## Highlights

- **Same environment, same scenarios** → fair policy comparison
- **Safety Shield** overrides unsafe actions near obstacles (hard constraints)
- **Hybrid fallback** improves robustness under distribution shift
- Outputs:
  - Summary metrics (success/collision/latency/clearance)
  - Plots
  - Side-by-side demos (GIF + optional MP4)

---

## Results (Unseen Scenario Test)

> **15 unseen scenarios** (generalization benchmark)

| Policy | Success ↑ | Collision ↓ | Avg steps | Mean min-front ↑ | Avg decision latency (ms/step) ↓ |
|---|---:|---:|---:|---:|---:|
| Classical FSM | 0.40 | 0.60 | 89.0 | 2.84 | 0.023 |
| Imitation (BC) | 0.53 | 0.47 | 87.3 | 3.23 | 0.112 |
| RL (PPO) | 0.93 | 0.07 | 116.0 | 3.51 | 2.201 |
| RL + Safety Shield | 0.93 | 0.07 | 116.1 | 3.50 | 1.423 |
| **Hybrid (PPO+Shield + FSM fallback)** | **1.00** | **0.00** | 121.9 | 17.30 | 0.345 |

**Safety Shield override rate:** 0.0034 (0.34% of steps)  
**Hybrid fallback rate:** 0.0055 (0.55% of steps)

---

## Demo (Cinematic Compare)

Side-by-side behavior in the same unseen scenario:

```bash
python -m evaluation.animate_compare \
  --scenario scenarios/test_unseen_001.json \
  --out evaluation/compare_fsm_vs_ppo_shield_cinematic.gif

Hybrid Behavior Planning: Classical vs Learning-Based vs Hybrid

This project implements and evaluates a minimal hybrid behavior planning architecture combining:

Deterministic rule-based planning (Finite State Machine)

Imitation Learning (Behavior Cloning)

Reinforcement Learning (PPO)

PPO + explicit Safety Shield

Hybrid planner (learning-based primary + deterministic fallback)

All policies operate in the same 2D vehicle simulation environment and are evaluated on 15 unseen scenarios.

Unseen Scenario Evaluation (15 Scenarios)
Policy	Success ↑	Collision ↓	Avg steps	Mean min-front ↑	Avg decision latency (ms/step) ↓
Classical FSM	0.40	0.60	89.0	2.84	0.023
Imitation (BC)	0.53	0.47	87.3	3.23	0.112
RL (PPO)	0.93	0.07	116.0	3.51	2.201
RL + Safety Shield	0.93	0.07	116.1	3.50	1.423
Hybrid (PPO+Shield + FSM fallback)	1.00	0.00	121.9	17.30	0.345

Safety Shield override rate: 0.0034 (0.34% of steps)
Hybrid fallback rate: 0.0055 (0.55% of steps)

Success vs Collision (Unseen Test)

Decision Latency (ms/step)

Safety Proxy: Mean Front Distance

Higher is safer (vehicle keeps larger margin from obstacles).

Side-by-Side Behavior Demo

Deterministic FSM vs Learning-based PPO+Shield in the same unseen scenario:

Architecture Overview

The system separates:

Environment (simulation + sensors)

Policy interface

Classical planner (FSM)

Learning policies (IL, PPO)

Safety shield (hard constraints)

Hybrid coordinator (fallback logic)

This allows:

Swapping planners without changing the environment

Measuring generalization across unseen scenarios

Injecting deterministic safety layers

Comparing latency and determinism trade-offs

Key Technical Insights
1️⃣ Determinism vs Generalization

FSM is fully deterministic and extremely fast (0.02 ms/step)

However, rule brittleness leads to high collision rate under distribution shift

Demonstrates classical planner limitations in complex ODDs

2️⃣ Imitation Learning

Improves over FSM in unseen scenarios

Still inherits expert bias

Suffers from compounding errors

3️⃣ Reinforcement Learning (PPO)

Strong generalization and success rate

Significantly higher computational cost

Less deterministic

4️⃣ Safety Shield

Hard constraint layer

Overrides unsafe actions near obstacles

Very low intervention rate (0.34%)

Preserves learning performance while enforcing safety bounds

5️⃣ Hybrid Planner

PPO+Shield handles nominal cases

FSM fallback activates in high-risk states

Achieves 100% success on unseen scenarios

Demonstrates production-oriented hybrid planning design

Why This Is Relevant to Behavior Planning

This project demonstrates:

Behavior architecture design

Modular planning interfaces

Learning-based planning integration

Safety constraints enforcement

Generalization evaluation

Failure-mode reasoning

Deterministic fallback strategies

Latency-aware decision making

It is a minimal but realistic example of transitioning from modular classical planning to hybrid AI-based planning systems with explicit safety reasoning.

How to Reproduce
cmake -S . -B build
cmake --build build -j

# Evaluate all planners
python -m evaluation.eval_all

# Generate plots
python evaluation/plot_all.py
python evaluation/plot_all_rates_combined.py

# Generate demo animation
python -m evaluation.animate_compare \
  --scenario scenarios/test_unseen_001.json \
  --out evaluation/compare_fsm_vs_ppo_shield.gif
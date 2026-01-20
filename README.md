# Reinforcement Learning–Based Policy Optimization for Wireless Sensor Networks

## Overview
This project studies Energy-aware data aggregation in Wireless Sensor Networks (WSNs) using **classical Reinforcement Learning planning algorithms**.  
The transmission decision problem is modeled as a **Markov Decision Process (MDP)** under stochastic wind dynamics, solar energy harvesting, battery constraints, and unreliable communication.

We implement and compare:
- **Greedy Policy** (baseline)
- **Value Iteration**
- **Policy Iteration**

to derive optimal transmission strategies that minimize long-term estimation error while conserving energy.

---

## Problem Setting
Wireless Sensor Nodes operate under:
- Limited battery capacity with stochastic solar recharging
- Uncertain wireless channel (probabilistic transmission success)
- Scheduling constraints (active/passive phases)
- Stochastic wind-speed dynamics modeled as a Markov chain

The node must decide **when and what value to transmit** to a monitoring station to optimize long-term performance.

---

## MDP Formulation
**State**  
\( s_t = (\phi_t, z_t, b_t, p_t) \)
- \( \phi_t \): current wind speed  
- \( z_t \): last successfully transmitted wind speed  
- \( b_t \): battery level  
- \( p_t \): remaining active phase slots  

**Action**
- Transmit a value from the discrete wind-speed set
- Or choose *no transmission*

**Reward**
- Negative squared estimation error, accounting for transmission success probability

**Objective**
- Minimize long-term discounted cumulative cost

---

## Algorithms Implemented
- **Greedy Policy** – minimizes immediate error only
- **Value Iteration** – Bellman optimality updates until convergence
- **Policy Iteration** – alternating policy evaluation and improvement

All algorithms are implemented **from scratch in Python**, without RL libraries.

---

## Key Results
- **Policy Iteration converges ~3× faster** than Value Iteration for the same optimal policy
- Optimal policies consistently outperform greedy strategies
- Policies adapt intelligently to:
  - Battery scarcity
  - Wind speed variability
  - Long upcoming passive phases
- Heatmaps visualize learned value functions and optimal policies

---


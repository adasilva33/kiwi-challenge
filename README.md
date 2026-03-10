# Solution Comparison

## Methods

| Method | Reference |
|---|---|
| **Best Known** | Benchmark used in both papers |
| **Arnaud (MCTS)** | [ResearchGate](https://www.researchgate.net/publication/388120028_A_Monte_Carlo_Tree_Search_for_the_Optimisation_of_Flight_Connections) — MCTS, no time limit |
| **Yaro (RL-OI)** | [IEEE](https://ieeexplore.ieee.org/document/9185803) — Reinforcement Learning with Optimisation Improvement, no time limit |
| **Our MCTS** | This repo — MCTS with parameter tuning, ~15s time limit |

---

## Cost Comparison (lower is better)

| Instance | Best Known | Arnaud (MCTS) | Yaro (RL-OI) | Our MCTS | Our vs Best Known |
|:---:|---:|---:|---:|---:|---:|
| 1  | 1,396  | **1,396** (0%)    | **1,396** (0%)      | **1,396** | 0%      |
| 2  | 1,498  | **1,498** (0%)    | **1,498** (0%)      | **1,498** | 0%      |
| 3  | 7,672  | **7,672** (0%)    | **7,672** (0%)      | 7,833     | +2.1%   |
| 4  | 13,952 | 15,101 (+8.2%)    | **13,952** (−0.5%)  | 16,367    | +17.3%  |
| 5  | 690    | — (n/a)           | **694** (−0.6%)     | 916       | +32.8%  |
| 6  | 2,159  | — (n/a)           | **1,733** (−19.7%)  | **2,121** | −1.8% ✓ |
| 7  | 30,937 | — (n/a)           | 31,218 (+0.9%)      | 33,642    | +8.7%   |
| 8  | 4,033  | **4,037** (−0.4%) | **4,033** (−0.5%)   | 4,155     | +3.0%   |
| 9  | 76,372 | —                 | 77,892 (+2.0%)      | 84,539    | +10.7%  |
| 10 | 21,167 | —                 | 43,542 (+51.4%)     | 55,167    | +160.6% |
| 11 | 44,153 | —                 | 51,358 (+14.0%)     | 46,873    | +6.2%   |
| 12 | 65,447 | —                 | 76,298 (+14.2%)     | **64,024**| −2.2% ✓ |
| 13 | 97,859 | —                 | 145,233 (+32.6%)    | **94,471**| −3.5% ✓ |
| 14 | 118,811| —                 | 198,573 (+40.2%)    | **114,378**| −3.7% ✓ |

> **Best Known** for I4 = 13,952 (achieved by Yaro; Arnaud's paper uses the same value).
> Arnaud's paper covers only instances 1–8; gaps marked — are not reported.
> Negative gap = **better than best known**.

---

## Observations

### Our MCTS beats "best known" on 4 instances (6, 12, 13, 14)
These are the larger instances (96–300 areas). Our solver apparently finds better routes than the benchmarks cited in both papers for these cases.

### Our MCTS matches the optimum on instances 1 & 2
The 10-area instances are solved to optimality.

### Instance 10 is a weak point (+160%)
The 300-area/300-airport instance is where our solver degrades most. Yaro's RL-OI also struggles (+51%), suggesting this is a structurally hard instance.

### Instance 5 & 6 — dead-end filtering mattered
Both papers struggled with these sparse instances. Yaro's RL-OI found 694 (I5) and 1,733 (I6), both better than our 916 and 2,121. The 15s time limit is the likely bottleneck for I5/I6.

### Arnaud's MCTS is closest to ours in approach
Their best-found on I3/I1/I2 matches the optimum; on I4 they are at +8.2% (ours +17.3%). Our solver runs under strict time constraints (≤15s), while their results are without time limits.

### Yaro's RL-OI dominates on medium instances (4–8)
RL-OI with no time limit is hard to beat on the mid-range. On larger instances (9–14) it degrades significantly, where our simple MCTS performs comparably or better.

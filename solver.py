"""
Kiwi Challenge - MCTS Solver with Parameter Tuning
===================================================
Problem:
  Given n areas (each with one or more airports), find the minimum-cost
  flight itinerary that:
    - Starts at a given airport (in the starting area)
    - Visits exactly ONE airport per remaining area (n-1 areas, one per day)
    - Ends in any airport of the STARTING AREA (day n)
  Flight costs are asymmetric and time-dependent.  Day-0 flights are every day.

  Path structure: n+1 airports, n flights (days 1..n).
    path[0]  = start_airport
    path[1..n-1] = one airport per non-start area
    path[n]  = any airport in start_area

Approach:
  - MCTS with UCT
  - Precomputed per-airport reachability (day-0 closure) to avoid dead-end airports
  - Rollout strategies: greedy, nn-lookahead, mixed, random
  - Automated parameter tuning
"""

import math
import random
import time
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, FrozenSet, Set


# ─────────────────────────────────────────────────────────────────────────────
# Problem
# ─────────────────────────────────────────────────────────────────────────────

class Problem:
    def __init__(self, filename: str):
        self._parse(filename)
        self._precompute_adjacency()
        self._precompute_reachability()

    def _parse(self, filename: str):
        with open(filename) as f:
            lines = [l.rstrip('\n') for l in f]
        idx = 0
        first = lines[idx].split(); idx += 1
        self.n_areas: int = int(first[0])
        self.start_airport: str = first[1]
        self.areas: List[List[str]] = []
        self.area_names: List[str] = []
        self.airport_to_area: Dict[str, int] = {}
        for i in range(self.n_areas):
            name = lines[idx].strip(); idx += 1
            airports = lines[idx].strip().split(); idx += 1
            self.areas.append(airports)
            self.area_names.append(name)
            for ap in airports:
                self.airport_to_area[ap] = i
        self.start_area: int = self.airport_to_area[self.start_airport]
        # All airports as a list
        self.all_airports: List[str] = list(self.airport_to_area.keys())
        # costs[(from, to, day)] = min cost; day=0 → every day
        self.costs: Dict[Tuple, float] = {}
        while idx < len(lines):
            line = lines[idx].strip(); idx += 1
            if not line: continue
            parts = line.split()
            if len(parts) < 4: continue
            frm, to, day, cost = parts[0], parts[1], int(parts[2]), float(parts[3])
            key = (frm, to, day)
            if key not in self.costs or cost < self.costs[key]:
                self.costs[key] = cost

    def _precompute_adjacency(self):
        """Group edges by (airport, day), sorted by cost."""
        raw: Dict[str, Dict[int, List]] = defaultdict(lambda: defaultdict(list))
        for (frm, to, day), cost in self.costs.items():
            raw[frm][day].append((cost, to, self.airport_to_area[to]))
        self.adj: Dict[str, Dict[int, List]] = {}
        for frm, days in raw.items():
            self.adj[frm] = {}
            for day, lst in days.items():
                lst.sort()
                self.adj[frm][day] = lst

    def _precompute_reachability(self):
        """
        For each airport, compute:
          - reachable_areas: set of areas reachable via any outgoing flights
            (day-0 or any day-specific) within one hop
          - has_outgoing: bool
        Also precompute area_reachable[airport] = frozenset of reachable areas
        (used to quickly check if an airport can lead to the remaining areas).
        """
        self.airport_reachable: Dict[str, FrozenSet[int]] = {}
        self.has_outgoing: Dict[str, bool] = {}

        for ap in self.all_airports:
            ap_adj = self.adj.get(ap, {})
            reachable: Set[int] = set()
            for day, lst in ap_adj.items():
                for _, _, area in lst:
                    reachable.add(area)
            self.airport_reachable[ap] = frozenset(reachable)
            self.has_outgoing[ap] = len(reachable) > 0

        # For a given airport and remaining areas, can we always find a next hop?
        # We precompute for each airport: set of reachable areas (any day or day-0)
        # This is a 1-step look: if airport can reach at least one area in remaining.
        # True feasibility requires multi-step checks (expensive), so we only do 1-step.

    def get_flights(self, airport: str, day: int) -> List[Tuple]:
        """
        Sorted (cost, to_airport, to_area) from airport on day,
        merging day-specific and day-0 flights. Keeps cheapest per destination.
        """
        ap_adj = self.adj.get(airport, {})
        specific = ap_adj.get(day, [])
        everyday = ap_adj.get(0, [])
        if not everyday: return specific
        if not specific: return everyday
        seen: Dict[str, float] = {}
        for cost, to, _ in specific:
            if to not in seen or cost < seen[to]: seen[to] = cost
        for cost, to, _ in everyday:
            if to not in seen or cost < seen[to]: seen[to] = cost
        result = [(cost, to, self.airport_to_area[to]) for to, cost in seen.items()]
        result.sort()
        return result

    def n_airports(self) -> int:
        return sum(len(a) for a in self.areas)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout strategies
# ─────────────────────────────────────────────────────────────────────────────

def rollout(
    airport: str,
    day: int,
    visited: Set[int],
    cost: float,
    path: List[str],
    problem: Problem,
    strategy: str,
    greedy_prob: float = 0.85,
) -> Tuple[float, List[str]]:
    """
    Complete tour from (airport, day, visited, cost, path).
    Returns (total_cost, full_path) or (inf, []).
    """
    n = problem.n_areas
    start_area = problem.start_area
    cur = airport
    visited = set(visited)
    path = list(path)

    while len(visited) < n:
        remaining: Set[int] = set(range(n)) - visited
        if len(remaining) > 1:
            remaining.discard(start_area)

        flights = problem.get_flights(cur, day)

        # Filter: only land at airports that have outgoing flights to something
        # needed later (unless it's the last step, where we land at start_area)
        if len(remaining) == 1:
            candidates = [(c, ap, ar) for c, ap, ar in flights if ar in remaining]
        else:
            # Prefer airports that have outgoing flights to remaining areas
            strong = []   # can reach something in remaining (after this step)
            weak = []     # might be dead-end
            for c, ap, ar in flights:
                if ar not in remaining:
                    continue
                # Check if ap can reach any area in (remaining - {ar})
                after = remaining - {ar}
                if not after:
                    strong.append((c, ap, ar))
                elif problem.airport_reachable.get(ap, frozenset()) & after:
                    strong.append((c, ap, ar))
                else:
                    weak.append((c, ap, ar))
            candidates = strong if strong else weak

        if not candidates:
            return math.inf, []

        if strategy == 'greedy':
            fc, ap, ar = candidates[0]

        elif strategy == 'random':
            random.shuffle(candidates)
            fc, ap, ar = candidates[0]

        elif strategy == 'nn':
            area_best: Dict[int, Tuple] = {}
            for c, ap, ar in candidates:
                if ar not in area_best: area_best[ar] = (c, ap)
            best_score = math.inf
            fc, ap, ar = candidates[0]
            for cand_ar, (cfc, cap) in area_best.items():
                next_rem = remaining - {cand_ar}
                if not next_rem:
                    la = 0.0
                else:
                    nf = problem.get_flights(cap, day + 1)
                    la = next((nfc for nfc, _, nar in nf if nar in next_rem), cfc * 8)
                score = cfc + 0.3 * la
                if score < best_score:
                    best_score = score
                    fc, ap, ar = cfc, cap, cand_ar

        else:  # 'mixed'
            if random.random() < greedy_prob:
                fc, ap, ar = candidates[0]
            else:
                k = max(1, min(5, len(candidates)))
                fc, ap, ar = random.choice(candidates[:k])

        cost += fc
        path.append(ap)
        visited.add(ar)
        cur = ap
        day += 1

    return cost, path


def baseline_rollout(problem: Problem, strategy: str) -> Tuple[float, List[str]]:
    return rollout(problem.start_airport, 1, set(), 0.0,
                   [problem.start_airport], problem, strategy)


# ─────────────────────────────────────────────────────────────────────────────
# MCTS Node
# ─────────────────────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = ('airport', 'day', 'visited', 'cost', 'path',
                 'parent', 'children', 'visits', 'total_reward', 'untried_moves')

    def __init__(self, airport: str, day: int, visited: FrozenSet[int],
                 cost: float, path: List[str], parent=None):
        self.airport = airport
        self.day = day
        self.visited = visited
        self.cost = cost
        self.path = path
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.untried_moves: Optional[List] = None

    def uct_score(self, c: float, log_parent: float) -> float:
        if self.visits == 0: return math.inf
        return (self.total_reward / self.visits) + c * math.sqrt(log_parent / self.visits)


# ─────────────────────────────────────────────────────────────────────────────
# MCTS Solver
# ─────────────────────────────────────────────────────────────────────────────

class MCTSSolver:
    def __init__(self, c: float = 1.41, rollout_strategy: str = 'mixed',
                 greedy_prob: float = 0.85, reward_scale: float = 10_000.0):
        self.c = c
        self.strategy = rollout_strategy
        self.greedy_prob = greedy_prob
        self.reward_scale = reward_scale

    def _reward(self, cost: float) -> float:
        return -1.0 if cost >= math.inf else self.reward_scale / (cost + 1.0)

    def _get_moves(self, node: MCTSNode, problem: Problem) -> List:
        """Return one move per reachable area (cheapest airport), with dead-end filtering."""
        n = problem.n_areas
        remaining: Set[int] = set(range(n)) - node.visited
        if len(remaining) > 1:
            remaining.discard(problem.start_area)
        if not remaining:
            return []

        flights = problem.get_flights(node.airport, node.day)
        area_strong: Dict[int, Tuple] = {}
        area_weak: Dict[int, Tuple] = {}

        for cost, to_ap, to_area in flights:
            if to_area not in remaining:
                continue
            after = remaining - {to_area}
            if not after or problem.airport_reachable.get(to_ap, frozenset()) & after:
                if to_area not in area_strong:
                    area_strong[to_area] = (cost, to_ap, to_area)
            else:
                if to_area not in area_weak:
                    area_weak[to_area] = (cost, to_ap, to_area)

        result = list((area_strong if area_strong else area_weak).values())
        return result

    def solve(self, problem: Problem, time_limit: float) -> Tuple[float, List[str]]:
        n = problem.n_areas
        if n == 1:
            return 0.0, [problem.start_airport]

        root = MCTSNode(airport=problem.start_airport, day=1,
                        visited=frozenset(), cost=0.0,
                        path=[problem.start_airport])

        best_cost = math.inf
        best_path: List[str] = []
        deadline = time.time() + time_limit

        while time.time() < deadline:
            # ── SELECTION ──
            node = root
            while (node.untried_moves is not None
                   and len(node.untried_moves) == 0
                   and node.children
                   and len(node.visited) < n):
                log_p = math.log(node.visits + 1)
                node = max(node.children,
                           key=lambda ch: ch.uct_score(self.c, log_p))

            # ── EXPANSION ──
            if len(node.visited) < n:
                if node.untried_moves is None:
                    moves = self._get_moves(node, problem)
                    random.shuffle(moves)
                    node.untried_moves = moves
                if node.untried_moves:
                    cost, to_ap, to_area = node.untried_moves.pop()
                    child = MCTSNode(
                        airport=to_ap, day=node.day + 1,
                        visited=node.visited | frozenset([to_area]),
                        cost=node.cost + cost,
                        path=node.path + [to_ap],
                        parent=node,
                    )
                    node.children.append(child)
                    node = child

            # ── SIMULATION ──
            if len(node.visited) == n:
                sim_cost, sim_path = node.cost, node.path
            else:
                sim_cost, sim_path = rollout(
                    node.airport, node.day, node.visited, node.cost,
                    node.path, problem, self.strategy, self.greedy_prob,
                )

            if sim_cost < best_cost and sim_path:
                best_cost, best_path = sim_cost, sim_path

            reward = self._reward(sim_cost)

            # ── BACKPROPAGATION ──
            cur = node
            while cur is not None:
                cur.visits += 1
                cur.total_reward += reward
                cur = cur.parent

        return best_cost, best_path


# ─────────────────────────────────────────────────────────────────────────────
# Parameter search grid
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParamConfig:
    c: float
    strategy: str
    greedy_prob: float
    reward_scale: float

    @property
    def label(self) -> str:
        gp = f"|gp={self.greedy_prob:.2f}" if self.strategy == 'mixed' else ''
        return f"c={self.c:.2f}|{self.strategy}{gp}"


PARAM_GRID: List[ParamConfig] = [
    ParamConfig(0.3,  'greedy', 1.00, 10_000),
    ParamConfig(0.3,  'nn',     1.00, 10_000),
    ParamConfig(0.3,  'mixed',  0.90, 10_000),
    ParamConfig(0.5,  'greedy', 1.00, 10_000),
    ParamConfig(0.5,  'nn',     1.00, 10_000),
    ParamConfig(0.5,  'mixed',  0.80, 10_000),
    ParamConfig(0.5,  'mixed',  0.95, 10_000),
    ParamConfig(1.0,  'greedy', 1.00, 10_000),
    ParamConfig(1.0,  'nn',     1.00, 10_000),
    ParamConfig(1.0,  'mixed',  0.80, 10_000),
    ParamConfig(1.0,  'mixed',  0.95, 10_000),
    ParamConfig(1.41, 'greedy', 1.00, 10_000),
    ParamConfig(1.41, 'nn',     1.00, 10_000),
    ParamConfig(1.41, 'mixed',  0.80, 10_000),
    ParamConfig(1.41, 'mixed',  0.95, 10_000),
    ParamConfig(2.0,  'nn',     1.00, 10_000),
    ParamConfig(2.0,  'mixed',  0.80, 10_000),
    ParamConfig(2.0,  'mixed',  0.95, 10_000),
    ParamConfig(3.0,  'mixed',  0.80, 10_000),
    ParamConfig(3.0,  'nn',     1.00, 10_000),
]


# ─────────────────────────────────────────────────────────────────────────────
# Tuner
# ─────────────────────────────────────────────────────────────────────────────

def tune_parameters(
    problem: Problem,
    tune_budget: float,
    verbose: bool = True,
) -> Tuple['ParamConfig', float, List[str]]:
    n_configs = len(PARAM_GRID)
    time_per_config = max(tune_budget / n_configs, 0.05)

    if verbose:
        print(f"\n[Tuner] {n_configs} configs × {time_per_config:.2f}s "
              f"≈ {n_configs * time_per_config:.1f}s")
        print(f"  {'Config':<40} {'Cost':>10}")
        print(f"  {'-'*52}")

    deadline = time.time() + tune_budget
    best_cfg = PARAM_GRID[0]
    best_cost = math.inf
    best_path: List[str] = []

    for cfg in PARAM_GRID:
        if time.time() >= deadline:
            break
        remaining_t = deadline - time.time()
        alloc = min(time_per_config, remaining_t)
        if alloc < 0.01:
            break

        solver = MCTSSolver(c=cfg.c, rollout_strategy=cfg.strategy,
                            greedy_prob=cfg.greedy_prob, reward_scale=cfg.reward_scale)
        cost, path = solver.solve(problem, time_limit=alloc)

        marker = ''
        if cost < best_cost:
            best_cost, best_path, best_cfg = cost, path, cfg
            marker = ' ◄'

        if verbose:
            cs = f"{cost:.0f}" if cost < math.inf else "inf"
            print(f"  {cfg.label:<40} {cs:>10}{marker}")

    if verbose:
        bc = f"{best_cost:.0f}" if best_cost < math.inf else "inf"
        print(f"\n[Tuner] Best: {best_cfg.label}  cost={bc}")

    return best_cfg, best_cost, best_path


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_solution(problem: Problem, path: List[str], cost: float) -> bool:
    n = problem.n_areas
    if len(path) != n + 1:
        print(f"  [verify] Wrong path length {len(path)}, expected {n+1}")
        return False
    if path[0] != problem.start_airport:
        print(f"  [verify] Doesn't start at {problem.start_airport}")
        return False
    if problem.airport_to_area[path[-1]] != problem.start_area:
        last_area = problem.airport_to_area[path[-1]]
        print(f"  [verify] Last airport {path[-1]} is in area {last_area}, "
              f"not start_area {problem.start_area}")
        return False
    visited_areas = set()
    total = 0.0
    for day, (frm, to) in enumerate(zip(path, path[1:]), start=1):
        area = problem.airport_to_area[to]
        if area in visited_areas:
            print(f"  [verify] Area {area} visited twice (day {day})")
            return False
        visited_areas.add(area)
        fc = problem.get_flights(frm, day)
        fc_dict = {ap: c for c, ap, _ in fc}
        if to not in fc_dict:
            print(f"  [verify] No flight {frm}→{to} on day {day}")
            return False
        total += fc_dict[to]
    if abs(total - cost) > 0.5:
        print(f"  [verify] Cost mismatch: computed={total:.0f}, claimed={cost:.0f}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Time budget
# ─────────────────────────────────────────────────────────────────────────────

def time_budget(problem: Problem) -> float:
    n, nap = problem.n_areas, problem.n_airports()
    if n <= 20 and nap < 50:   return 3.0
    if n <= 100 and nap < 200: return 5.0
    return 15.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python solver.py <instance_file> [tune_fraction=0.40]")
        sys.exit(1)

    filename = sys.argv[1]
    tune_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.40

    problem = Problem(filename)
    total_time = time_budget(problem)
    tune_time = total_time * tune_frac
    solve_time = total_time - tune_time

    print(f"Instance : {filename}")
    print(f"Areas    : {problem.n_areas}   Airports: {problem.n_airports()}")
    print(f"Start    : {problem.start_airport} (area {problem.start_area})")
    print(f"Budget   : {total_time:.1f}s  (tune={tune_time:.1f}s  solve={solve_time:.1f}s)")

    # Baselines
    g_cost, g_path = baseline_rollout(problem, 'greedy')
    nn_cost, nn_path = baseline_rollout(problem, 'nn')
    best_cost = min(g_cost, nn_cost)
    best_path = g_path if g_cost <= nn_cost else nn_path
    gc = f"{g_cost:.0f}" if g_cost < math.inf else "inf"
    nc = f"{nn_cost:.0f}" if nn_cost < math.inf else "inf"
    print(f"\nBaselines: greedy={gc}  nn={nc}")

    # Phase 1: tuning
    print(f"\n=== Phase 1: Parameter Tuning ({tune_time:.1f}s) ===")
    best_cfg, tune_cost, tune_path = tune_parameters(
        problem, tune_budget=tune_time, verbose=True)
    if tune_cost < best_cost:
        best_cost, best_path = tune_cost, tune_path

    # Phase 2: solve
    print(f"\n=== Phase 2: Solve with {best_cfg.label} ({solve_time:.1f}s) ===")
    solver = MCTSSolver(c=best_cfg.c, rollout_strategy=best_cfg.strategy,
                        greedy_prob=best_cfg.greedy_prob, reward_scale=best_cfg.reward_scale)
    mcts_cost, mcts_path = solver.solve(problem, time_limit=solve_time)
    if mcts_cost < best_cost:
        best_cost, best_path = mcts_cost, mcts_path
        print(f"  Phase 2 improved: {best_cost:.0f}")
    else:
        mc = f"{mcts_cost:.0f}" if mcts_cost < math.inf else "inf"
        bc = f"{best_cost:.0f}" if best_cost < math.inf else "inf"
        print(f"  Phase 2: {mc}  (best retained: {bc})")

    # Output
    print(f"\n=== Final Result ===")
    if best_cost < math.inf:
        ok = verify_solution(problem, best_path, best_cost)
        print(f"Total cost : {best_cost:.0f}  [{'VALID' if ok else 'INVALID'}]")
        print(f"Path ({len(best_path)} airports): {best_path}")
    else:
        print("No feasible solution found.")

    return best_cost, best_path


if __name__ == '__main__':
    main()

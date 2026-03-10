import math
from typing import Dict, List, Tuple


class Problem:
    def __init__(self, filename: str):
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

    def flight_cost(self, frm: str, to: str, day: int) -> float:
        return min(self.costs.get((frm, to, day), math.inf),
                   self.costs.get((frm, to, 0),   math.inf))


def validate(instance_file: str, path_input):
    """
    Validate a route and compute its cost.

    Parameters
    ----------
    instance_file : str         path to the .in file
    path_input    : str | list  space-separated codes or a list

    Returns
    -------
    Prints a formatted report and returns (valid: bool, total_cost: float).
    """
    path = path_input.strip().split() if isinstance(path_input, str) else list(path_input)
    p = Problem(instance_file)
    errors = []
    flights = []
    total_cost = 0.0

    # 1. correct length
    if len(path) != p.n_areas + 1:
        errors.append(f"Wrong path length: got {len(path)}, expected {p.n_areas + 1}")

    # 2. starts at declared airport
    if path[0] != p.start_airport:
        errors.append(f"Must start at '{p.start_airport}', got '{path[0]}'")

    # 3. all codes exist
    unknown = [ap for ap in path if ap not in p.airport_to_area]
    if unknown:
        errors.append(f"Unknown airport(s): {unknown}")
        _print_report(path, p, errors, flights, total_cost)
        return False, math.inf

    # 4. ends in start area
    last_area = p.airport_to_area[path[-1]]
    if last_area != p.start_area:
        errors.append(
            f"Must end in start area {p.start_area} ('{p.area_names[p.start_area]}'), "
            f"but '{path[-1]}' is in area {last_area} ('{p.area_names[last_area]}')"
        )

    # 5. each area visited exactly once
    visited: Dict[int, str] = {}
    for ap in path[1:]:
        a = p.airport_to_area[ap]
        if a in visited:
            errors.append(f"Area {a} ('{p.area_names[a]}') visited twice: '{visited[a]}' and '{ap}'")
        else:
            visited[a] = ap

    # 6. no area missing
    missing = [i for i in range(p.n_areas) if i not in visited]
    if missing:
        errors.append(f"Area(s) not visited: {[p.area_names[i] for i in missing]}")

    # 7. every flight exists
    for day, (frm, to) in enumerate(zip(path, path[1:]), start=1):
        cost = p.flight_cost(frm, to, day)
        ok = cost < math.inf
        flights.append({"day": day, "from": frm, "to": to,
                        "area": p.airport_to_area[to],
                        "area_name": p.area_names[p.airport_to_area[to]],
                        "cost": cost, "ok": ok})
        if not ok:
            errors.append(f"Day {day}: no flight '{frm}' → '{to}'")
        else:
            total_cost += cost

    valid = not errors
    _print_report(path, p, errors, flights, total_cost, valid)
    return valid, (total_cost if valid else math.inf)


def _print_report(path, p, errors, flights, total_cost, valid=False):
    W = 66
    print("=" * W)
    if valid:
        print(f"  ✅  VALID   —   total cost = {total_cost:,.0f}")
    else:
        print(f"  ❌  INVALID —   {len(errors)} error(s) found")
    print("=" * W)

    if errors:
        print()
        for i, e in enumerate(errors, 1):
            print(f"  [{i}] {e}")

    if flights:
        print()
        print(f"  {'Day':>3}  {'From':>5} → {'To':<5}  {'Area':<22}  {'Cost':>8}  {'':>2}")
        print("  " + "-" * (W - 2))
        for f in flights:
            cost_s = f"{f['cost']:>8,.0f}" if f["ok"] else "     inf"
            flag   = "✓" if f["ok"] else "✗"
            print(f"  {f['day']:>3}  {f['from']:>5} → {f['to']:<5}  "
                  f"{f['area_name']:<22}  {cost_s}  {flag}")
        print("  " + "-" * (W - 2))
        if valid:
            print(f"  {'':>3}  {'':>5}   {'':5}  {'TOTAL':>22}  {total_cost:>8,.0f}")
    print()

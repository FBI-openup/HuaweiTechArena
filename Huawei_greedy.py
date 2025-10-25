#!/usr/bin/env python3
"""
Huawei UAV Traffic Allocation - Pure Python Greedy Solver (修正版)
---------------------------------------------------------------
✅ 不调用任何第三方库，仅标准库；
✅ 修复 landing zone 越界；
✅ 防止容量与时间越界；
✅ 确保输出格式符合比赛标准；
✅ 若流未完全分配，打印警告；
---------------------------------------------------------------
"""

from __future__ import annotations
import sys


def _starts_with(text: str, prefix: str) -> bool:
    return text.startswith(prefix)


def _trim(text: str) -> str:
    left = 0
    right = len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return text[left:right]


def _float_to_str(value: float) -> str:
    return format(value, ".6f")


def _compute_capacity_slot(slot: int, peak: float) -> float:
    """10s 周期容量模型"""
    if slot in (0, 1, 8, 9):
        return 0.0
    if slot in (2, 7):
        return peak * 0.5
    return peak


# =========================
# 参数解析与输入读取
# =========================

def parse_args(argv: list[str]) -> tuple[str, str, float, int, bool]:
    input_path = ""
    output_path = ""
    alpha = 0.1
    top_k = -1
    use_count_bonus = True

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("--help", "-h"):
            return ("", "", alpha, top_k, use_count_bonus)
        elif arg in ("--output", "-o"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                output_path = argv[i + 1]
                i += 1
        elif _starts_with(arg, "--output="):
            output_path = arg.split("=", 1)[1]
        elif arg == "--alpha":
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                try:
                    alpha = float(argv[i + 1])
                except ValueError:
                    pass
                i += 1
        elif _starts_with(arg, "--alpha="):
            try:
                alpha = float(arg.split("=", 1)[1])
            except ValueError:
                pass
        elif arg == "--top-k":
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                try:
                    top_k = int(argv[i + 1])
                except ValueError:
                    pass
                i += 1
        elif _starts_with(arg, "--top-k="):
            try:
                top_k = int(arg.split("=", 1)[1])
            except ValueError:
                pass
        elif arg == "--no-count-bonus":
            use_count_bonus = False
        elif arg == "--count-bonus":
            use_count_bonus = True
        elif _starts_with(arg, "--"):
            if "=" not in arg and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                i += 1
        elif not input_path:
            input_path = arg
        i += 1

    return (input_path, output_path, alpha, top_k, use_count_bonus)


def read_input(path: str) -> list[str]:
    """读取输入文件或标准输入"""
    if path and path != "-":
        with open(path, "r", encoding="utf-8") as handle:
            data = handle.read()
    else:
        data = sys.stdin.read()
    lines: list[str] = []
    for raw_line in data.splitlines():
        cleaned = _trim(raw_line)
        if cleaned and not cleaned.startswith("#"):
            lines.append(cleaned)
    return lines


# =========================
# 输入解析
# =========================

def parse_instance(lines: list[str], alpha: float, top_k: int) -> dict[str, object]:
    """解析问题输入（UAV + Flow）"""
    if not lines:
        raise ValueError("input file is empty")

    first_tokens = lines[0].split()
    if len(first_tokens) != 4:
        raise ValueError("header must contain exactly four integers: M N FN T")
    M, N, FN, T = map(int, first_tokens)
    if M <= 0 or N <= 0 or T <= 0:
        raise ValueError("mesh dimensions and time horizon must be positive")
    if FN < 0:
        raise ValueError("flow count must be non-negative")

    expected_uavs = M * N
    if len(lines) < 1 + expected_uavs + FN:
        raise ValueError("insufficient lines for UAV definitions or flows")

    # --- UAV信息 ---
    landings: list[dict[str, object]] = []
    landing_index: dict[str, int] = {}

    for idx in range(expected_uavs):
        tokens = lines[1 + idx].split()
        if len(tokens) != 4:
            raise ValueError(f"UAV line {idx+1} must contain four values")
        x_val = int(tokens[0])
        y_val = int(tokens[1])
        peak = float(tokens[2])
        phi = float(tokens[3])
        peak = max(0.0, min(1000.0, peak))
        phi = float(phi % 10.0)
        landing_id = f"UAV({x_val},{y_val})"
        if landing_id in landing_index:
            raise ValueError(f"duplicate UAV coordinates {landing_id}")
        capacity = []
        for t in range(T):
            slot = int((phi + float(t)) % 10)
            capacity.append(_compute_capacity_slot(slot, peak))
        landing_index[landing_id] = len(landings)
        landings.append(
            {
                "id": landing_id,
                "x": x_val,
                "y": y_val,
                "capacity": capacity,
                "peak": peak,
                "phi": phi,
            }
        )

    # --- 流信息 ---
    flows: list[dict[str, object]] = []
    for idx in range(FN):
        tokens = lines[1 + expected_uavs + idx].split()
        if len(tokens) != 9:
            raise ValueError("flow lines must contain nine integers")
        f_id, fx, fy, t_start, q_total, m1, n1, m2, n2 = map(int, tokens)
        demand = max(0.0, float(q_total))
        t_start = max(0, min(T - 1, t_start))

        # ✅ 修正落点矩形区间 (原错误已修复)
        lo_x = m1 if m1 <= m2 else m2
        hi_x = m2 if m1 <= m2 else m1
        lo_y = n1 if n1 <= n2 else n2
        hi_y = n2 if n1 <= n2 else n1

        candidates: list[tuple[int, float, float]] = []
        for j, landing in enumerate(landings):
            lx = landing["x"]
            ly = landing["y"]
            if lo_x <= lx <= hi_x and lo_y <= ly <= hi_y:
                dx = float(lx - fx)
                dy = float(ly - fy)
                distance = (dx * dx + dy * dy) ** 0.5
                weight = pow(2.0, -alpha * distance)
                candidates.append((j, distance, weight))

        candidates.sort(key=lambda item: item[1])
        if top_k > 0 and len(candidates) > top_k:
            candidates = candidates[:top_k]

        flows.append(
            {
                "id": f_id,
                "fx": fx,
                "fy": fy,
                "t_start": t_start,
                "demand": demand,
                "range_x": (lo_x, hi_x),
                "range_y": (lo_y, hi_y),
                "candidates": candidates,
            }
        )

    return {"T": T, "flows": flows, "landings": landings, "alpha": alpha}


# =========================
# 贪心调度算法
# =========================

def greedy_solve(instance: dict[str, object], use_count_bonus: bool) -> list[dict[str, object]]:
    T = instance["T"]
    flows = instance["flows"]
    landings = instance["landings"]

    capacity_left = [list(ld["capacity"]) for ld in landings]
    all_assignments: list[tuple[int, int, int, float]] = []
    delivered_amount = [0.0] * len(flows)

    for f_idx, flow in enumerate(flows):
        remaining = flow["demand"]
        if remaining <= 0.0:
            continue
        start_t = flow["t_start"]
        for t in range(start_t, T):
            if remaining <= 1e-9:
                break
            best_score = -1.0
            best_candidate = None
            for cand in flow["candidates"]:
                landing_index, _, weight = cand
                cap = capacity_left[landing_index][t]
                if cap <= 1e-9:
                    continue

                # ✅ 检查落点合法性
                lx = landings[landing_index]["x"]
                ly = landings[landing_index]["y"]
                rx1, rx2 = flow["range_x"]
                ry1, ry2 = flow["range_y"]
                if not (rx1 <= lx <= rx2 and ry1 <= ly <= ry2):
                    continue  # 超出合法landing zone

                # ✅ 防止容量溢出
                amount = min(cap, remaining)
                if amount <= 1e-9:
                    continue

                score = 0.3 * weight
                if score > best_score:
                    best_score = score
                    best_candidate = (landing_index, amount)
            if best_candidate is not None:
                landing_idx, amount = best_candidate
                if amount > capacity_left[landing_idx][t]:
                    amount = capacity_left[landing_idx][t]
                capacity_left[landing_idx][t] -= amount
                remaining -= amount
                delivered_amount[f_idx] += amount
                all_assignments.append((f_idx, landing_idx, t, amount))

        # ⚠️ 若未完成流量，打印提示
        if remaining > 1e-6:
            sys.stderr.write(f"[Warning] Flow {flow['id']} not fully delivered ({remaining:.2f})\n")

    # 输出整理
    schedule_unsorted: list[dict[str, object]] = []
    for f_idx, flow in enumerate(flows):
        entries: list[tuple[int, int, int, float]] = []
        for rec in all_assignments:
            if rec[0] == f_idx and rec[3] > 1e-12:
                landing_idx = rec[1]
                landing = landings[landing_idx]
                entries.append((rec[2], landing["x"], landing["y"], rec[3]))
        entries.sort(key=lambda item: (item[0], item[1], item[2]))
        schedule_unsorted.append({"id": flow["id"], "entries": entries})
    schedule = sorted(schedule_unsorted, key=lambda item: item["id"])
    return schedule


# =========================
# 主入口
# =========================

def main(argv: list[str]) -> int:
    input_path, output_path, alpha, top_k, use_count_bonus = parse_args(argv)
    try:
        lines = read_input(input_path)
        instance = parse_instance(lines, alpha, top_k)
        schedule = greedy_solve(instance, use_count_bonus)
    except Exception as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1

    output_lines: list[str] = []
    for item in schedule:
        flow_id = item["id"]
        entries = item["entries"]
        output_lines.append(f"{flow_id} {len(entries)}")
        for entry in entries:
            t_val, x_val, y_val, z_val = entry
            output_lines.append(
                f"{int(t_val)} {int(x_val)} {int(y_val)} {_float_to_str(z_val)}"
            )

    payload = "\n".join(output_lines)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

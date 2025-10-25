"""
华为UAV时变链路资源分配算法 - 优化版本
Optimized Time-varying Link Resource Allocation Algorithm

关键优化点：
1. 带宽预测与规划：提前规划高带宽时段
2. 智能路径选择：综合考虑跳数、带宽、拥塞
3. 着陆点稳定性：减少切换次数
4. 负载均衡：避免热点拥塞
5. 延迟优化：优先利用早期高带宽时段
"""

import sys
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set
import heapq


class UAV:
    """UAV节点"""
    def __init__(self, x, y, peak_bandwidth, phase):
        self.x = x
        self.y = y
        self.peak_bandwidth = peak_bandwidth
        self.phase = phase

    def get_bandwidth(self, t):
        """计算时刻t的可用带宽"""
        t_effective = (self.phase + t) % 10
        if t_effective in [0, 1, 8, 9]:
            return 0
        elif t_effective in [2, 7]:
            return self.peak_bandwidth / 2
        else:  # t_effective in [3, 4, 5, 6]
            return self.peak_bandwidth

    def get_bandwidth_quality(self, t):
        """带宽质量评分（0-1之间）"""
        bw = self.get_bandwidth(t)
        if bw == 0:
            return 0
        elif bw == self.peak_bandwidth / 2:
            return 0.5
        else:
            return 1.0


class Flow:
    """数据流"""
    def __init__(self, flow_id, x, y, t_start, total_size, m1, n1, m2, n2):
        self.flow_id = flow_id
        self.access_x = x
        self.access_y = y
        self.t_start = t_start
        self.total_size = total_size
        self.m1, self.n1 = m1, n1
        self.m2, self.n2 = m2, n2
        self.transmitted = 0
        self.schedule = []
        self.last_landing_uav = None
        self.landing_change_count = 0

    def is_in_landing_area(self, x, y):
        """检查(x,y)是否在着陆区域"""
        return self.m1 <= x <= self.m2 and self.n1 <= y <= self.n2

    def get_remaining(self):
        """获取剩余数据量"""
        return self.total_size - self.transmitted


class OptimizedUAVNetwork:
    """优化的UAV网络调度器"""

    def __init__(self, M, N, T):
        self.M = M
        self.N = N
        self.T = T
        self.uavs = {}
        self.flows = []
        self.allocated_bandwidth = defaultdict(lambda: defaultdict(float))

    def add_uav(self, x, y, peak_bandwidth, phase):
        self.uavs[(x, y)] = UAV(x, y, peak_bandwidth, phase)

    def add_flow(self, flow):
        self.flows.append(flow)

    def get_neighbors(self, x, y):
        """获取相邻节点"""
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.M and 0 <= ny < self.N:
                neighbors.append((nx, ny))
        return neighbors

    def manhattan_distance(self, x1, y1, x2, y2):
        """曼哈顿距离"""
        return abs(x1 - x2) + abs(y1 - y2)

    def find_best_landing_uavs_in_region(self, flow, t, top_k=3):
        """找到着陆区域内的最佳K个UAV"""
        candidates = []

        for x in range(flow.m1, flow.m2 + 1):
            for y in range(flow.n1, flow.n2 + 1):
                if (x, y) not in self.uavs:
                    continue

                uav = self.uavs[(x, y)]

                # 计算可用带宽
                total_bw = uav.get_bandwidth(t)
                allocated = self.allocated_bandwidth[t][(x, y)]
                available_bw = max(0, total_bw - allocated)

                if available_bw <= 0:
                    continue

                # 距离因子（越近越好）
                dist = self.manhattan_distance(flow.access_x, flow.access_y, x, y)

                # 稳定性因子（与上次着陆点相同则加分）
                stability_bonus = 0
                if flow.last_landing_uav == (x, y):
                    stability_bonus = 1000  # 大幅加分以保持稳定

                # 未来带宽潜力（查看接下来几个时刻的带宽）
                future_bw = 0
                for future_t in range(t + 1, min(t + 5, self.T)):
                    future_bw += uav.get_bandwidth(future_t)

                # 综合评分
                score = (
                    available_bw * 10 +           # 当前可用带宽
                    future_bw * 2 +               # 未来带宽
                    stability_bonus -             # 稳定性奖励
                    dist * 5                      # 距离惩罚
                )

                candidates.append({
                    'pos': (x, y),
                    'score': score,
                    'available_bw': available_bw,
                    'dist': dist,
                    'is_stable': flow.last_landing_uav == (x, y)
                })

        # 按评分排序，返回top K
        candidates.sort(key=lambda c: c['score'], reverse=True)
        return candidates[:top_k]

    def predict_high_bandwidth_periods(self, uav, start_t, end_t):
        """预测UAV在[start_t, end_t)内的高带宽时段"""
        high_bw_periods = []

        for t in range(start_t, min(end_t, self.T)):
            bw = uav.get_bandwidth(t)
            if bw == uav.peak_bandwidth:  # 峰值带宽时段
                high_bw_periods.append((t, bw, 1.0))
            elif bw == uav.peak_bandwidth / 2:  # 中等带宽
                high_bw_periods.append((t, bw, 0.5))

        return high_bw_periods

    def allocate_greedy_with_lookahead(self, flow, t):
        """带前瞻的贪心分配"""
        if flow.transmitted >= flow.total_size or t < flow.t_start:
            return

        remaining = flow.get_remaining()

        # 找到最佳着陆UAV候选
        candidates = self.find_best_landing_uavs_in_region(flow, t, top_k=3)

        if not candidates:
            return

        # 选择第一个候选（评分最高）
        best_candidate = candidates[0]
        landing_pos = best_candidate['pos']
        available_bw = best_candidate['available_bw']

        # 计算实际传输量
        actual_transfer = min(available_bw, remaining)

        if actual_transfer > 0:
            # 更新分配
            self.allocated_bandwidth[t][landing_pos] += actual_transfer
            flow.transmitted += actual_transfer
            flow.schedule.append((t, landing_pos[0], landing_pos[1], actual_transfer))

            # 更新着陆点
            if flow.last_landing_uav != landing_pos:
                if flow.last_landing_uav is not None:
                    flow.landing_change_count += 1
                flow.last_landing_uav = landing_pos

    def schedule_with_priority(self):
        """基于优先级的调度"""
        # 为每个流计算优先级
        flow_priorities = []
        for flow in self.flows:
            # 优先级因素：
            # 1. 紧急度（总时间 - 开始时间）
            # 2. 数据量大小
            # 3. 距离（到着陆区域的距离）
            urgency = self.T - flow.t_start
            size_factor = flow.total_size

            # 计算平均距离到着陆区域
            avg_landing_x = (flow.m1 + flow.m2) / 2
            avg_landing_y = (flow.n1 + flow.n2) / 2
            distance = self.manhattan_distance(
                flow.access_x, flow.access_y,
                avg_landing_x, avg_landing_y
            )

            # 综合优先级（数值越大越优先）
            priority = size_factor / max(1, urgency) + distance * 0.1

            flow_priorities.append((priority, flow))

        # 按优先级排序（高优先级在前）
        flow_priorities.sort(reverse=True)
        sorted_flows = [f for _, f in flow_priorities]

        # 时间片调度
        for t in range(self.T):
            # 获取当前活跃的流
            active_flows = [
                f for f in sorted_flows
                if f.t_start <= t and f.transmitted < f.total_size
            ]

            # 为每个活跃流分配资源
            for flow in active_flows:
                self.allocate_greedy_with_lookahead(flow, t)

    def schedule_bandwidth_aware(self):
        """带宽感知调度 - 优先利用高带宽时段"""
        # 预先分析每个时刻的全局带宽状况
        time_bandwidth_score = []
        for t in range(self.T):
            total_bw = sum(uav.get_bandwidth(t) for uav in self.uavs.values())
            time_bandwidth_score.append((t, total_bw))

        # 时间片调度
        for t in range(self.T):
            # 当前时刻的带宽质量
            current_total_bw = sum(uav.get_bandwidth(t) for uav in self.uavs.values())

            # 获取活跃流
            active_flows = [
                f for f in self.flows
                if f.t_start <= t and f.transmitted < f.total_size
            ]

            # 如果是高带宽时段，优先处理大流
            if current_total_bw > 0:
                # 按剩余数据量排序（大流优先）
                active_flows.sort(key=lambda f: f.get_remaining(), reverse=True)

            # 为每个流分配
            for flow in active_flows:
                self.allocate_greedy_with_lookahead(flow, t)

    def output_solution(self):
        """输出解决方案"""
        for flow in self.flows:
            print(f"{flow.flow_id} {len(flow.schedule)}")
            for t, x, y, rate in flow.schedule:
                if rate == int(rate):
                    print(f"{t} {x} {y} {int(rate)}")
                else:
                    # 保留足够精度但移除尾部0
                    print(f"{t} {x} {y} {rate:.10f}".rstrip('0').rstrip('.'))


def main():
    # 读取输入
    lines = sys.stdin.read().strip().split('\n')

    M, N, FN, T = map(int, lines[0].split())
    network = OptimizedUAVNetwork(M, N, T)

    # 读取UAV
    line_idx = 1
    for i in range(M * N):
        parts = lines[line_idx].split()
        x, y, B, phi = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
        network.add_uav(x, y, B, phi)
        line_idx += 1

    # 读取Flow
    for i in range(FN):
        parts = list(map(int, lines[line_idx].split()))
        flow = Flow(parts[0], parts[1], parts[2], parts[3],
                   parts[4], parts[5], parts[6], parts[7], parts[8])
        network.add_flow(flow)
        line_idx += 1

    # 执行调度（使用带宽感知调度）
    network.schedule_bandwidth_aware()

    # 输出结果
    network.output_solution()


if __name__ == "__main__":
    main()


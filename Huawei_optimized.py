"""
华为UAV时变链路资源分配算法 - 优化版本 v2.1.4
Optimized Time-varying Link Resource Allocation Algorithm

版本历史:
- v1.0 (2025-10-25): 基础贪心算法 - 得分 7078.09
- v2.0 (2025-10-26): 边际收益优化 - 得分 7119.11 (+41.02, +0.58%)
- v2.1 (2025-10-26): 流调度顺序优化 - 超时
- v2.1.2 (2025-10-26): 性能优化（采样估算）- 得分 7132.27 (+13.16, +0.18%)
- v2.1.3 (2025-10-26): 精度提升优化 - 得分 7134.85 (+2.58, +0.04%)
- v2.1.4 (2025-10-26): 权重微调优化 - 得分 7148.32 (+13.47, +0.19%)

v2.1.4 权重微调优化:
12. 距离衰减系数提升: alpha从0.1提升到0.15，强化近距离偏好
13. 紧急度权重调整: 需求率×12(+20%), 稀缺性×60(+20%), 数据量×1.5(+50%)
14. 候选过滤优化: 过滤传输量<剩余量1%的低效候选

v2.1.3 精度提升优化:
9. 增强采样精度: 大区域采样点从5个增加到9个（四角+四边中点+中心）
10. 提升精确计算阈值: 小区域阈值从9格提升到16格（≤4x4精确计算）
11. 增加候选数量: Top-K从5增加到7，提升选择质量

v2.1.2 性能优化:
8. 采样式候选计数: 小区域(≤9格)精确计算，大区域采样5点估算，避免O(W×H)遍历

v2.1 新增优化:
6. 动态紧急度评分: calculate_flow_urgency() 综合考虑平均需求率、候选稀缺性、数据量
7. 智能流优先级: 每个时刻动态计算流紧急度，紧急流优先调度

v2.0 关键优化:
1. 边际收益评分函数: 融合四项评分标准的数学建模
2. 流量单位化: amount/total_size，避免大流垄断
3. 时间衰减因子: 10/(t+10)，激励早期传输
4. 指数距离惩罚: 2^(-0.1*dist)，强制近距离选择
5. 着陆点切换追踪: 动态计算k值，减少切换

算法核心思想:
从「局部最优（最大化单次带宽）」转向「全局加权最优（最大化评分函数期望）」
通过精确建模题目评分标准，让贪心决策与最终目标高度对齐。
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
        self.used_landing_positions = set()  # 记录使用过的不同着陆点

    def is_in_landing_area(self, x, y):
        """检查(x,y)是否在着陆区域"""
        return self.m1 <= x <= self.m2 and self.n1 <= y <= self.n2

    def get_remaining(self):
        """获取剩余数据量"""
        return self.total_size - self.transmitted

    def get_current_k(self):
        """获取当前使用的不同着陆点数量k"""
        return len(self.used_landing_positions)


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

    def calculate_allocation_score(self, flow, landing_pos, t, amount):
        """
        计算分配的边际收益评分
        按照题目四项评分标准的权重计算：
        - U2G流量得分 (40%)
        - 延迟得分 (20%)
        - 距离得分 (30%)
        - 着陆点得分 (10%)
        """
        # 计算距离
        dist = self.manhattan_distance(flow.access_x, flow.access_y,
                                       landing_pos[0], landing_pos[1])

        # 1. U2G流量得分 (40%) - 归一化到总数据量
        u2g_score = 0.4 * (amount / flow.total_size)

        # 2. 延迟得分 (20%) - 时间衰减因子
        # 公式: τ / (t_i + τ)，其中 τ = 10, t_i 是相对开始时间的延迟
        tau = 10
        delay_from_start = t - flow.t_start
        delay_score = 0.2 * (tau / (delay_from_start + tau))

        # 3. 距离得分 (30%) - 指数衰减
        # 公式: 2^(-λ * h)，其中 λ = 0.1, h 是跳数（这里用曼哈顿距离近似）
        alpha = 0.15  # 提升衰减系数（0.1 → 0.15），更强力惩罚远距离
        distance_score = 0.3 * (2 ** (-alpha * dist))

        # 4. 着陆点得分 (10%) - 1/k，k是使用的不同着陆点数
        # 如果选择新的着陆点，k会增加
        current_k = flow.get_current_k()
        if landing_pos in flow.used_landing_positions or current_k == 0:
            # 继续使用已有着陆点，k不变
            landing_score = 0.1 * (1.0 / max(1, current_k))
        else:
            # 使用新着陆点，k会+1
            landing_score = 0.1 * (1.0 / (current_k + 1))

        # 返回综合评分
        total_score = u2g_score + delay_score + distance_score + landing_score

        return total_score

    def find_best_landing_uavs_in_region(self, flow, t, top_k=3):
        """找到着陆区域内的最佳K个UAV - 使用边际收益评分"""
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

                landing_pos = (x, y)

                # 计算可能的传输量（不超过剩余数据量）
                potential_amount = min(available_bw, flow.get_remaining())

                # 🔥 优化：过滤太小的传输量（低于剩余量的1%）
                if potential_amount < flow.get_remaining() * 0.01:
                    continue

                # 使用新的边际收益评分函数
                marginal_score = self.calculate_allocation_score(
                    flow, landing_pos, t, potential_amount
                )

                # 额外考虑：未来带宽潜力（作为tie-breaker）
                future_bw = 0
                for future_t in range(t + 1, min(t + 5, self.T)):
                    future_bw += uav.get_bandwidth(future_t)

                # 最终评分：边际收益为主，未来带宽作为微调
                final_score = marginal_score * 1000 + future_bw * 0.01

                candidates.append({
                    'pos': landing_pos,
                    'score': final_score,
                    'marginal_score': marginal_score,
                    'available_bw': available_bw,
                    'potential_amount': potential_amount,
                    'dist': self.manhattan_distance(flow.access_x, flow.access_y, x, y),
                    'is_stable': flow.last_landing_uav == landing_pos
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
        """带前瞻的贪心分配 - 使用边际收益评分"""
        if flow.transmitted >= flow.total_size or t < flow.t_start:
            return

        remaining = flow.get_remaining()

        # 找到最佳着陆UAV候选（使用边际收益评分，增加候选数到7）
        candidates = self.find_best_landing_uavs_in_region(flow, t, top_k=7)

        if not candidates:
            return

        # 选择评分最高的候选
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

            # 更新着陆点集合（用于计算k值）
            flow.used_landing_positions.add(landing_pos)

            # 更新最后使用的着陆点
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

    def calculate_flow_urgency(self, flow, t):
        """
        计算流在时刻t的紧急度评分（性能优化版v2：增强采样精度）
        综合考虑：
        1. 平均需求率（剩余量/剩余时间）- 越高越紧急
        2. 可用候选数量（增强采样）- 越少越紧急
        3. 数据量 - 大流适当优先
        """
        remaining = flow.get_remaining()
        remaining_time = max(1, self.T - t)

        # 1. 平均需求率（每秒需要传输的数据量）
        avg_demand_rate = remaining / remaining_time

        # 2. 🔥 增强采样式候选计数（提升精度）
        region_width = flow.m2 - flow.m1 + 1
        region_height = flow.n2 - flow.n1 + 1
        region_size = region_width * region_height

        if region_size <= 16:  # 小区域（≤4x4）直接精确计算
            available_candidates = sum(
                1 for x in range(flow.m1, flow.m2 + 1)
                for y in range(flow.n1, flow.n2 + 1)
                if (x, y) in self.uavs and
                   self.uavs[(x, y)].get_bandwidth(t) - self.allocated_bandwidth[t][(x, y)] > 0
            )
        else:  # 大区域采样估算（增加采样点到9个）
            # 增强采样策略：四角 + 四边中点 + 中心点（9个点）
            mid_x = (flow.m1 + flow.m2) // 2
            mid_y = (flow.n1 + flow.n2) // 2

            sample_positions = [
                (flow.m1, flow.n1),      # 左上角
                (flow.m2, flow.n2),      # 右下角
                (flow.m1, flow.n2),      # 左下角
                (flow.m2, flow.n1),      # 右上角
                (mid_x, flow.n1),        # 上边中点
                (mid_x, flow.n2),        # 下边中点
                (flow.m1, mid_y),        # 左边中点
                (flow.m2, mid_y),        # 右边中点
                (mid_x, mid_y)           # 中心点
            ]

            sampled_count = sum(
                1 for (x, y) in sample_positions
                if (x, y) in self.uavs and
                   self.uavs[(x, y)].get_bandwidth(t) - self.allocated_bandwidth[t][(x, y)] > 0
            )

            # 根据采样比例估算总数（9点采样）
            available_candidates = max(1, int(sampled_count * region_size / 9))

        # 候选少则更紧急（避免除零）
        candidate_scarcity = 1.0 / max(1, available_candidates)

        # 3. 数据量因子（适度考虑）
        size_factor = remaining / 1000.0  # 归一化

        # 综合紧急度评分（微调权重）
        urgency = (
            avg_demand_rate * 12.0 +      # 需求率权重（提升20%）
            candidate_scarcity * 60.0 +   # 候选稀缺性（提升20%）
            size_factor * 1.5             # 数据量（提升50%）
        )

        return urgency

    def schedule_bandwidth_aware(self):
        """带宽感知调度 - 优先利用高带宽时段，动态调整流优先级"""
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

            if not active_flows:
                continue

            # 动态计算每个流的紧急度并排序
            # 使用字典存储紧急度，避免排序时比较Flow对象
            flow_urgency_dict = {}
            for flow in active_flows:
                urgency = self.calculate_flow_urgency(flow, t)
                flow_urgency_dict[id(flow)] = urgency

            # 按紧急度从高到低排序（使用flow对象的id作为稳定排序依据）
            sorted_flows = sorted(active_flows,
                                 key=lambda f: (flow_urgency_dict[id(f)], id(f)),
                                 reverse=True)

            # 为每个流分配（紧急流优先）
            for flow in sorted_flows:
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


import numpy as np

# WOA (Whale Optimization Algorithm) 類別
class WOA:
    def __init__(self, obj_func, dim, pop_size, max_iter, bounds, tol=1e-6, patience=10):
        self.obj_func = obj_func  # 目標函數
        self.dim = dim  # 問題的維度
        self.pop_size = pop_size  # 群體大小
        self.max_iter = max_iter  # 最大迭代次數
        self.bounds = bounds  # 搜索範圍
        self.tol = tol  # 早停容忍度
        self.patience = patience  # 早停計數器

    def optimize(self):
        alpha, beta, delta = None, None, None
        alpha_score, beta_score, delta_score = np.inf, np.inf, np.inf
        whales = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))  # 初始化位置
        history = []  # 記錄收斂過程

        no_improve_count = 0  # 早停計數器

        for iter_num in range(self.max_iter):
            scores = np.array([self.obj_func(whale) for whale in whales])  # 計算每個鯨魚的位置的適應度
            sorted_indices = np.argsort(scores)  # 根據適應度排序
            alpha, beta, delta = whales[sorted_indices[:3]]  # 取最好的3個解
            alpha_score_new = scores[sorted_indices[0]]
            beta_score, delta_score = scores[sorted_indices[1:3]]

            # 檢查早停條件
            if abs(alpha_score_new - alpha_score) < self.tol:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    print(f"Early stopping at iteration {iter_num}")
                    break
            else:
                no_improve_count = 0  # 重置計數器

            alpha_score = alpha_score_new
            history.append(alpha_score)  # 記錄每次迭代的最優適應度

            a = 2 - iter_num * (2 / self.max_iter)  # 控制探索和開發的平衡

            for i in range(self.pop_size):  # 更新每個鯨魚的位置
                for j in range(self.dim):
                    p = np.random.rand()  # 隨機決定使用包圍還是螺旋更新機制
                    if p < 0.5:
                        # 包圍機制（Encircling prey）
                        r1, r2 = np.random.rand(), np.random.rand()
                        A = 2 * a * r1 - a
                        C = 2 * r2
                        D_alpha = abs(C * alpha[j] - whales[i, j])
                        whales[i, j] = alpha[j] - A * D_alpha
                    else:
                        # 螺旋更新機制（Spiral updating）
                        b = 1  # log-spiral constant
                        l = np.random.uniform(-1, 1)
                        distance_to_leader = abs(alpha[j] - whales[i, j])
                        whales[i, j] = distance_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l) + alpha[j]

                    # 邊界控制
                    whales[i, j] = np.clip(whales[i, j], self.bounds[j, 0], self.bounds[j, 1])

        return alpha, alpha_score, history  # 返回最優解和收斂過程

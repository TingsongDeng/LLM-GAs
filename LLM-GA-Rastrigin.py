import numpy as np
import random
import time
import json
import re
import os
from typing import Callable, List, Tuple, Union
from openai import OpenAI


class EnhancedDeepSeekGA:
    def __init__(self,
                 objective_func: Callable[[List[float]], float],
                 dim: int = 10,
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9,
                 use_deepseek: bool = True,
                 deepseek_temperature: float = 0.4,
                 deepseek_api_key: str = None):

        # 核心参数
        self.objective_func = objective_func
        self.dim = dim
        self.population_size = population_size
        self.base_mutation_rate = mutation_rate
        self.base_crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.deepseek_temperature = deepseek_temperature

        # API控制参数
        self.api_failure_count = 0
        self.max_api_failures = 10
        self.consecutive_failures = 0
        self.deepseek_api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        self.use_deepseek = use_deepseek if self.deepseek_api_key else False
        self.last_improvement = 0

        # 初始化种群和历史记录
        self.population = self.initialize_population()
        self.best_history = []
        self.diversity_history = []

    def get_optimization_context_prompt(self) -> str:
        """生成包含完整优化任务描述的提示前缀"""
        return f"""
        ## 优化任务说明
        你正在协助解决一个{self.dim}维连续优化问题，需要最小化Rastrigin函数：

        ### 目标函数表达式
        f(x) = 10·n + Σ[x_i² - 10·cos(2πx_i)]   (i=1→n)

        ### 优化要求
        1. 理论最小值：0 (当所有x_i=0时取得)
        2. 变量范围：x_i ∈ [-5.0, 5.0]
        3. 典型特征：多模态、非线性、高度震荡

        ## 你的任务
        请基于遗传算法原理，帮助生成：
        - 优质的交叉后代（保持多样性同时继承父代优点）
        - 有效的变异向量（跳出局部最优）
        
        ## 输出要求
        1. 返回包含{self.dim}维向量的JSON对象
        2. 接受字段名如"vector"/"child"/"mutation"等
        3. 示例正确响应：
        {{"optimized_vector": [1.2345,-2.3456,3.4567]}}
        或
        {{"children": [[1.2345,-2.3456], [3.4567,-4.5678]]}}
        """

    def initialize_population(self) -> List[np.ndarray]:
        """初始化具有多样性的种群"""
        return [np.random.uniform(-5, 5, self.dim) for _ in range(self.population_size)]

    def evaluate_population(self) -> List[float]:
        """评估种群适应度"""
        return [self.objective_func(ind) for ind in self.population]

    def calculate_diversity(self) -> float:
        """计算种群多样性（基因标准差）"""
        return np.mean(np.std(self.population, axis=0))

    def select_parents(self, fitness: List[float]) -> List[np.ndarray]:
        """增强版锦标赛选择"""
        selected = []
        tournament_size = max(2, self.population_size // 10)
        for _ in range(self.population_size):
            candidates = np.random.choice(len(self.population), size=tournament_size, replace=False)
            winner = max(candidates, key=lambda x: fitness[x])
            selected.append(self.population[winner])
        return selected

    def call_deepseek_api(self, prompt: str, max_retries: int = 2) -> str:
        """强化版API调用方法"""
        if not self.use_deepseek or self.api_failure_count >= self.max_api_failures:
            raise ValueError("API调用被禁用")

        client = OpenAI(
            api_key=self.deepseek_api_key,
            base_url="https://api.deepseek.com"
        )

        # 强化系统提示
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个严格的数值优化助手。必须：\n"
                    "1. 只返回有效的JSON数组\n"
                    "2. 每个数值保留4位小数\n"
                    "3. 范围严格在[-5.0,5.0]\n"
                    "4. 禁止任何解释性文本\n"
                    "示例正确响应：\n"
                    "[[1.2345,-2.3456,3.4567],[4.5678,-5.0000,0.1234]]"
                )
            },
            {
                "role": "user",
                "content": prompt + "\n\n警告：必须严格按示例格式返回纯JSON数组！"
            }
        ]

        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=self.deepseek_temperature,
                    max_tokens=300,
                    top_p=0.8,
                    response_format={"type": "json_object"},
                    stream=False
                )

                content = response.choices[0].message.content
                if not self.validate_response(content):
                    raise ValueError("响应格式验证失败")

                return content

            except Exception as e:
                wait_time = min((2 ** attempt) + random.random(), 5)
                time.sleep(wait_time)
                continue

        self.api_failure_count += 1
        self.consecutive_failures += 1
        raise ValueError(f"API请求失败")

    def validate_response(self, response: str) -> bool:
        """响应内容预验证"""
        return response.strip().startswith(('[', '{')) and '"error"' not in response.lower()

    def parse_api_response(self, response: str, expected_items: int) -> List[List[float]]:
        """增强版解析器，处理字典错误响应"""
        try:
            parsed = json.loads(response.strip())
            # 新增：统一处理字典响应
            if isinstance(parsed, dict):
                if "error" in parsed:
                    raise ValueError(f"API返回错误: {parsed['error']}")
                # 尝试提取所有可能的数值数组
                vectors = [v for v in parsed.values()
                           if isinstance(v, list) and len(v) == self.dim]
                if len(vectors) >= expected_items:
                    return vectors[:expected_items]


            # 正常数组处理流程
            if isinstance(parsed, list):
                valid_vectors = [vec for vec in parsed if isinstance(vec, list) and len(vec) == self.dim]
                if len(valid_vectors) >= expected_items:
                    return valid_vectors[:expected_items]

            # 终极备用方案：正则提取
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if len(numbers) >= self.dim * expected_items:
                return [
                    [float(numbers[i * self.dim + j]) for j in range(self.dim)]
                    for i in range(expected_items)
                ]

        except Exception as e:
            print(f"响应解析异常: {str(e)}\n原始响应: {response}")
        raise ValueError(f"无法从响应中提取{expected_items}个{self.dim}维向量")

    def adaptive_operators(self, generation: int, improvement: float):
        """动态调整遗传算子参数"""
        # 基于改进程度调整
        if improvement < 0.01:  # 进步缓慢
            self.mutation_rate = min(0.3, self.base_mutation_rate * 1.5)
            self.crossover_rate = max(0.7, self.base_crossover_rate * 0.9)
        else:
            self.mutation_rate = self.base_mutation_rate
            self.crossover_rate = self.base_crossover_rate

        # 基于代数调整
        if generation > 30:
            self.mutation_rate *= 0.95  # 后期减少变异

    def deepseek_guided_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """增强版智能交叉（含完整任务说明）"""
        if (not self.use_deepseek or
                random.random() > 0.7 or
                self.consecutive_failures > 3):
            return self.traditional_crossover(parent1, parent2)

        try:
            prompt = self.get_optimization_context_prompt() + f"""

            ## 当前交叉请求
            父代1: {np.round(parent1, 4).tolist()}
            父代2: {np.round(parent2, 4).tolist()}

            ## 输出要求
            1. 返回两个{self.dim}维子代向量
            2. 格式：严格的JSON数组 [[子代1],[子代2]]
            3. 数值范围：[-5.0000, 5.0000]
            4. 保留4位小数

            ## 策略建议
            - 结合父代优势特征
            - 保持适度多样性
            - 避免超出变量范围

            示例正确响应：
            [[1.2345,-2.3456,3.4567], [4.5678,-5.0000,0.1234]]
            """

            response = self.call_deepseek_api(prompt)
            children = self.parse_api_response(response, 2)

            child1 = np.clip(np.round(children[0], 4), -5, 5)
            child2 = np.clip(np.round(children[1], 4), -5, 5)

            self.consecutive_failures = 0
            return child1, child2

        except Exception as e:
            print(f"智能交叉失败: {str(e)}")
            self.api_failure_count += 1
            return self.traditional_crossover(parent1, parent2)

    def traditional_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """自适应模拟二进制交叉"""
        eta = 15  # 分布指数
        rand = np.random.rand(self.dim)

        gamma = np.empty_like(rand)
        mask = rand <= 0.5
        gamma[mask] = (2 * rand[mask]) ** (1 / (eta + 1))
        gamma[~mask] = (1 / (2 * (1 - rand[~mask]))) ** (1 / (eta + 1))

        child1 = 0.5 * ((1 + gamma) * parent1 + (1 - gamma) * parent2)
        child2 = 0.5 * ((1 - gamma) * parent1 + (1 + gamma) * parent2)

        return np.clip(child1, -5, 5), np.clip(child2, -5, 5)

    def deepseek_guided_mutation(self, individual: np.ndarray) -> np.ndarray:
        """增强版智能变异（含完整任务说明）"""
        if (not self.use_deepseek or
                random.random() > 0.7 or
                self.consecutive_failures > 3):
            return self.traditional_mutation(individual)

        try:
            prompt = self.get_optimization_context_prompt() + f"""

            ## 当前变异请求
            原始个体: {np.round(individual, 4).tolist()}
            当前适应度: {-self.objective_func(individual):.4f}

            ## 输出要求
            1. 返回一个{self.dim}维变异向量
            2. 格式：严格的JSON数组 [变异向量]
            3. 变异幅度：建议±0.5范围内
            4. 数值范围：[-5.0000, 5.0000]
            5. 保留4位小数

            ## 策略建议
            - 对1-2个维度进行显著变异
            - 保持其他维度微调
            - 有助于跳出局部最优

            示例正确响应：
            [1.2345,-2.3456,0.1234]
            """

            response = self.call_deepseek_api(prompt)
            mutated = self.parse_api_response(response, 1)[0]

            mutated = np.clip(np.round(mutated, 4), -5, 5)
            self.consecutive_failures = 0
            return mutated

        except Exception as e:
            print(f"智能变异失败: {str(e)}")
            self.api_failure_count += 1
            return self.traditional_mutation(individual)

    def traditional_mutation(self, individual: np.ndarray) -> np.ndarray:
        """多项式变异"""
        eta_m = 20  # 变异分布指数
        mutation_mask = np.random.rand(self.dim) < self.mutation_rate

        delta = np.zeros_like(individual)
        rand = np.random.rand(self.dim)

        mask = rand < 0.5
        delta[mask] = (2 * rand[mask]) ** (1 / (eta_m + 1)) - 1
        delta[~mask] = 1 - (2 * (1 - rand[~mask])) ** (1 / (eta_m + 1))

        mutated = individual + delta * np.random.uniform(0.5, 1.5, self.dim)
        return np.clip(mutated, -5, 5)

    def evolve(self, generations: int = 100) -> Tuple[np.ndarray, float]:
        """增强版进化流程"""
        best_individual, best_fitness = None, -float('inf')

        for gen in range(1, generations + 1):
            # 评估种群
            fitness = self.evaluate_population()
            current_best = max(fitness)
            current_avg = sum(fitness) / len(fitness)
            improvement = current_best - best_fitness if best_fitness != -float('inf') else 0

            # 更新最佳个体
            if current_best > best_fitness:
                best_idx = fitness.index(current_best)
                best_individual = self.population[best_idx].copy()
                best_fitness = current_best
                self.last_improvement = gen

            # 记录历史数据
            self.best_history.append(current_best)
            self.diversity_history.append(self.calculate_diversity())

            # 动态调整参数
            self.adaptive_operators(gen, improvement)

            # 控制台输出
            print(f"Gen {gen:03d}: Best={current_best:.4f} Avg={current_avg:.4f} "
                  f"Mut={self.mutation_rate:.2f} Div={self.diversity_history[-1]:.3f} "
                  f"API Fail={self.api_failure_count}/{self.max_api_failures}")

            # 早熟检测与处理
            if gen - self.last_improvement > 15:
                print("检测到早熟收敛，增加多样性...")
                self.population = self.population[:self.population_size // 2] + \
                                  self.initialize_population()[self.population_size // 2:]
                self.mutation_rate = min(0.5, self.mutation_rate * 1.5)
                self.last_improvement = gen

            # 选择父代
            parents = self.select_parents(fitness)
            next_population = []

            # 生成下一代
            for i in range(0, self.population_size, 2):
                parent1, parent2 = parents[i], parents[i + 1]

                # 交叉
                if random.random() < self.crossover_rate:
                    child1, child2 = self.deepseek_guided_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # 变异
                child1 = self.deepseek_guided_mutation(child1)
                child2 = self.deepseek_guided_mutation(child2)

                next_population.extend([child1, child2])

            self.population = next_population[:self.population_size]

            # 完全禁用API如果失败太多
            if self.api_failure_count >= self.max_api_failures and self.use_deepseek:
                self.use_deepseek = False
                print("API失败次数达到上限，完全切换至传统模式")

        return best_individual, best_fitness


# 示例使用
if __name__ == "__main__":
    # Rastrigin函数（最小化问题）
    def rastrigin(x: List[float]) -> float:
        A = 10
        return A * len(x) + sum(xi ** 2 - A * np.cos(2 * np.pi * xi) for xi in x)


    # 转换为最大化问题
    def objective_func(x):
        return -rastrigin(x)


    # 初始化算法
    ga = EnhancedDeepSeekGA(
        objective_func=objective_func,
        dim=5,
        population_size=30,
        mutation_rate=0.15,
        crossover_rate=0.85,
        use_deepseek=True,
        deepseek_temperature=0.3,
        deepseek_api_key="your api"  # 建议使用环境变量
    )

    # 运行进化
    start_time = time.time()
    best_solution, best_fitness = ga.evolve(generations=20)
    elapsed = time.time() - start_time

    print(f"\n优化完成，耗时: {elapsed:.2f}秒")
    print(f"最佳解: {np.round(best_solution, 4)}")
    print(f"最佳适应度: {-best_fitness:.6f} (原始问题的最小值)")
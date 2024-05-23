import numpy as np
import pulp
from deap import base, creator, tools, algorithms
import cvxpy as cp
from scipy.optimize import lsq_linear, linprog

solver_dict = {
    "PULP_CBC_CMD": pulp.PULP_CBC_CMD,
    "CLARABEL": cp.CLARABEL
}

# 1. Линейное программирование
def solve_with_linear_programming_with_constraints(A, y0, x_min=0, method='highs'):
    
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
    
    # Целевая функция (минимизация суммы x)
    c = np.ones(n)
    
    # Матрица ограничений
    A_ub = -A
    b_ub = -y0
    
    # Ограничения на x снизу
    bounds = [(x_min[i], None) for i in range(n)]
    
    # Решение задачи линейного программирования
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
    
    # Округление решения до целых чисел
    x = np.ceil(result.x).astype(int)
    
    # Вычисление результирующего вектора y
    y = A @ x
    
    return x, y

# 2.Mixed-Integer Linear Programming (MILP)
def solve_with_milp(A, y0, x_min=0, solver_name = "PULP_CBC_CMD", msg=False):
    # Число переменных
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
    
    # Создаем модель
    model = pulp.LpProblem("My_LP_Model", pulp.LpMinimize)
    
    # Определяем переменные
    x = [pulp.LpVariable(f"x_{i}", lowBound=x_min[i], cat=pulp.LpInteger) for i in range(n)]
    
    # Целевая функция: минимизация суммы x (можно выбрать другую цель)
    model += pulp.lpSum([x[i] for i in range(n)])
    
    # Ограничения Ax >= y0
    for i in range(len(y0)):
        model += pulp.lpDot([A[i][j] for j in range(n)], [x[j] for j in range(n)]) >= y0[i], f"Constraint_{i}"
    
    # Решаем задачу, отключаем вывод отладочных сообщений
    model.solve(solver = solver_dict[solver_name](msg=msg))
    
    if model.status == pulp.LpStatusOptimal:
        # Если решение найдено, выводим его
        solution = np.array([pulp.value(x[i]) for i in range(n)], dtype=int)
        resulting_y = np.dot(A, solution)
        return solution, resulting_y
    else:
        return None, None
    
# 3. Метод последовательного квадратичного программирования (SQP)
def solve_with_linear_inequality(A, y0, x_min=0, solver_name="CLARABEL"):
    
    # Проверка размеров матрицы и вектора
    if A.shape[0] != y0.shape[0]:
        raise ValueError("Размерности матрицы A и вектора y0 не совпадают.")
    
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
    
    # Переменные оптимизации
    x = cp.Variable(n)
    y = A @ x
    
    # Ограничения
    constraints = [y >= y0, x >= x_min]
    
    # Целевая функция (минимизация нормы)
    objective = cp.Minimize(cp.norm(y - y0))
    
    # Формулировка и решение задачи
    problem = cp.Problem(objective, constraints)
    
    # Решение задачи с использованием решателя по умолчанию (Clarabel)
    problem.solve(solver=solver_dict[solver_name])
    
    # Проверка статуса решения
    if problem.status not in ["infeasible", "unbounded"]:
        # Округление результата до целых чисел
        x_int = np.round(x.value).astype(int)
        y_result = A @ x_int
        return x_int, y_result
    else:
        raise ValueError("Задача не имеет допустимого решения.")

# 4. Метод наименьших квадратов с ограничениями    
def solve_with_least_squares_with_constraints(A, y0, x_min=0, method='trf'):
    
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
    
    # Ограничения снизу
    bounds = (x_min, np.inf)
    
    # Решение задачи наименьших квадратов с ограничениями
    result = lsq_linear(A, y0, bounds=bounds, method=method)
    
    x = np.ceil(result.x).astype(int)
    y = A @ x
    
    return x, y

# 5. Метод координатного спуска
def solve_with_coordinate_descent(A, y0, x_min=0, max_iter=1000):
    
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
    
    x = x_min
    y = A @ x
    
    for _ in range(max_iter):
        for j in range(n):
            # Рассчитаем вклад j-ой координаты
            aj = A[:, j]
            
            # Удалим вклад текущей координаты из y
            y -= aj * x[j]
            
            # Определим новую координату, минимизируя отклонение от y0 и учитывая x_min
            valid_indices = aj != 0
        
            delta = np.zeros_like(y)
            delta[valid_indices] = (y0[valid_indices] - y[valid_indices]) * aj[valid_indices]
            delta = np.maximum(delta, 0)
            x[j] = int(np.ceil(np.max(delta)))
            x[j] = max(x[j], x_min[j]) 
            
            # Добавим обновленный вклад текущей координаты
            y += aj * x[j]
        
        if np.all(y >= y0):
            break
    
    return x, y

# 6. Метод градиентного спуска с проекцией
def solve_with_projected_gradient_descent(A, y0, x_min=0, max_iter=1000, initial_step_size=0.01):
    
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
    
    # Инициализация x
    x = np.maximum(np.max(A * y0[:, np.newaxis], axis=0).astype(int), x_min)
    
    # Функция проекции на множество допустимых значений
    def project(x):
        return np.maximum(x, x_min)
    
    # Адаптивное изменение размера шага
    step_size = initial_step_size
    
    for iteration in range(max_iter):
        # Вычисление градиента
        gradient = A.T @ np.maximum(0, y0 - A @ x)
        
        # Обновление размера шага (затухание)
        step_size = initial_step_size / (1 + iteration / 100)
        
        # Обновление x с учетом адаптивного размера шага
        x = x + step_size * gradient
        
        # Проекция на множество допустимых значений
        x = project(x)
        
        # Проверка условия выхода
        if np.all(A @ x >= y0):
            break
    
    # Округление решения до целых чисел
    x = np.ceil(x).astype(int)
    
    # Вычисление результирующего вектора y
    y = A @ x
    
    return x, y

# 7. Генетический алгоритм
def solve_with_genetic_algorithm(A, y0, x_min=0, population_size=100, generations=50, cxpb=0.5, mutpb=0.2):
    
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
    
    # Определение целевой функции
    def evaluate(individual):
        x = np.array(individual)
        y = np.dot(A, x)
        penalty = np.sum(np.maximum(0, y0 - y))  # Штраф за нарушение условия y >= y0
        mse = np.mean((y - y0) ** 2) + 1000 * penalty  # Среднеквадратичная ошибка с штрафом
        return mse,

    # Проверка и корректировка индивидов на границы
    def check_bounds(individual, x_min):
        for i in range(len(individual)):
            if individual[i] < x_min[i]:
                individual[i] = x_min[i]
        return individual

    # Удаление ранее созданных классов, если они существуют
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    # Генетический алгоритм с использованием DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    
    # Инициализация x случайным образом в рамках ограничений
    def init_individual(icls):
        individual = [np.random.randint(low=x_min[i], high=x_min[i] + 10) for i in range(A.shape[1])]
        return icls(individual)

    toolbox.register("individual", init_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # Основной цикл генетического алгоритма
    def apply_bounds(population, x_min):
        for individual in population:
            check_bounds(individual, x_min)
        return population

    population = toolbox.population(n=population_size)
    population = apply_bounds(population, x_min)
    
    # Основной цикл генетического алгоритма
    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
        offspring = apply_bounds(offspring, x_min)
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    best_ind = tools.selBest(population, 1)[0]
    x = np.array(best_ind)
    y = np.dot(A, x)
    
    return x, y

# 8. Метод имитации отжига
def solve_with_simulated_annealing(A, y0, x_min=0, initial_temp = 1000, cooling_rate = 0.99, max_iter = 10000):
    
    def objective_function(A, x, y0):
        y = np.dot(A, x)
        penalty = np.sum(np.maximum(y0 - y, 0))  # штраф за нарушение условия y >= y0
        return np.sum((y - y0)**2) + penalty
    
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
        
    x = np.random.randint(x_min, x_min + 10, size=n)  # начальное решение
    T = initial_temp
    
    for _ in range(max_iter):
        x_new = x + np.random.randint(-1, 2, size=n)  # новое решение с небольшим случайным изменением
        x_new = np.maximum(x_new, x_min)  # применение ограничения x >= xmin
        
        cost_old = objective_function(A, x, y0)
        cost_new = objective_function(A, x_new, y0)
        
        if cost_new < cost_old:
            x = x_new
        else:
            acceptance_prob = np.exp(-(cost_new - cost_old) / T)
            if acceptance_prob > np.random.rand():
                x = x_new
        
        T *= cooling_rate  # уменьшение температуры
    
    y = np.dot(A, x)
    return x, y

# 9. Метод ветвей и границ
def solve_with_branch_and_bound(A, y0, x_min=0, solver_name="PULP_CBC_CMD", msg=0):
    m, n = A.shape
    
    # Если x_min является числом, создаем массив нижних границ
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
    
    # Создаем модель
    model = pulp.LpProblem("Branch_and_Bound_Problem", pulp.LpMinimize)
    
    # Определяем переменные
    x = {i: pulp.LpVariable(f"x_{i}", lowBound=x_min[i], cat=pulp.LpInteger) for i in range(n)}
    
    # Определяем целевую функцию - минимизация суммы x
    model += pulp.lpSum(x[i] for i in range(n))
    
    # Добавляем ограничения Ax >= y0
    for i in range(m):
        model += pulp.lpSum(A[i, j] * x[j] for j in range(n)) >= y0[i], f"Constraint_{i}"
    
    # Решаем задачу, отключаем вывод отладочных сообщений
    model.solve(solver = solver_dict[solver_name](msg=msg))
    
    if model.status == pulp.LpStatusOptimal:
        result_x = np.array([x[j].value() for j in range(n)], dtype=int)
        result_y = np.dot(A, result_x)
        return result_x, result_y
    else:
        return None, None

functions = [
    solve_with_linear_programming_with_constraints,
    solve_with_milp,
    solve_with_linear_inequality,
    solve_with_least_squares_with_constraints,
    solve_with_coordinate_descent,
    solve_with_projected_gradient_descent,
    solve_with_genetic_algorithm,
    solve_with_simulated_annealing,
    solve_with_branch_and_bound
]

functions_names = [
    "Linear Programming with Constraints",
    "MILP",
    "Linear Inequality",
    "Least Squares with Constraints",
    "Coordinate Descent",
    "Projected Gradient Descent",
    "Genetic Algorithm",
    "Simulated Annealing",
    "Branch and Bound"
]
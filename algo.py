import numpy as np
import pulp
from deap import base, creator, tools, algorithms
from scipy.optimize import lsq_linear, linprog, minimize

solver_dict = {
    "PULP_CBC_CMD": pulp.PULP_CBC_CMD
}

def validate_input_data(A, y0, x_min):
    if not np.all(np.isin(A, [0, 1])):
        return False
    
    if np.any(y0 < 0):
        return False
    
    if np.any(x_min < 0):
        return False
    
    if len(y0.shape) != 1:
        return False
    
    if A.shape[0] != y0.shape[0]:
        return False
    
    return True

# 1. Линейное программирование
def solve_with_linear_programming_with_constraints(A, y0, x_min=0, method='highs'):
    
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
    
    if not validate_input_data(A, y0, x_min):
        raise ValueError("Invalid input")
    
    c = np.ones(n)
    A_ub = -A
    b_ub = -y0
    bounds = [(x_min[i], None) for i in range(n)]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
    
    if result.success:
        x = np.ceil(result.x).astype(int)
        y = A @ x
    else:
        x = np.zeros(n, dtype=int)
        y = A @ x
    
    return x, y

# 2. Метод последовательного квадратичного программирования (SQP)
def solve_with_sqp(A, y0, x_min=0, method='SLSQP', max_iter=100):
    n = A.shape[1]

    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)

    if not validate_input_data(A, y0, x_min):
        raise ValueError("Invalid input")
    
    def objective(x):
        y = A @ x
        return (1 / len(y0)) * np.sum((y - y0) ** 2)
    
    def constraint_y(x):
        return A @ x - y0  # y >= y0 => A @ x - y0 >= 0
    def constraint_x_min(x):
        return x - x_min  # x >= x_min => x - x_min >= 0
    
    x0 = np.maximum(x_min, np.zeros(n))
    
    constraints = [{'type': 'ineq', 'fun': constraint_y},
                   {'type': 'ineq', 'fun': constraint_x_min}]
    
    result = minimize(objective, x0, method=method, constraints=constraints, options={'maxiter': max_iter})
    
    if result.success:
        x_opt = np.round(result.x).astype(int)
        y_result = A @ x_opt
        return x_opt, y_result
    else:
        result_x = np.zeros(n, dtype=int)
        result_y = A @ result_x
        return result_x, result_y

# 3. Решение методом наименьших квадратов с ограничениями
def solve_with_least_squares_with_constraints(A, y0, x_min=0, method='trf'):
    
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
        
    if not validate_input_data(A, y0, x_min):
        raise ValueError("Invalid input")
    
    bounds = (x_min, np.inf)
    result = lsq_linear(A, y0, bounds=bounds, method=method)
    
    if result.success:
        x = np.ceil(result.x).astype(int)
        y = A @ x
    else:
        x = np.zeros(n, dtype=int)
        y = A @ x
    
    return x, y

# 4. Метод координатного спуска
def solve_with_coordinate_descent(A, y0, x_min=0, max_iter=1000):
    
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
        
    if not validate_input_data(A, y0, x_min):
        raise ValueError("Invalid input")
    
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

# 5. Метод градиентного спуска с проекцией
def solve_with_projected_gradient_descent(A, y0, x_min=0, max_iter=1000, initial_step_size=0.01):
    
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
        
    if not validate_input_data(A, y0, x_min):
        raise ValueError("Invalid input")
    
    # Инициализация x
    x = np.maximum(np.max(A * y0[:, np.newaxis], axis=0).astype(int), x_min)
    
    # Функция проекции на множество допустимых значений
    def project(x):
        return np.maximum(x, x_min)
    
    # Адаптивное изменение размера шага
    step_size = initial_step_size
    
    for iteration in range(max_iter):

        gradient = A.T @ np.maximum(0, y0 - A @ x)
        step_size = initial_step_size / (1 + iteration / 100)
        
        x = x + step_size * gradient
        x = project(x)
        
        if np.all(A @ x >= y0):
            break
    
    x = np.ceil(x).astype(int)
    y = A @ x
    
    return x, y

# 6. Генетический алгоритм
def solve_with_genetic_algorithm(A, y0, x_min=0, population_size=100, generations=50, cxpb=0.5, mutpb=0.2):
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
    
    if not validate_input_data(A, y0, x_min):
        raise ValueError("Invalid input")
    
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
        individual = []
        for i in range(A.shape[1]):
            low = x_min[i]
            high = low + 10
            if high <= low:
                high = low + 1
            try:
                val = np.random.randint(low=low, high=high)
            except ValueError as e:
                print(f"ValueError: {e}, low={low}, high={high}")
                raise
            individual.append(val)
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
        offspring = [toolbox.clone(ind) for ind in population]

        # Применение кроссовера и мутации к потомкам
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if len(child1) > 1 and len(child2) > 1:
                if np.random.rand() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        offspring = apply_bounds(offspring, x_min)
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

    best_ind = tools.selBest(population, 1)[0]
    x = np.array(best_ind)
    y = np.dot(A, x)
    
    return x, y

# 7. Метод имитации отжига
def solve_with_simulated_annealing(A, y0, x_min=0, initial_temp = 1000, cooling_rate = 0.99, max_iter = 10000):
    
    n = A.shape[1]
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
        
    if not validate_input_data(A, y0, x_min):
        raise ValueError("Invalid input")
    
    def objective_function(A, x, y0):
        y = np.dot(A, x)
        penalty = np.sum(np.maximum(y0 - y, 0))  # штраф за нарушение условия y >= y0
        return np.sum((y - y0)**2) + penalty
        
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

# 8. Метод ветвей и границ
def solve_with_branch_and_bound(A, y0, x_min=0, solver_name="PULP_CBC_CMD", msg=0):
    m, n = A.shape
    
    if isinstance(x_min, (int, float)):
        x_min = np.full(n, np.ceil(x_min), dtype=int)
        
    if not validate_input_data(A, y0, x_min):
        raise ValueError("Invalid input")
    
    model = pulp.LpProblem("Branch_and_Bound_Problem", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}", lowBound=x_min[i], cat=pulp.LpInteger) for i in range(n)}
    model += pulp.lpSum(x[i] for i in range(n))
    
    for i in range(m):
        model += pulp.lpSum(A[i, j] * x[j] for j in range(n)) >= y0[i], f"Constraint_{i}"
    
    model.solve(solver=solver_dict[solver_name](msg=msg))
    
    if model.status == pulp.LpStatusOptimal:
        result_x = np.array([x[j].value() for j in range(n)], dtype=int)
        result_y = np.dot(A, result_x)
    else:
        result_x = np.zeros(n, dtype=int)
        result_y = A @ result_x
    
    return result_x, result_y

functions = [
    solve_with_linear_programming_with_constraints,
    solve_with_sqp,
    solve_with_least_squares_with_constraints,
    solve_with_coordinate_descent,
    solve_with_projected_gradient_descent,
    solve_with_genetic_algorithm,
    solve_with_simulated_annealing,
    solve_with_branch_and_bound
]

functions_names = [
    "Linear Programming with Constraints",
    "SQP",
    "Least Squares with Constraints",
    "Coordinate Descent",
    "Projected Gradient Descent",
    "Genetic Algorithm",
    "Simulated Annealing",
    "Branch and Bound"
]

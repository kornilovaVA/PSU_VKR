import numpy as np
import pandas as pd
import time
from pathlib import Path
import json
from algo import *

script_dir = Path(__file__).parent

with open(script_dir/'test_data.json', 'r') as f:
    data = json.load(f)
    
def process_xmin(xmin, num_columns):
    if isinstance(xmin, (int, float)):
        return np.full(num_columns, np.ceil(xmin), dtype=int)
    elif isinstance(xmin, list):
        return np.array(xmin, dtype=int)
    else:
        raise ValueError("xmin should be either an int, float, or list.")
    
test_cases = []
for case in data.values():
    A = np.array(case['matrix'])
    y0 = np.array(case['vector'])
    num_columns = A.shape[1]
    xmin = process_xmin(case['xmin'], num_columns)
    test_cases.append((case['name'], A, y0, xmin))

test_cases_names = [
    case['name'] for case in data.values()
    ]

with open(script_dir/'config.json', 'r') as file:
    config = json.load(file)

def run_comprehensive_tests(functions, test_cases):
    def calc_mse(y, y0):
        return np.mean((y - y0) ** 2)

    def calc_sum(x):
        
        return np.sum(np.abs(x))

    def calc_uniform_distr(x):
        x_mean = np.mean(x)
        return np.sum((x - x_mean) ** 2)

    results = []

    for func in functions:
        func_name = func.__name__  
        print(func_name)
        func_results = []
        for name, A, y0, xmin in test_cases:
            
            print(name)
            
            kwargs = config.get(func_name, {})
            
            start_time = time.time()
            try:
                x, y = func(A, y0, xmin, **kwargs)
            except Exception as e:
                print(f"Ошибка при выполнении функции {func_name}: {e}")
                continue 

            end_time = time.time()
            mse = calc_mse(y, y0)
            sum_abs_x = calc_sum(x)
            sum_sq_dev = calc_uniform_distr(x)
            duration = end_time - start_time
            func_results.append((y.tolist(), mse, duration, sum_abs_x, sum_sq_dev))
        results.append(func_results)

    return results

def save_results_to_excel(results, functions_names, test_cases_names, filename="results.xlsx"):
    values_data = []
    mse_data = []
    time_data = []
    sum_abs_x_data = []
    sum_sq_dev_data = []

    for func_name, func_results in zip(functions_names, results):
        values_row = [func_name]
        mse_row = [func_name]
        time_row = [func_name]
        sum_abs_x_row = [func_name]
        sum_sq_dev_row = [func_name]
        
        for (y, mse, duration, sum_abs_x, sum_sq_dev) in func_results:
            values_row.append(f"{y}" if y is not None else "None")
            mse_row.append(f"{mse:.6f}" if mse != float('inf') else "inf")
            time_row.append(f"{duration:.6f}") 
            sum_abs_x_row.append(f"{sum_abs_x:.6f}")
            sum_sq_dev_row.append(f"{sum_sq_dev:.6f}")
        
        values_data.append(values_row)
        mse_data.append(mse_row)
        time_data.append(time_row)
        sum_abs_x_data.append(sum_abs_x_row)
        sum_sq_dev_data.append(sum_sq_dev_row)

    values_df = pd.DataFrame(values_data, columns=["Function/Test Case"] + test_cases_names)
    mse_df = pd.DataFrame(mse_data, columns=["Function/Test Case"] + test_cases_names)
    time_df = pd.DataFrame(time_data, columns=["Function/Test Case"] + test_cases_names)
    sum_abs_x_df = pd.DataFrame(sum_abs_x_data, columns=["Function/Test Case"] + test_cases_names)
    sum_sq_dev_df = pd.DataFrame(sum_sq_dev_data, columns=["Function/Test Case"] + test_cases_names)

    with pd.ExcelWriter(filename) as writer:
        values_df.to_excel(writer, sheet_name='Values of y', index=False)
        time_df.to_excel(writer, sheet_name='Execution Time (s)', index=False) 
        mse_df.to_excel(writer, sheet_name='MSE', index=False)
        sum_abs_x_df.to_excel(writer, sheet_name='Sum x', index=False)
        sum_sq_dev_df.to_excel(writer, sheet_name='Uniform distr x', index=False)

    print(f"Results saved to {filename}")

results = run_comprehensive_tests(functions, test_cases)

excel_file_path = script_dir / "data/results.xlsx"
save_results_to_excel(results, functions_names, test_cases_names, filename=excel_file_path)
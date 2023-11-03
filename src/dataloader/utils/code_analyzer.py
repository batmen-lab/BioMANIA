from typing import Tuple, Set
import ast
import types

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import math
import sklearn.preprocessing
import sklearn
import scipy

def is_variable_in_parentheses(var: str, code: str) -> bool:
    """
    Check if a variable is within parentheses in the given code.
    """
    tmp = code.split('var')
    if len(tmp)<=1:
        return False
    elif len(tmp)>=2:
        for i in range(len(tmp)-1):
            if '(' in tmp[i] and ')' in tmp[i+1]:
                return True
            else:
                continue
    return False
def extract_io_variables(code: str, namespace: dict) -> Tuple[Set[str], Set[str]]:
    """
    Extract input and output variables with refined logic.
    - code: The input code segment.
    - namespace: Dictionary containing the initial namespace for the code execution.
    Returns a tuple (input variables, output variables).
    """
    initial_vars = set(namespace.keys())
    
    # Attempt to execute the code
    try:
        exec(code, namespace)
    except Exception as e:
        print('?', e)  # If there's an error, we simply pass. 
    
    final_vars = set(namespace.keys())
    
    # Variables that appeared on the left or right of an assignment
    left_vars, right_vars = set(), set()
    lines = code.split('\n')
    for line in lines:
        if '=' in line:
            left, right = line.split('=', 1)
            for var in initial_vars.union(final_vars):
                if var in left:
                    left_vars.add(var)
                if var in right:
                    right_vars.add(var)
        else:
            right = line
            for var in initial_vars.union(final_vars):
                if (var in right):
                    right_vars.add(var)
                if (var in right) and (not is_variable_in_parentheses(var, right)):
                    left_vars.add(var)
    
    # Input variables: Variables that are in the initial namespace and appear on the right side of an assignment
    input_vars = (initial_vars & right_vars)
    if  "__builtins__" in input_vars:
        input_vars = input_vars  - {"__builtins__"}
    # Filter out module types and built-in functions
    input_vars = {var for var in input_vars if not isinstance(namespace.get(var, None), (types.ModuleType, types.BuiltinFunctionType))}
    input_vars = {var for var in input_vars if not var.startswith('_')}
    
    # Output variables: Variables that appear on the left side of an assignment or are new after the execution
    output_vars = (left_vars | (final_vars - initial_vars))
    if  "__builtins__" in output_vars:
        output_vars = output_vars  - {"__builtins__"}
    # Filter out loop variables by checking if they appear after 'for' and before 'in'
    loop_vars = {var for var in output_vars if f"for {var} in" in code}
    output_vars -= loop_vars
    # Filter out module types and built-in functions
    output_vars = {var for var in output_vars if not isinstance(namespace.get(var, None), (types.ModuleType, types.BuiltinFunctionType))}
    output_vars = {var for var in output_vars if not var.startswith('_')}
    
    return (input_vars, output_vars)
if __name__=='__main__':
    # Testing the refined execution-based method on the modified extended attack test cases
    refined_execution_extended_attack_results = []
    modified_extended_attack_test_cases = [
        ("df = pd.DataFrame(data)\nresult = df[df['A'] > 5].groupby('B').sum()", {"pd": pd, "data": {"A": [1, 2, 3, 10], "B": [5, 6, 7, 8]}}),
        ("result = df.groupby('A').filter(lambda x: x['B'].mean() > 5).pivot_table(index='C', columns='D', values='E', aggfunc=np.sum)", {"df": pd.DataFrame({"A": [1, 2, 3, 4], "B": [10, 20, 30, 40], "C": [1, 1, 2, 2], "D": ["x", "y", "x", "y"], "E": [1, 2, 3, 4]}), "np": np}),
        ("grouped = df.groupby('A')\nfiltered = grouped.filter(lambda x: x['B'].mean() > 5)\nresult = filtered.pivot_table(index='C', columns='D', values='E', aggfunc=np.sum)", {"df": pd.DataFrame({"A": [1, 2, 3, 4], "B": [10, 20, 30, 40], "C": [1, 1, 2, 2], "D": ["x", "y", "x", "y"], "E": [1, 2, 3, 4]}), "np": np}),
        ("arr = np.array(data); mean_val = np.mean(arr)", {"np": np, "data": [1, 2, 3]}),
        ("result = math.sqrt(value)", {"math": math, "value": 25}),
        ("plt.plot(x, y)\nplt.show()", {"plt": plt, "x": [1, 2, 3], "y": [4, 5, 6]}),
        ("sns.heatmap(data)", {"sns": sns, "data": [[1, 2], [3, 4]], "plt": plt}),
        ("transformed = sklearn.preprocessing.scale(data)", {"sklearn": sklearn, "sklearn.preprocessing": sklearn.preprocessing, "data": [1, 2, 3, 4, 5]}),
        ("for i in range(len(data)):\tprocessed_data.append(data[i] + 5)", {"data": [1, 2, 3], "processed_data": []}),
        ("import scipy.stats\nresult = scipy.stats.zscore(data)", {"data": [1, 2, 3, 4, 5]}),
        ("fig, ax = plt.subplots()\nax.plot(x, y)\nax.set_title('Sample Plot')", {"plt": plt, "x": [1, 2, 3], "y": [4, 5, 6]})
    ]
    for code, ns in modified_extended_attack_test_cases:
        try:
            inputs, outputs = extract_io_variables(code, ns)
            refined_execution_extended_attack_results.append((code.strip(), (inputs, outputs)))
            print(code, (inputs, outputs))
        except Exception as e:
            refined_execution_extended_attack_results.append((code.strip(), str(e)))
        #print(refined_execution_extended_attack_results)
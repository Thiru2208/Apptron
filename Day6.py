import numpy as np

#Employee Dataset
# Columns: Employee ID, Age, Department, Salary, Experience, Performance Score

employee_data = np.array([
    [1, 28, 'Sales', 50000, 3, 85],
    [2, 35, 'Marketing', 60000, 7, 90],
    [3, 30, 'IT', -70000, 5, 88],
    [4, 40, 'HR', 55000, 10, 80],
    [5, 25, 'Finance', 65000, 2, 92],
    [6, 32, 'Sales', 52000, 4, 87],
    [7, 29, 'Marketing', 58000, 3, 89],
    [8, 38, 'IT', -72000, 8, 91],
    [9, 27, 'HR', 54000, 2, 84],
    [10, 31, 'Finance', 68000, 6, 93]
])

print("ORIGINAL EMPLOYEE DATASET")
print("-------------------------")
print("ID Age Dept Salary Experience Performance")
print(employee_data)


# 1. Increase salary by 10%
employee_data[:, 3] = (employee_data[:, 3].astype(float) * 1.10
                       ).astype(int)
print("\nEMPLOYEE DATASET AFTER 10% SALARY INCREASE")
print("--------------------------------------------")
print("ID Age Dept Salary Experience Performance")
print(employee_data)

# 2. Replace negative salaries 

salary_column = employee_data[:, 3].astype(float)

#find positive salary average

positive_salaries = salary_column[salary_column > 0]
average_positive_salary = np.mean(positive_salaries)

#replace negative salaries with average positive salary
salary_column[salary_column < 0] = average_positive_salary
employee_data[:, 3] = salary_column.astype(int)

print("\nEMPLOYEE DATASET AFTER REPLACING NEGATIVE SALARIES")
print("---------------------------------------------------")
print("ID Age Dept Salary Experience Performance")
print(employee_data)

# 3. Filter employess with performance > 90
performance = employee_data[:, 5].astype(int)
high_performance_employees = employee_data[performance > 90]

print("\nEMPLOYEES WITH PERFORMANCE SCORE GREATER THAN 90")
print("------------------------------------------------")
print("ID Age Dept Salary Experience Performance")
print(high_performance_employees)

# 4. find employess above average salary

salary_column = employee_data[:, 3].astype(float)

average_salary = np.mean(salary_column)
above_average = employee_data[salary_column > average_salary]

print("\nAVERAGE SALARY:", average_salary)

print("\nEMPLOYEES WITH ABOVE AVERAGE SALARY")
print("------------------------------------")
print("ID Age Dept Salary Experience Performance")
print(above_average)

# 5. Reshape dataset
reshaped_data = employee_data.reshape(5, 12)

print("\nRESHAPED EMPLOYEE DATASET")
print("-------------------------")
print(reshaped_data)

# 6. Transpose dataset
transposed_data = employee_data.T

print("\nTRANSPOSED EMPLOYEE DATASET")
print("---------------------------")
print(transposed_data)


# cleaned dataset
cleaned_data = employee_data.copy()
print("\nCLEANED EMPLOYEE DATASET")
print("-------------------------")
print(cleaned_data)

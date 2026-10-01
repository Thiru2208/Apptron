import numpy as np

# Columns : ID, age, blood pressure, sugar level, heart rate, BMI

patients = np.array([
    [1, 25, 120, 95, 72, 22.5],
    [2, 67, 150, 160, 80, 27.8],
    [3, 45, 135, 110, 76, 25.2],
    [4, 72, 165, 180, 85, 29.4],
    [5, 34, 118, 90, 70, 23.1],
    [6, 61, 145, 155, 78, 26.7],
    [7, 29, 125, 100, 74, 24.0],
    [8, 55, 138, 120, 77, 25.9],
    [9, 64, 155, 170, 82, 28.3],
    [10, 41, 130, 105, 75, 24.8],
    [11, 70, 160, 175, 84, 30.1],
    [12, 38, 122, 98, 71, 23.7],
    [13, 62, 148, 150, 79, 27.2],
    [14, 50, 140, 130, 78, 26.0],
    [15, 27, 115, 88, 69, 21.9],
    [16, 66, 152, 165, 81, 28.0],
    [17, 48, 132, 115, 76, 25.0],
    [18, 73, 170, 190, 86, 31.2],
    [19, 36, 124, 92, 72, 23.4],
    [20, 59, 142, 140, 79, 26.5]
])


print("HOSPITAL PATIENT DATA")
print(patients)


# 1. Display first patient
print("\nFirst Patient:")
print(patients[0])


# Display last patient
print("\nLast Patient:")
print(patients[-1])


# 2. Display all Blood Pressure values
print("\nBlood Pressure Values:")
print(patients[:, 2])


# 3. Display only Age and BMI columns
print("\nAge and BMI:")
print(patients[:, [1, 5]])


# 4. Retrieve patients 5 to 15
print("\nPatients 5 to 15:")
print(patients[4:15])


# 5. Identify patients with high blood pressure
# High BP = greater than 140
high_bp = patients[patients[:, 2] > 140]

print("\nPatients with High Blood Pressure:")
print(high_bp)


# 6. Extract senior citizens
# Age > 60
senior = patients[patients[:, 1] > 60]

print("\nSenior Citizens:")
print(senior)


# 7. Bonus
# High Blood Pressure AND High Sugar Level
# High BP > 140
# High Sugar > 140

high_bp_sugar = patients[
    (patients[:, 2] > 140) &
    (patients[:, 3] > 140)
]

print("\nPatients with High BP and High Sugar:")
print(high_bp_sugar)
import numpy as np
import sys

# Dataset columns: Student ID, Age, Math, Science, English, AI

students = np.array([
    [1, 18, 75, 80, 72, 85],
    [2, 19, 68, 74, 70, 78],
    [3, 18, 90, 88, 92, 95],
    [4, 20, 55, 60, 58, 65],
    [5, 19, 82, 79, 85, 88],
    [6, 18, 73, 76, 69, 80],
    [7, 20, 64, 70, 66, 72],
    [8, 19, 88, 91, 86, 93],
    [9, 18, 59, 65, 62, 68],
    [10, 20, 77, 83, 79, 84],
    [11, 19, 69, 72, 75, 76],
    [12, 18, 91, 89, 94, 96],
    [13, 20, 62, 67, 64, 70],
    [14, 19, 84, 81, 87, 90],
    [15, 18, 71, 75, 73, 79],
    [16, 20, 66, 69, 68, 74],
    [17, 19, 87, 90, 89, 92],
    [18, 18, 58, 63, 60, 67],
    [19, 20, 79, 85, 82, 86],
    [20, 19, 93, 92, 95, 98]
])

print("STUDENT DATASET")
print("ID  Age Math Science English AI")
print(students)

# Array properties
print("\nARRAY PROPERTIES")
print("Shape:", students.shape)
print("Dimensions:", students.ndim)
print("Size:", students.size)
print("Data Type:", students.dtype)

# Separate marks columns
marks = students[:, 2:]

print("\nMARKS DATASET")
print(marks)

# Calculate average marks for each student
student_averages = np.mean(marks, axis=1)

print("\nAVERAGE MARK OF EACH STUDENT")
for i in range(len(students)):
    print(
        "Student ID",
        students[i, 0],
        "- Average:",
        round(student_averages[i], 2)
    )

# Subject averages
subject_averages = np.mean(marks, axis=0)

print("\nSUBJECT AVERAGES")
print("Math Average:", round(subject_averages[0], 2))
print("Science Average:", round(subject_averages[1], 2))
print("English Average:", round(subject_averages[2], 2))
print("AI Average:", round(subject_averages[3], 2))

# Highest mark in each subject
highest_marks = np.max(marks, axis=0)

print("\nHIGHEST MARKS")
print("Highest Math Mark:", highest_marks[0])
print("Highest Science Mark:", highest_marks[1])
print("Highest English Mark:", highest_marks[2])
print("Highest AI Mark:", highest_marks[3])

# Find the student with the highest overall average
best_student_index = np.argmax(student_averages)

print("\nBEST STUDENT")
print("Student ID:", students[best_student_index, 0])
print("Average:", round(student_averages[best_student_index], 2))

# Python list vs NumPy array
python_list = students.tolist()

python_memory = sys.getsizeof(python_list)

for row in python_list:
    python_memory += sys.getsizeof(row)

    for value in row:
        python_memory += sys.getsizeof(value)

numpy_memory = students.nbytes

print("\n========== MEMORY COMPARISON ==========")
print("Python List Memory :", python_memory, "bytes")
print("NumPy Array Memory :", numpy_memory, "bytes")

if numpy_memory < python_memory:
    print("NumPy uses less memory.")
else:
    print("Python list uses less memory.")

# Bonus - 3D Array
classrooms = students.reshape(2, 10, 6)

print("\n========== 3D ARRAY ==========")
print(classrooms)

print("\n========== 3D ARRAY PROPERTIES ==========")
print("Shape      :", classrooms.shape)
print("Dimensions :", classrooms.ndim)
print("Size       :", classrooms.size)
print("Data Type  :", classrooms.dtype)

import pandas as pd

# Create Student IDs
student_ids = ["S001", "S002", "S003", "S004", "S005",
               "S006", "S007", "S008", "S009", "S010"]

# Create panadas series
student_names = pd.Series(["Alice", "Bob", "Charlie", "David", "Eva",
                           "Frank", "Grace", "Hannah", "Ian", "Julia"],
                            index=student_ids)

ages = pd.Series([20, 21, 19, 23, 24, 25, 19, 27, 28, 29],
                    index=student_ids)

python_marks = pd.Series([85, 72, 91, 68, 77, 88, 60, 95, 74, 81],
                             index=student_ids)

ai_marks = pd.Series([80, 75, 94, 70, 82, 90, 65, 92, 78, 85],
                        index=student_ids)

# Display all student information
print("STUDENT INFORMATION")
print("-------------------")

for student_id in student_ids:
    print(f"Student ID: {student_id}")
    print(f"Name: {student_names[student_id]}")
    print(f"Age: {ages[student_id]}")
    print(f"Python Marks: {python_marks[student_id]}")
    print(f"AI Marks: {ai_marks[student_id]}")
    print("-------------------")

# Highest Python mark

highest_python_id = python_marks.idxmax()

print("\nSTUDENT WITH HIGHEST PYTHON MARK")
print("--------------------------------")
print("Student ID:", highest_python_id)
print("Name:", student_names[highest_python_id])
print("Python Mark:", python_marks[highest_python_id])


# Lowest Python mark

lowest_python_id = python_marks.idxmin()

print("\nSTUDENT WITH LOWEST PYTHON MARK")
print("--------------------------------")
print("Student ID:", lowest_python_id)
print("Name:", student_names[lowest_python_id])
print("Python Mark:", python_marks[lowest_python_id])


# Average Python marks

python_average = python_marks.mean()

print("\nAVERAGE PYTHON MARK")
print("--------------------------------")
print("Average:", python_average)


# Average AI marks

ai_average = ai_marks.mean()

print("\nAVERAGE AI MARK")
print("--------------------------------")
print("Average:", ai_average)


# Students scoring above 75 in Python

above_75 = python_marks[python_marks > 75]

print("\nSTUDENTS SCORING ABOVE 75 IN PYTHON")
print("--------------------------------")

for student_id in above_75.index:
    print(
        student_id,
        student_names[student_id],
        above_75[student_id]
    )


# Number of passing students
# Passing mark = 50

passing_students = python_marks[python_marks >= 50]

print("\nNUMBER OF PASSING STUDENTS")
print("--------------------------------")
print("Total Passing Students:", passing_students.count())


# Modify a value

python_marks["S004"] = 75

print("\nUPDATED PYTHON MARK")
print("--------------------------------")
print("S004 New Mark:", python_marks["S004"])
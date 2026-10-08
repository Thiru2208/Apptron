import matplotlib.pyplot as plt
import numpy as np 
# STUDENT PERFORMANCE VISUALIZER
 
# 1. Create Student Dataset

students = [
    "Student 1", "Student 2", "Student 3", "Student 4", "Student 5",
    "Student 6", "Student 7", "Student 8", "Student 9", "Student 10",
    "Student 11", "Student 12", "Student 13", "Student 14", "Student 15",
    "Student 16", "Student 17", "Student 18", "Student 19", "Student 20"
]

math_marks = np.array([
    78, 65, 90, 55, 82,
    73, 68, 88, 59, 77,
    69, 91, 62, 84, 75,
    58, 80, 92, 71, 86
])

science_marks = np.array([
    80, 70, 88, 60, 79,
    76, 72, 91, 65, 83,
    74, 89, 67, 85, 78,
    63, 82, 94, 73, 87
])

python_marks = np.array([
    85, 68, 92, 65, 88,
    80, 74, 93, 70, 84,
    76, 95, 69, 89, 81,
    66, 86, 96, 77, 90
])

ai_marks = np.array([
    88, 72, 95, 67, 90,
    82, 76, 94, 68, 86,
    78, 96, 70, 91, 83,
    64, 87, 98, 79, 92
])
 
# 2. Calculate Total Marks
 
total_marks = math_marks + science_marks + python_marks + ai_marks

print("\nSTUDENT TOTAL MARKS")
print("-" * 40)

for i in range(len(students)):
    print(students[i], ":", total_marks[i])
 
# 3. Calculate Average Marks
 
average_marks = total_marks / 4

print("\nSTUDENT AVERAGE MARKS")
print("-" * 40)

for i in range(len(students)):
    print(students[i], ":", round(average_marks[i], 2))
 
# 4. Assign Grades
grades = []

for average in average_marks:

    if average >= 85:
        grades.append("A")

    elif average >= 75:
        grades.append("B")

    elif average >= 65:
        grades.append("C")

    elif average >= 50:
        grades.append("D")

    else:
        grades.append("F")


print("\nSTUDENT GRADES")
print("-" * 40)

for i in range(len(students)):
    print(
        students[i],
        "- Average:",
        round(average_marks[i], 2),
        "- Grade:",
        grades[i]
    )
 
# 5. LINE CHART
# Student vs Total Marks
 
plt.figure(figsize=(12, 6))

plt.plot(
    students,
    total_marks,
    marker="o",
    label="Total Marks"
)

plt.title("Student vs Total Marks")
plt.xlabel("Students")
plt.ylabel("Total Marks")

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

plt.show()
 
# 6. BAR CHART
# Student vs Average Marks
 
plt.figure(figsize=(12, 6))

plt.bar(
    students,
    average_marks,
    label="Average Marks"
)

plt.title("Student vs Average Marks")
plt.xlabel("Students")
plt.ylabel("Average Marks")

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

plt.show()
 
# 7. PIE CHART
# Grade Distribution
 
grade_names = ["A", "B", "C", "D", "F"]

grade_counts = [
    grades.count("A"),
    grades.count("B"),
    grades.count("C"),
    grades.count("D"),
    grades.count("F")
]

# Remove grades with zero students

used_grades = []
used_counts = []

for i in range(len(grade_names)):

    if grade_counts[i] > 0:
        used_grades.append(grade_names[i])
        used_counts.append(grade_counts[i])


plt.figure(figsize=(7, 7))

plt.pie(
    used_counts,
    labels=used_grades,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Student Grade Distribution")

plt.show()
 
# 8.Identify Top 5 Students
 
top_5_indexes = np.argsort(total_marks)[-5:][::-1]

top_5_students = []
top_5_marks = []

for index in top_5_indexes:
    top_5_students.append(students[index])
    top_5_marks.append(total_marks[index])


print("\nTOP 5 STUDENTS")
print("-" * 40)

for i in range(5):
    print(
        i + 1,
        ".",
        top_5_students[i],
        "-",
        top_5_marks[i]
    )
 
# 9. TOP 5 STUDENTS BAR CHART
 
plt.figure(figsize=(8, 5))

plt.bar(
    top_5_students,
    top_5_marks,
    label="Total Marks"
)

plt.title("Top 5 Students")
plt.xlabel("Students")
plt.ylabel("Total Marks")

plt.legend()

plt.tight_layout()

plt.show()
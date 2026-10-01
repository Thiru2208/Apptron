import numpy as np

# 1. Data collection 
np.random.seed(10)
studentId = np.arange(1, 101)
age = np.random.randint(18, 23, size=100)

Mathmarks = np.random.randint(40, 101, 100).astype(float)
ScienceMarks = np.random.randint(40, 101, 100).astype(float)
EnglishMarks = np.random.randint(40, 101, 100).astype(float)
ITMarks = np.random.randint(40, 101, 100).astype(float)

Mathmarks[5] = -10
ScienceMarks[15] = 120
EnglishMarks[10] = np.nan
ITMarks[20] = np.nan

StudentData = np.column_stack((studentId, age, Mathmarks, ScienceMarks, EnglishMarks, ITMarks)) 

print("Student Dataset:")
print(StudentData)

# Data Exploration
print("\nData Exploration:")
print("Shape of the dataset:", StudentData.shape)
print("Dimensions of the dataset:", StudentData.ndim)
print("Total number of elements in the dataset:", StudentData.size)
print("Data type of the dataset:", StudentData.dtype)

# Data Cleaning
print("\nData Cleaning:")

marks = StudentData[:, 2:6].astype(float)

# identify invalid marks (negative or greater than 100)
invalid_marks = (marks < 0) | (marks > 100) | np.isnan(marks)
print("Number of invalid marks:", np.sum(invalid_marks))

# Replace invalid marks with NaN
marks[invalid_marks] = np.nan
print("Marks after Invalid values converted to NaN:")
print(marks)    

# Handle missing data
print("\nHandling Missing Data:")
print("--------------------------")
print("Missing values before cleaning")
print(np.isnan(marks).sum())

# calculate mean while ignoring NaN values
mean_marks = np.nanmean(marks, axis=0)

print("Subject Means")
print("Math mean:", mean_marks[0])
print("Science mean:", mean_marks[1])
print("English mean:", mean_marks[2])
print("IT mean:", mean_marks[3])

#Replace missing values with subject mean
row_index, column_index = np.where(np.isnan(marks))
marks[row_index, column_index] = mean_marks[column_index]

print("\nMarks after handling missing data:")
print(np.isnan(marks).sum())
StudentData[:, 2:6] = marks

# Filter records using conditions
print("High performance students:")
print("--------------------------")

#calculate IT marks greater than 90
high_performance_students = StudentData[marks[:, 3] > 90]

print("ID Age Math Science English IT")
print(high_performance_students)    

# Students with Math > 75 AND Science > 75

good_Math_Science = StudentData[(marks[:, 0] > 75) & (marks[:, 1] > 75)]
print("\nStudents with Math > 75 AND Science > 75:")
print("-------------------------------------------")
print("ID Age Math Science English IT")
print(good_Math_Science)

# Statistical Analysis
print("\nStatistical Analysis:")
print("---------------------")

mean_marks = np.mean(marks, axis=0)
median_marks = np.median(marks, axis=0)
std_marks = np.std(marks, axis=0)
variance_marks = np.var(marks, axis=0)
min_marks = np.min(marks, axis=0)
max_marks = np.max(marks, axis=0)

subjects = ["Math", "Science", "English", "IT"]

for i in range(len(subjects)):
    print("\n", subjects[i])
    print("Mean:", round(mean_marks[i], 2))
    print("Median:", round(median_marks[i], 2))
    print("Standard Deviation:", round(std_marks[i], 2))
    print("Variance:", round(variance_marks[i], 2))
    print("Minimum:", min_marks[i])
    print("Maximum:", max_marks[i])


# Calculate total and average marks

print("\nTotal and Average Marks:")
print("-------------------------")

total_marks = np.sum(marks, axis=1)
average_marks = np.mean(marks, axis=1)

print("First 10 Total Marks:")
print(total_marks[:10])

print("\nFirst 10 Average Marks:")
print(np.round(average_marks[:10], 2))


# Broadcasting

print("\nBroadcasting:")
print("-------------------------")

# Add 5 bonus marks to all subjects
bonus_marks = marks + 5

# Make sure marks do not go above 100
bonus_marks = np.clip(bonus_marks, 0, 100)

print("Original Marks:")
print(marks[:5])

print("\nMarks after adding 5 bonus marks:")
print(bonus_marks[:5])


# Reshaping

print("\nReshaping:")
print("-------------------------")

print("Original shape:", marks.shape)

reshaped_marks = marks.reshape(20, 20)

print("Reshaped shape:", reshaped_marks.shape)

print("\nReshaped Marks:")
print(reshaped_marks)


# Data Transformation

print("\nData Transformation:")
print("-------------------------")

# Normalize marks from 0-100 into 0-1
normalized_marks = marks / 100

print("First 5 Normalized Records:")
print(np.round(normalized_marks[:5], 2))


# Summary Statistics

print("\nSummary Statistics:")
print("-------------------------")

summary_statistics = np.array([
    np.mean(marks, axis=0),
    np.median(marks, axis=0),
    np.std(marks, axis=0),
    np.min(marks, axis=0),
    np.max(marks, axis=0)
])

print("Columns: Math Science English IT")
print("Rows: Mean, Median, Std, Min, Max")

print(np.round(summary_statistics, 2))


# Final Dataset Preparation

print("\nFinal Dataset:")
print("-------------------------")

FinalDataset = np.column_stack((
    StudentData,
    total_marks,
    average_marks
))

print("ID Age Math Science English IT Total Average")
print(np.round(FinalDataset, 2))

print("\nFinal Dataset Shape:")
print(FinalDataset.shape)


# Save Processed Dataset
np.savetxt(
    r"C:\Users\LENOVO\Documents\ProcessedStudentData.csv",
    FinalDataset,
    delimiter=",",
    fmt="%.2f",
    header="ID,Age,Math,Science,English,IT,Total,Average",
    comments=""
)

print("\nProcessed dataset saved successfully.")
print("File name: ProcessedStudentData.csv")
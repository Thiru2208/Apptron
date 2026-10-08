import matplotlib.pyplot as plt

def collect_marks():
    marks = []

    print("Enter marks for 10 students:")

    for i in range(10):
        while True:
            try:
                mark = float(input(f"Enter mark for Student {i + 1}: "))

                if 0 <= mark <= 100:
                    marks.append(mark)
                    break
                else:
                    print("Mark must be between 0 and 100.")

            except ValueError:
                print("Please enter a valid number.")

    return marks


def calculate_mean(marks):
    return sum(marks) / len(marks)


def calculate_median(marks):
    sorted_marks = sorted(marks)
    n = len(sorted_marks)

    middle = n // 2

    if n % 2 == 0:
        return (
            sorted_marks[middle - 1] + sorted_marks[middle]
        ) / 2
    else:
        return sorted_marks[middle]


def calculate_variance(marks):
    mean = calculate_mean(marks)

    squared_differences = []

    for mark in marks:
        difference = mark - mean
        squared_differences.append(difference ** 2)

    variance = sum(squared_differences) / len(marks)

    return variance


def calculate_probability(marks):
    total_students = len(marks)

    passed_students = 0
    failed_students = 0
    above_80_students = 0

    for mark in marks:
        if mark >= 50:
            passed_students += 1
        else:
            failed_students += 1

        if mark > 80:
            above_80_students += 1

    pass_probability = passed_students / total_students
    fail_probability = failed_students / total_students
    above_80_probability = above_80_students / total_students

    return (
        pass_probability,
        fail_probability,
        above_80_probability
    )


def display_statistics(marks):
    mean = calculate_mean(marks)
    median = calculate_median(marks)
    variance = calculate_variance(marks)

    maximum_score = max(marks)
    minimum_score = min(marks)

    pass_probability, fail_probability, above_80_probability = (
        calculate_probability(marks)
    )

    pass_percentage = pass_probability * 100
    fail_percentage = fail_probability * 100

    print("\n--- Student Marks Statistics ---")
    print(f"Marks: {marks}")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Maximum Score: {maximum_score:.2f}")
    print(f"Minimum Score: {minimum_score:.2f}")
    print(f"Variance: {variance:.2f}")
    print(f"Pass Percentage: {pass_percentage:.2f}%")
    print(f"Fail Percentage: {fail_percentage:.2f}%")

    print("\n--- Probability Analysis ---")
    print(
        f"Probability of selecting a passed student: "
        f"{pass_probability:.2f}"
    )
    print(
        f"Probability of selecting a failed student: "
        f"{fail_probability:.2f}"
    )
    print(
        f"Probability of selecting a student above 80: "
        f"{above_80_probability:.2f}"
    )


def generate_graphs(marks):
    student_numbers = list(range(1, len(marks) + 1))

    # Bar Chart
    plt.figure()
    plt.bar(student_numbers, marks)
    plt.title("Student Marks")
    plt.xlabel("Student Number")
    plt.ylabel("Marks")
    plt.xticks(student_numbers)
    plt.ylim(0, 100)
    plt.show()

    # Line Chart
    plt.figure()
    plt.plot(student_numbers, marks, marker="o")
    plt.title("Marks Trend")
    plt.xlabel("Student Number")
    plt.ylabel("Marks")
    plt.xticks(student_numbers)
    plt.ylim(0, 100)
    plt.grid()
    plt.show()

    # Histogram
    plt.figure()
    plt.hist(marks, bins=5, edgecolor="black")
    plt.title("Score Distribution")
    plt.xlabel("Marks Range")
    plt.ylabel("Number of Students")
    plt.show()

    # Scatter Plot
    plt.figure()
    plt.scatter(student_numbers, marks)
    plt.title("Student Number vs Marks")
    plt.xlabel("Student Number")
    plt.ylabel("Marks")
    plt.xticks(student_numbers)
    plt.ylim(0, 100)
    plt.grid()
    plt.show()


def main():
    print("Student Exam Score Analysis System")

    marks = collect_marks()

    display_statistics(marks)

    generate_graphs(marks)


if __name__ == "__main__":
    main()
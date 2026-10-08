students = []

while True:
    print()
    print("===================================")
    print("Student Management System")
    print("1. Add Student")
    print("2. View All Students")
    print("3. Search Student by ID")
    print("4. Update Student Marks")
    print("5. Delete Student")
    print("6. Display Grade Report")
    print("7. Exit")    
    print("===================================")
    print()

    choice = input("Enter your choice (1-7): ")

    #Add Student
    if choice == '1':
        while True:
            try:
                student_id = int(input("Enter Student ID: "))

                duplicate = False
                
                for student in students:
                    if student["id"] == student_id:
                       duplicate = True
                       break
                if duplicate:
                    print("Student ID already exists. Please enter a unique ID.")
                else:
                    break
            except ValueError:
                print("Invalid input. Student ID must be an integer.")

        while True:
            name = input("Enter Student Name: ").strip()

            if name == "":
                print("Student name cannot be empty. Please try again.")

            elif not all(char.isalpha() or char.isspace() for char in name):
                print("Student name must contain only letters and spaces. Please try again.")
            else:
                break    

        while True:
            marks = input("Enter Student Marks (0-100): ")
            try:
                marks = float(marks)
                if 0 <= marks <= 100:
                    break
                else:
                    print("Marks must be between 0 and 100. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        student = {
            'id': student_id,
            'name': name,
            'marks': marks
        }

        students.append(student)
        print("Student added successfully.")

    #View All Students
    elif choice == '2':
        if len(students) == 0:
            print("No students found.")
            continue
        print("All Student list:")

        for student in students:
            print(f"ID: {student['id']}, Name: {student['name']}, Marks: {student['marks']}")

    #Search Student by ID
    elif choice == '3':
        while True:
            try:
                student_id = int(input("Enter Student ID to search: "))
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

        found = False
        for student in students:
            if student['id'] == student_id:
                print("Student found:")
                print(f"ID: {student['id']}, Name: {student['name']}, Marks: {student['marks']}")
                found = True
        if not found:
            print("Student not found.")

    #Update Student Marks
    elif choice == '4':
        while True:
            try:
                student_id = int(input("Enter Student ID to update marks: "))
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

        for student in students:
            if student["id"] == student_id:

                while True:
                    try:
                        new_marks = float(input("Enter new marks (0-100): "))

                        if 0 <= new_marks <= 100:
                                student["marks"] = new_marks
                                print("Marks updated successfully.")
                                break
                        else:
                                print("Marks must be between 0 and 100. Please try again.")
                    except ValueError:
                            print("Invalid input. Please enter a number.")
                break
        else:
            print("Student not found.")

#Delete Student
    elif choice == '5':
        try:
            student_id = int(input("Enter Student ID to delete: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        for student in students:
            if student["id"] == student_id:
                students.remove(student)
                print("Student deleted successfully.")
                break
        else:
            print("Student not found.")

#Display Grade Report
    elif choice == '6':
        if len(students) == 0:
            print("No students found.")
            continue

        total = 0
        topper = students[0]

        print("Grade Report:")

        for student in students:
            marks = student['marks']
            if marks >= 90:
                grade = 'A'
            elif marks >= 75:
                grade = 'B'
            elif marks >= 50:
                grade = 'C'
            elif marks >= 40:
                grade = 'D'
            else:
                grade = 'F'

            print(f"ID: {student['id']}, Name: {student['name']}, Marks: {marks}, Grade: {grade}")

            total += marks
            if marks > topper['marks']:
                topper = student

        average = total / len(students)

        print("Average Marks:", round(average, 2))
        print("Topper:", topper['name'], "with Highest Marks:", topper['marks'])

    #Exit
    elif choice == '7':
        print("Exiting the program.")
        break
    else:
        print("Invalid choice. Please enter a valid option (1-7).")
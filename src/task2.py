"""
Task 2: Student Sorting (Variant S7)
Sort all students by m_phone_number and save to CSV file.

Implementations:
1. sort_students_builtin: Standard Python sort (Timsort O(N log N)).
2. sort_students_radix: Custom LSD Radix Sort for phone numbers (O(d * N) = O(N)).
"""

from typing import List

from src.models import Student, write_data


def sort_students_builtin(students: List[Student]) -> List[Student]:
    """Sort students by phone number using Python's standard sorted() (Timsort)."""
    return sorted(students, key=lambda s: s.phone_number)


def sort_students_radix(students: List[Student]) -> List[Student]:
    """
    Sort students by phone number using custom LSD (Least Significant Digit) Radix Sort.
    Phone number format: '38(0xx)xx-xx-xxx' (length 16).
    Fixed characters: '38(0' at [0..3], ')' at [6], '-' at [9], '-' at [12].
    Variable digits at indices: [4, 5, 7, 8, 10, 11, 13, 14, 15].
    Sorting passes run from least significant digit (index 15) to most significant digit (index 4).
    Each pass performs stable counting sort / bucket distribution in O(N).
    Total time complexity: O(d * N) where d = 9 variable digits.
    """
    if len(students) <= 1:
        return list(students)

    current = list(students)
    # Positions of variable digits in reverse order (LSD)
    variable_positions = [15, 14, 13, 11, 10, 8, 7, 5, 4]

    for pos in variable_positions:
        buckets: list[list[Student]] = [[] for _ in range(10)]
        for s in current:
            # ord('0') is 48
            digit = ord(s.phone_number[pos]) - 48
            buckets[digit].append(s)

        # Flatten buckets stably
        current = [s for bucket in buckets for s in bucket]

    return current


def save_sorted_to_csv(students: List[Student], output_filename: str) -> None:
    """Save sorted students list to CSV file with proper headers."""
    write_data(output_filename, students)


if __name__ == "__main__":
    import os

    from src.models import read_data

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "students.csv")
    students = read_data(data_path, limit=1000)

    s1 = sort_students_builtin(students)
    s2 = sort_students_radix(students)

    assert [s.phone_number for s in s1] == [s.phone_number for s in s2], (
        "Radix sort mismatch!"
    )
    print("Task 2 sort check passed! 1000 students sorted identically.")

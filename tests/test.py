"""
Unit and integration tests for Homework 1.
"""

import os
import unittest

from src.generator import generate_operations
from src.models import Student, db, read_data, write_data
from src.task1 import StudentDBIndexed, StudentDBNaive, StudentDBOptimized
from src.task2 import save_sorted_to_csv, sort_students_builtin, sort_students_radix


class TestStudentModel(unittest.TestCase):
    def test_type_conversion_post_init(self):
        s = Student(
            name="Іван",
            surname="Коваленко",
            email="ivan.kovalenko@student.org",
            birth_year="2005",
            birth_month="6",
            birth_day="15",
            group="ПКН25/Б-1",
            rating="87.5",
            phone_number="38(050)12-34-567",
        )
        self.assertIsInstance(s.birth_year, int)
        self.assertEqual(s.birth_year, 2005)
        self.assertIsInstance(s.birth_month, int)
        self.assertEqual(s.birth_month, 6)
        self.assertIsInstance(s.birth_day, int)
        self.assertEqual(s.birth_day, 15)
        self.assertIsInstance(s.rating, float)
        self.assertAlmostEqual(s.rating, 87.5)

    def test_property_aliases(self):
        s = Student(
            "Олена",
            "Шевченко",
            "olena@student.org",
            2004,
            3,
            10,
            "ПСА25/Б-1",
            95.0,
            "38(067)11-22-333",
        )
        self.assertEqual(s.m_name, "Олена")
        self.assertEqual(s.m_surname, "Шевченко")
        self.assertEqual(s.m_email, "olena@student.org")
        self.assertEqual(s.m_birth_year, 2004)
        self.assertEqual(s.m_group, "ПСА25/Б-1")
        self.assertEqual(s.m_phone_number, "38(067)11-22-333")


class TestTask1Operations(unittest.TestCase):
    def setUp(self):
        # Create a controlled set of 150 students
        self.students = []
        for i in range(150):
            grp = f"GRP-{i % 5}"
            rating = float((i * 7) % 100)
            s = Student(
                name=f"Name_{i}",
                surname=f"Surname_{i}",
                email=f"student_{i}@student.org",
                birth_year=2000 + (i % 10),
                birth_month=1 + (i % 12),
                birth_day=1 + (i % 28),
                group=grp,
                rating=rating,
                phone_number=f"38(050)10-00-{i:03d}",
            )
            self.students.append(s)

        self.db_naive = StudentDBNaive(self.students)
        self.db_indexed = StudentDBIndexed(self.students)
        self.db_optimized = StudentDBOptimized(self.students)

    def test_top_100_consistency(self):
        t_naive = self.db_naive.get_top_100_students()
        t_idx = self.db_indexed.get_top_100_students()
        t_opt = self.db_optimized.get_top_100_students()

        self.assertEqual(len(t_naive), 100)
        self.assertEqual(len(t_idx), 100)
        self.assertEqual(len(t_opt), 100)

        # Check descending order
        for t_list in [t_naive, t_idx, t_opt]:
            for i in range(len(t_list) - 1):
                self.assertGreaterEqual(t_list[i].rating, t_list[i + 1].rating)

        # Check ratings match across all 3 approaches
        r_naive = [s.rating for s in t_naive]
        r_idx = [s.rating for s in t_idx]
        r_opt = [s.rating for s in t_opt]
        self.assertEqual(r_naive, r_idx)
        self.assertEqual(r_naive, r_opt)

    def test_update_student_rating(self):
        target_email = "student_42@student.org"
        new_rating = 99.99

        # Update in all 3
        self.assertTrue(self.db_naive.set_student_rating(target_email, new_rating))
        self.assertTrue(self.db_indexed.set_student_rating(target_email, new_rating))
        self.assertTrue(self.db_optimized.set_student_rating(target_email, new_rating))

        # Check that top student is now student_42
        self.assertEqual(self.db_naive.get_top_100_students()[0].email, target_email)
        self.assertEqual(self.db_indexed.get_top_100_students()[0].email, target_email)
        self.assertEqual(
            self.db_optimized.get_top_100_students()[0].email, target_email
        )

    def test_highest_average_group_consistency(self):
        g_naive = self.db_naive.get_groups_with_highest_average_rating()
        g_idx = self.db_indexed.get_groups_with_highest_average_rating()
        g_opt = self.db_optimized.get_groups_with_highest_average_rating()

        self.assertEqual(g_naive, g_idx)
        self.assertEqual(g_naive, g_opt)

    def test_random_operations_equivalence(self):
        ops = generate_operations(
            self.students, total_count=100, ratio=(10, 10, 1), seed=999
        )
        for op in ops:
            if op[0] == 1:
                r1 = [s.rating for s in self.db_naive.get_top_100_students()]
                r2 = [s.rating for s in self.db_indexed.get_top_100_students()]
                r3 = [s.rating for s in self.db_optimized.get_top_100_students()]
                self.assertEqual(r1, r2)
                self.assertEqual(r1, r3)
            elif op[0] == 2:
                _, email, new_val = op
                u1 = self.db_naive.set_student_rating(email, new_val)
                u2 = self.db_indexed.set_student_rating(email, new_val)
                u3 = self.db_optimized.set_student_rating(email, new_val)
                self.assertEqual(u1, u2)
                self.assertEqual(u1, u3)
            elif op[0] == 3:
                g1 = self.db_naive.get_groups_with_highest_average_rating()
                g2 = self.db_indexed.get_groups_with_highest_average_rating()
                g3 = self.db_optimized.get_groups_with_highest_average_rating()
                self.assertEqual(g1, g2)
                self.assertEqual(g1, g3)


class TestTask2Sorting(unittest.TestCase):
    def test_sorting_equivalence(self):
        test_phones = [
            "38(067)99-88-777",
            "38(050)11-22-333",
            "38(093)00-11-222",
            "38(050)00-00-001",
            "38(067)11-22-333",
            "38(050)11-22-332",
        ]
        students = [
            Student(f"N{i}", f"S{i}", f"e{i}@s.org", 2000, 1, 1, "GRP", 50.0, ph)
            for i, ph in enumerate(test_phones)
        ]

        sorted_builtin = sort_students_builtin(students)
        sorted_radix = sort_students_radix(students)

        phones_builtin = [s.phone_number for s in sorted_builtin]
        phones_radix = [s.phone_number for s in sorted_radix]

        self.assertEqual(phones_builtin, phones_radix)
        # Check non-decreasing
        for i in range(len(phones_builtin) - 1):
            self.assertLessEqual(phones_builtin[i], phones_builtin[i + 1])


if __name__ == "__main__":
    unittest.main()

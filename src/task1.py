"""
Task 1: In-memory student database implementations (Variant V5)
Operations:
1. get_top_100_students() -> list[Student]: Return 100 students with highest rating.
2. set_student_rating(email: str, new_rating: float) -> bool: Set rating for student by email.
3. get_groups_with_highest_average_rating() -> list[str]: Find group(s) with highest average rating.

Three approaches implemented:
- StudentDBNaive: List-based baseline (O(N log N) / O(N) operations).
- StudentDBIndexed: Hash maps for email & groups + Heap for top-100 (O(N) top-100, O(1) update, O(G) group avg).
- StudentDBOptimized: Rating buckets + Hash map + Cached group tracking (O(1) / O(K) top-100, O(1) update, O(1) amortized group avg).
"""

import heapq
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

from src.models import Student


class BaseStudentDB(ABC):
    @abstractmethod
    def get_top_100_students(self) -> list[Student]:
        """Operation 1: Return top 100 students by rating (descending)."""
        pass

    @abstractmethod
    def set_student_rating(self, email: str, new_rating: float) -> bool:
        """Operation 2: Update student rating by email."""
        pass

    @abstractmethod
    def get_groups_with_highest_average_rating(self) -> list[str]:
        """Operation 3: Return list of groups with maximum average student rating."""
        pass


# ==============================================================================
# Approach 1: Naive (List-based baseline)
# ==============================================================================
class StudentDBNaive(BaseStudentDB):
    def __init__(self, students: list[Student]):
        # Store shallow copy of student references
        self.students: list[Student] = [
            Student(
                name=s.name,
                surname=s.surname,
                email=s.email,
                birth_year=s.birth_year,
                birth_month=s.birth_month,
                birth_day=s.birth_day,
                group=s.group,
                rating=s.rating,
                phone_number=s.phone_number,
            )
            for s in students
        ]

    def get_top_100_students(self) -> list[Student]:
        # O(N log N) full sort
        return sorted(self.students, key=lambda s: s.rating, reverse=True)[:100]

    def set_student_rating(self, email: str, new_rating: float) -> bool:
        # O(N) linear search
        for s in self.students:
            if s.email == email:
                s.rating = float(new_rating)
                return True
        return False

    def get_groups_with_highest_average_rating(self) -> list[str]:
        # O(N) full recomputation of all group averages
        group_sums: dict[str, float] = defaultdict(float)
        group_counts: dict[str, int] = defaultdict(int)

        for s in self.students:
            group_sums[s.group] += s.rating
            group_counts[s.group] += 1

        if not group_counts:
            return []

        max_avg = -1.0
        best_groups: list[str] = []
        for g, count in group_counts.items():
            avg = group_sums[g] / count
            if avg > max_avg + 1e-9:
                max_avg = avg
                best_groups = [g]
            elif abs(avg - max_avg) <= 1e-9:
                best_groups.append(g)

        return sorted(best_groups)


# ==============================================================================
# Approach 2: Indexed (Hash maps + Min-Heap / nlargest)
# ==============================================================================
class StudentDBIndexed(BaseStudentDB):
    def __init__(self, students: list[Student]):
        self.students: list[Student] = [
            Student(
                name=s.name,
                surname=s.surname,
                email=s.email,
                birth_year=s.birth_year,
                birth_month=s.birth_month,
                birth_day=s.birth_day,
                group=s.group,
                rating=s.rating,
                phone_number=s.phone_number,
            )
            for s in students
        ]
        # O(1) email lookup
        self.email_to_student: dict[str, Student] = {s.email: s for s in self.students}

        # O(G) group tracking: group -> [total_rating, count]
        self.group_totals: dict[str, float] = defaultdict(float)
        self.group_counts: dict[str, int] = defaultdict(int)
        for s in self.students:
            self.group_totals[s.group] += s.rating
            self.group_counts[s.group] += 1

    def get_top_100_students(self) -> list[Student]:
        # O(N log 100) using heapq.nlargest
        return heapq.nlargest(100, self.students, key=lambda s: s.rating)

    def set_student_rating(self, email: str, new_rating: float) -> bool:
        s = self.email_to_student.get(email)
        if s is None:
            return False

        new_val = float(new_rating)
        delta = new_val - s.rating
        self.group_totals[s.group] += delta
        s.rating = new_val
        return True

    def get_groups_with_highest_average_rating(self) -> list[str]:
        # O(G) scan over distinct groups (G << N)
        if not self.group_counts:
            return []

        max_avg = -1.0
        best_groups: list[str] = []
        for g, count in self.group_counts.items():
            avg = self.group_totals[g] / count
            if avg > max_avg + 1e-9:
                max_avg = avg
                best_groups = [g]
            elif abs(avg - max_avg) <= 1e-9:
                best_groups.append(g)

        return sorted(best_groups)


# ==============================================================================
# Approach 3: Optimized (Rating Buckets + Hash Map + Incremental Group Caching)
# ==============================================================================
class StudentDBOptimized(BaseStudentDB):
    def __init__(self, students: list[Student], num_buckets: int = 500):
        self.num_buckets = num_buckets
        self.buckets: list[dict[str, Student]] = [dict() for _ in range(num_buckets)]
        self.email_to_student: dict[str, Student] = {}

        self.group_totals: dict[str, float] = defaultdict(float)
        self.group_counts: dict[str, int] = defaultdict(int)

        for s in students:
            copy_s = Student(
                name=s.name,
                surname=s.surname,
                email=s.email,
                birth_year=s.birth_year,
                birth_month=s.birth_month,
                birth_day=s.birth_day,
                group=s.group,
                rating=s.rating,
                phone_number=s.phone_number,
            )
            self.email_to_student[copy_s.email] = copy_s
            b_idx = self._bucket_idx(copy_s.rating)
            self.buckets[b_idx][copy_s.email] = copy_s

            self.group_totals[copy_s.group] += copy_s.rating
            self.group_counts[copy_s.group] += 1

        self.cached_best_groups: list[str] = []
        self.cached_best_avg: float = -1.0
        self.groups_dirty: bool = True
        self._recompute_best_groups()

    def _bucket_idx(self, rating: float) -> int:
        idx = int(rating / 100.0 * self.num_buckets)
        if idx < 0:
            return 0
        if idx >= self.num_buckets:
            return self.num_buckets - 1
        return idx

    def _recompute_best_groups(self) -> None:
        if not self.group_counts:
            self.cached_best_groups = []
            self.cached_best_avg = -1.0
            self.groups_dirty = False
            return

        max_avg = -1.0
        best_groups: list[str] = []
        for g, count in self.group_counts.items():
            avg = self.group_totals[g] / count
            if avg > max_avg + 1e-9:
                max_avg = avg
                best_groups = [g]
            elif abs(avg - max_avg) <= 1e-9:
                best_groups.append(g)

        self.cached_best_groups = sorted(best_groups)
        self.cached_best_avg = max_avg
        self.groups_dirty = False

    def get_top_100_students(self) -> list[Student]:
        """O(1) / O(K log K) top-100 collection from highest rating buckets."""
        result: list[Student] = []
        # Traverse buckets in descending order
        for b_idx in range(self.num_buckets - 1, -1, -1):
            bucket = self.buckets[b_idx]
            if not bucket:
                continue

            b_len = len(bucket)
            if len(result) + b_len <= 100:
                # Add all students from bucket sorted descending
                result.extend(
                    sorted(bucket.values(), key=lambda s: s.rating, reverse=True)
                )
                if len(result) == 100:
                    break
            else:
                needed = 100 - len(result)
                sorted_bucket = sorted(
                    bucket.values(), key=lambda s: s.rating, reverse=True
                )
                result.extend(sorted_bucket[:needed])
                break

        return result

    def set_student_rating(self, email: str, new_rating: float) -> bool:
        """O(1) update across email index, rating buckets, and group tracking."""
        s = self.email_to_student.get(email)
        if s is None:
            return False

        new_val = float(new_rating)
        old_val = s.rating
        if old_val == new_val:
            return True

        old_b = self._bucket_idx(old_val)
        new_b = self._bucket_idx(new_val)

        if old_b != new_b:
            del self.buckets[old_b][email]
            s.rating = new_val
            self.buckets[new_b][email] = s
        else:
            s.rating = new_val

        # Update group stats
        delta = new_val - old_val
        self.group_totals[s.group] += delta
        count = self.group_counts[s.group]
        new_group_avg = self.group_totals[s.group] / count

        # Check impact on best groups cache
        if self.groups_dirty:
            pass
        else:
            if new_group_avg > self.cached_best_avg + 1e-9:
                self.cached_best_avg = new_group_avg
                self.cached_best_groups = [s.group]
            elif abs(new_group_avg - self.cached_best_avg) <= 1e-9:
                if s.group not in self.cached_best_groups:
                    self.cached_best_groups.append(s.group)
                    self.cached_best_groups.sort()
            elif s.group in self.cached_best_groups and delta < 0:
                # Top group decreased, mark dirty for lazy recomputation
                self.groups_dirty = True

        return True

    def get_groups_with_highest_average_rating(self) -> list[str]:
        """O(1) amortized cached return of best group(s)."""
        if self.groups_dirty:
            self._recompute_best_groups()
        return list(self.cached_best_groups)

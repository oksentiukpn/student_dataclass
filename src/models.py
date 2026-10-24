import csv
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class Student:
    name: str = ""
    surname: str = ""
    email: str = ""

    birth_year: int = 0
    birth_month: int = 0
    birth_day: int = 0

    group: str = ""
    rating: float = 0.0

    phone_number: str = ""

    def __post_init__(self):
        self.birth_year = int(self.birth_year)
        self.birth_month = int(self.birth_month)
        self.birth_day = int(self.birth_day)
        self.rating = float(self.rating)

    @property
    def m_name(self) -> str:
        return self.name

    @property
    def m_surname(self) -> str:
        return self.surname

    @property
    def m_email(self) -> str:
        return self.email

    @property
    def m_birth_year(self) -> int:
        return self.birth_year

    @property
    def m_birth_month(self) -> int:
        return self.birth_month

    @property
    def m_birth_day(self) -> int:
        return self.birth_day

    @property
    def m_group(self) -> str:
        return self.group

    @property
    def m_rating(self) -> float:
        return self.rating

    @m_rating.setter
    def m_rating(self, value: float):
        self.rating = float(value)

    @property
    def m_phone_number(self) -> str:
        return self.phone_number

    def to_csv_row(self) -> list[str]:
        return [
            self.name,
            self.surname,
            self.email,
            str(self.birth_year),
            str(self.birth_month),
            str(self.birth_day),
            self.group,
            f"{self.rating:.2f}"
            if isinstance(self.rating, float)
            else str(self.rating),
            self.phone_number,
        ]


class db:
    students: list[Student]

    def __init__(self, students: Optional[list[Student]] = None):
        self.students = students if students is not None else []


def read_data(
    filename: str, my_db: Optional[db] = None, limit: Optional[int] = None
) -> list[Student]:
    students: list[Student] = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            students.append(Student(*row))
            if limit is not None and len(students) >= limit:
                break

    if my_db is not None:
        my_db.students = students
    return students


def write_data(filename: str, students: list[Student]) -> None:
    header = [
        "m_name",
        "m_surname",
        "m_email",
        "m_birth_year",
        "m_birth_month",
        "m_birth_day",
        "m_group",
        "m_rating",
        "m_phone_number",
    ]
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for s in students:
            writer.writerow(s.to_csv_row())


if __name__ == "__main__":
    import os

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "students.csv")
    my_db = db()
    read_data(data_path, my_db, limit=5)
    print("Loaded 5 students:")
    for s in my_db.students:
        print(s.name, s.rating, s.phone_number, type(s.birth_month))

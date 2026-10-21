from csv import reader
from dataclasses import dataclass

DATA_FILE = "../data/students.csv"


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


class db:
    students: list[Student]


def read_data(filename: str, my_db: db):
    my_db.students = []
    with open(filename, "r", encoding="utf-8") as f:
        f.readline()
        r = reader(f)
        for row in r:
            my_db.students.append(Student(*row))


def write_data(filename: str): ...


if __name__ == "__main__":
    my_db = db()
    read_data(DATA_FILE, my_db)
    print(type(my_db.students[2].birth_day))

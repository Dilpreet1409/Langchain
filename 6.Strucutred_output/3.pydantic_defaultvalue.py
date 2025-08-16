from pydantic import BaseModel

class Student(BaseModel):

    name: str = 'Dolly'


new_student = {}

student = Student(**new_student)

print(student.name)
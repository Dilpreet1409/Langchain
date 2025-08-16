from pydantic import BaseModel,EmailStr,Field
from typing import Optional
class Student(BaseModel):

    name: str = 'Dolly'
    agr: Optional[int]= None
    email:EmailStr
    cgpa: float= Field(gt=0,lt=10,default=5,description='A decimal value representing the cgpaof the student') #constraint

new_student = {'age':32, 'email':'dolly@gmail.com'}

student = Student(**new_student)

student_dict= dict(student)
print(student_dict['age'])

student_json=student.model_dump_json()
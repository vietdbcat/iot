import pandas as pd
from person import Person

dt = pd.read_excel("data.xlsx")

# danh sách các đối tượng có trong dữ liệu
person_list = []
for index, row in dt.iterrows():
    person = Person(
        ten=row['MSV'],
        msv=row['Name'],
        ngaysinh=row['Dob'],
        diachi=row['Address'],
        lop=row['Class']
    )
    person_list.append(person)
    


import matplotlib.pyplot as p
classes=['Class A','Class B','Class C','Class D','Class E']
n_students=[58,12,17,28,30]
p.figure(figsize=(10,6))
p.bar(classes,n_students,color='pink')
p.xlabel('classes')
p.ylabel('no of studnets')
p.title('no of students in different classes')
for i,value in enumerate(n_students):
p.text(i,value+0.5,str(value),ha='center')
p.show()

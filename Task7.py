import seaborn as s
import matplotlib.pyplot as p
tips=s.load_dataset("tips")
p.figure(figsize=(10,6))
s.histplot(tips['total_bill'],bins=30,kde=True,color='blue')
p.xlabel('Total bills')
p.ylabel("frequency")
p.title("distribution of total bills in tips dataset")
p.show()

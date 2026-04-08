import math
from collections import Counter
#set up data:

point=['A','B','C','D']
x1=[7,8,5,9]

x2=[2,1,4,6]
labels=['pos','neg','pos','neg']
p=[8 ,2]
k=3

#2. calculate Euclidean Distances:

dataset=[]
for i in range(len(point)):
  dist= math.sqrt((p[0]-x1[i])**2+(p[1]-x2[i])**2)
  dataset.append({'name' : point[i],'dist' : dist,'label' : labels[i]})
  dataset.sort(key=lambda x:x['dist'])
  neighbors=dataset[:k]
  neighbor_labels=[n['label'] for n in neighbors]
prediction= Counter (neighbor_labels).most_common(1)[0][0]

print(f"target point p {p} results:")
print("-" * 30)
for n in neighbors:
  print(f"neighbor {n['name']}: distances {n['dist']:.2f}, class: {n['label']}")

print("-" * 30)
print(f"final prediction for p : {prediction}")

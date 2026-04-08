import math
point=['a','b','c','d']
x1=[7,8,5,9]
x2=[2,1,4,6]
p=[8 ,2]
k=3

distances=[]
for i in range(len(point)):
  dist= math.sqrt((p[0]-x1[i])**2+(p[1]-x2[i])**2)
  distances.append((point[i],dist))
  distances.sort(key=lambda x:x[1])
  neighbors=distances[:k]
  print(f"the {k} nearest neighbors to point {p} are: ")
for name,distances in neighbors:
  print(f"point {name} with distance {dist:.2f}")

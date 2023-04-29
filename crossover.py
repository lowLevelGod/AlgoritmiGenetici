
fin = open('input.txt')

l = int(fin.readline().strip())

c1 = fin.readline().strip()
c2 = fin.readline().strip()

pct = int(fin.readline().strip())

res1 = c1[:pct] + c2[pct:]
res2 = c2[:pct] + c1[pct:]

print(res1)
print(res2)
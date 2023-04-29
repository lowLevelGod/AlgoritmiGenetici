fin = open('input.txt')

n, m = [int(x) for x in fin.readline().strip().split()]

c = [int(x) for x in fin.readline().strip()]

mutatii = [int(x) for x in fin.readline().strip().split()]

for m in mutatii:
    c[m] = 1 - c[m]
    
print("".join([str(x) for x in c]))

import itertools
from itertools import combinations

file = open("c_m_unique.txt")
data = [line.strip() for line in file.readlines()]

merchant_data = set(line.strip() for line in open('merchant_id.txt'))
merchants = set([])
c_map = {}

for line in data:
    line = line.split(',')
    c = line[0]
    for i in range(1, len(line)):
        m = line[i].strip()
        if m in merchant_data:
            merchants.add(m)
            if m not in c_map:
                c_map[m] = set([])
            c_map[m].add(c)

file.close()
with open("results.txt", 'w') as results:
    results.write('Merchant 1' + '  Merchant 2' + '   Weight' + '   Shared Customers\n')
    for m1, m2 in combinations(merchants, 2):
        c1 = c_map[m1]
        c2 = c_map[m2]
        common = c1.intersection(c2)
        if common:
            results.write(m1 + ',  ' + m2 + ',  ' + str(len(common)) + ',  ' + str(common) + '\n')           
results.close()



        


    


    
    

test = open('../../data/test.txt','r')
weird = open('./centrality_result/weird_cent.txt','r')


test_list = test.readlines()
test_list = [i.rstrip('\n') for i in test_list] # rstrip('\n')
#print(test_list)

weird_list = weird.readline().split(',')
#print(weird_list)

c = list(set(test_list) & set(weird_list))
c.sort()
print(c)
print(len(c))

f = open('./centrality_result/co_weird_test.txt','w')
f.write(','.join(map(str,c)))

test.close()
weird.close()
f.close()
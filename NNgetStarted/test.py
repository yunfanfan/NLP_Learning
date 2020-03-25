from functools import reduce

input = 5
bias = 0.44
weights = [2.0 for _ in range(input)]
input_vec = [0.7 for _ in range(input)]
#print(weights)
mmap = map(lambda x, w: x * w, input_vec, weights)
reduce1 = reduce(lambda a,b:a+b,mmap, 0.1)
# for x in mmap:
#     print(x, end = ' ')
print(mmap)
print(reduce1)
#print('weights\t:%s\nbias\t:%f\n' % (weights, bias))

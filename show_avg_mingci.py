import pickle

# f = open('all_map', 'rb+')
# the_map = pickle.load(f)
# st = 0
# all = 0
#
# the_dict = dict()
# for k, v in the_map.items():
#     # print(k, v[k])
#     the_dict[k] = v[k]
#     if v[k] < 1:
#
#         st += 1
#     all += 1
# f.close()
#
# new_dict = sorted(the_dict.items(),key = lambda t:int(t[0]))
# print(new_dict)
# print(st/all)

f = open('positive_num', 'rb+')
positive_num = pickle.load(f)
f.close()


f = open('num_map', 'rb+')
num_map = pickle.load(f)
f.close()

result = dict()

for i in range(1, 55):
    if str(i) not in positive_num:
        positive_num[str(i)] = 0

    result[str(i)] = positive_num[str(i)] / num_map[str(i)]
    print(result[str(i)])
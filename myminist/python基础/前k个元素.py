def topKFrequent(nums, k):
    if k ==1 and len(nums) == 1:
        return nums

    n_dict = {}
    for n in nums:
        if n in n_dict:
            n_dict[n] += 1
        else:
            n_dict[n] = 1

    print(n_dict)

    res = n_dict.items()

    s = sorted(res, key=lambda item: item[1], reverse=True)

    return [s[i][0] for i in range(k)]


print(topKFrequent([4, 1, -1, 2, -1, 2, 3], 2))


# sorted([1, 2, 3, 4, 5, 6, 7, 8, 9], key=lambda x: abs(5-x))
# 将列表[1, 2, 3, 4, 5, 6, 7, 8, 9]按照元素与5距离从小到大进行排序，其结果是[5, 4, 6, 3, 7, 2, 8, 1, 9]。
#
# my_dict = {"a": "2", "c": "5", "b": "1"}
# result2 = sorted(my_dict, key=lambda x: my_dict[x])
# print(result2)
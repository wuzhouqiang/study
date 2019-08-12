def threeSum(nums):
    if len(nums) < 3:
        return []

    d = {}
    res = set()

    for i, v in enumerate(nums):
        d[v] = i

    for j in range(len(nums)):
        for k in range(j+1, len(nums)):
            if -nums[i] - nums[j] in d:
                res.add(tuple(sorted([nums[i], nums[j], -nums[i] - nums[j]])))

    return map(list, res)


print(list(threeSum([2, -1, 1, 0, -2])))


# dict_1 = {'33': '1', 'oo': '2'}
# if '33' in dict_1:
#   print(dict_1['33'])
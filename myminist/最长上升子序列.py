# [1, 3, 2, 5,1,6]


# def twoSum(nums, target):
#     n = len(nums)
#
#     for i in range(n):
#         for j in range(n):
#             if i != j and nums[i] + nums[j] == target:
#                 return [i, j]


# def twoSum(nums, target):
#
#     mydict = {}
#     for i, element in enumerate(nums):
#         mydict[element] = i
#
#     print(mydict)
#
#     for i, element in enumerate(nums):
#         j = mydict.get(target - element)
#         if j is not None and i != j:
#             return [i, j]
#
# print(twoSum([2, 7, 11, 15], 9))

def quick_sort(nums):

    if len(nums) <= 1:
        return nums

    base = nums[0]
    left = [x for x in nums[1:] if x <= base]
    right = [x for x in nums[1:] if x > base]

    return quick_sort(left) + [base] + quick_sort(right)

print(quick_sort([3, 4, 6, 2, 1]))
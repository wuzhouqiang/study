
def MaxSubarray(nums):  # [1,-2,3,1,-3,4,1]
    max_sum = cur_sum = 0
    for num in nums:
        cur_sum = max(cur_sum + num, 0)
        max_sum = max(max_sum, cur_sum)
    return max_sum


print(MaxSubarray([1, -2, 3, 1, -3, 4, 1]))
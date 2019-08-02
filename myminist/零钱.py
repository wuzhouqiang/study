def coinChange(coins, amount):
    """
    :type coins: List[int]
    :type amount: int
    :rtype: int
    """
    res = [0 for _ in range(amount + 1)]

    for i in range(1, amount + 1):
        cost = float('inf')
        for c in coins:
            if i - c >= 0:
                cost = min(cost, res[i - c] + 1)

        res[i] = cost

    if res[amount] == float('inf'):
        return -1
    else:
        return res[amount]


# def fb(n):
#     if n <= 2:
#         return 1
#
#     return fb(n-1) + fb(n-2)
#
# def fb_2(n):
#     res = [0 for _ in range(n)]
#
#     for i in range(n):
#         if i <= 1:
#             res[i] = 1
#             continue
#         res[i] = res[i-1] + res[i-2]
#
#     return res[n-1]


# print(fb_2(4))

def maxProfit(prices) -> int:
    # dp[i] = dp[i-1] if prices[i]<=prices[i-1] else dp[i-1]+(prices[i]-prices[i-1])
    res = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            res += (prices[i] - prices[i - 1])
    return res


print(maxProfit([4, 1, 3, 5, 7, 3]))
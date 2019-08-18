import math


def numSquares(n: int) -> int:
    max_num = int(math.floor(n ** 0.5))
    # 列出符合条件的完全平方数
    nums = [i * i for i in range(1, max_num + 1)]

    dp = [0 for _ in range(0, n + 1)]

    for i in range(1, n + 1):
        count = float('inf')
        for num in nums:
            if i - num >= 0:
                count = min(count, dp[i - num] + 1)
            else:
                break
        dp[i] = count

    return dp[n]


if __name__ == '__main__':
    print(numSquares(7115))
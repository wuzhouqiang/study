

def min_distance(A, B):
    m = len(A)
    n = len(B)
    f = [[0 for _ in range(n+1)] for _ in range(m+1)]  # A的前m个  和b的前n个字符的最小距离

    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                f[i][j] = j
                continue
            if j == 0:
                f[i][j] = i
                continue

            if A[i-1] == B[j-1]:
                f[i][j] = f[i-1][j-1]
            else:
                f[i][j] = min(f[i-1][j-1], f[i][j-1], f[i-1][j]) + 1

    return f[m][n]


print(min_distance('hellow', 'helle'))
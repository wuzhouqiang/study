def isValid(self, s: str) -> bool:
    stack = []
    lookup = {
        "(": ")",
        "[": "]",
        "{": "}"
    }
    for alp in s:
        if alp in lookup:
            stack.append(alp)
            continue
        if stack and lookup[stack[-1]] == alp:
            stack.pop()
        else:
            return False
    return True if not stack else False



if __name__ == '__main__':
    print(isValid("{}(]"))
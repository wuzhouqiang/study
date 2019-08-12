def isAnagram(s: str, t: str) -> bool:
    sDict = {}
    tDict={}
    if len(s) != len(t):
        return False

    for element in s:
        if element in sDict:
            sDict[element] += 1
        else:
            sDict[element] = 1

    for element in t:
        if element in tDict:
            tDict[element] += 1
        else:
            tDict[element] = 1

    print(sDict)
    print(tDict)

    return sDict == tDict


if __name__ == '__main__':
    print(isAnagram('android', 'andrpod'))
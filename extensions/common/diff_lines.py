import re
import difflib
from typing import Tuple, List

from labml import monit

d = difflib.Differ()


def compare_line(l1, l2):
    l1, s1 = l1
    l2, s2 = l2

    if len(l1) == 0 and len(l2) == 0:
        return 0.0

    s = len(s1 & s2) / max(len(s1), len(s2))
    if s < 0.2:
        return -1.
    elif s < 0.5:
        return s - 1.
    else:
        pass
        # return s

    d = [[0 for _ in range(len(l2) + 1)] for _ in range(len(l1) + 1)]
    for i in range(len(l1) - 1, -1, -1):
        for j in range(len(l2) - 1, -1, -1):
            d[i][j] = max(
                d[i + 1][j],
                d[i][j + 1],
                d[i + 1][j + 1] + (len(l1[i]) if l1[i] == l2[j] else 0),
            )

    s = d[0][0] / max(len(''.join(l1)), len(''.join(l2)))
    if s > 0.5:
        return s
    else:
        return s - 1.


def split_string(s):
    pattern = r'[a-zA-Z0-9]+|[^\s]'
    return re.findall(pattern, s)


def compress_line(line):
    parts = split_string(line)
    return parts, set(parts)


def get_matches(v1: str, v2: str) -> Tuple[List[List[int]], List[str]]:
    v1, v2 = v1.splitlines(keepends=True), v2.splitlines(keepends=True)

    sv1 = [compress_line(line) for line in v1]
    sv2 = [compress_line(line) for line in v2]

    diff = [[compare_line(l1, l2) for l2 in sv2] for l1 in monit.iterate(sv1)]

    dp = [[0 for _ in range(len(v2) + 1)] for _ in range(len(v1) + 1)]

    for i in range(len(v1) - 1, -1, -1):
        for j in range(len(v2) - 1, -1, -1):
            dp[i][j] = max(
                dp[i + 1][j],
                dp[i][j + 1],
                dp[i + 1][j + 1] + diff[i][j],
            )

    matches = []
    i, j = 0, 0
    while i < len(v1) and j < len(v2):
        if dp[i][j] == dp[i + 1][j + 1] + diff[i][j]:
            matches.append((i, j))
            i += 1
            j += 1
        elif dp[i][j] == dp[i + 1][j]:
            i += 1
        elif dp[i][j] == dp[i][j + 1]:
            j += 1
        else:
            raise RuntimeError()

    matches.append([len(v1), len(v2)])
    
    return matches, v2

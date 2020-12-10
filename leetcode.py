from typing import List

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    # TODO 贪心算法证明
    def isPossible(self, nums: List[int]) -> bool:
        num_count = {}
        for num in nums:
            if not num in num_count:
                num_count[num] = 1
            else:
                num_count[num] += 1
        tail = {}
        for num in nums:
            if num_count[num] == 0:
                continue
            if num-1 in tail and tail[num-1] > 0:
                if num in tail:
                    tail[num] += 1
                else:
                    tail[num] = 1
                tail[num-1] -= 1
                num_count[num] -= 1
            elif num+1 in num_count and  num_count[num+1] > 0 and num+2 in num_count and num_count[num+2] > 0:
                num_count[num] -= 1
                num_count[num+1] -= 1
                num_count[num+2] -= 1
                if num+2 in tail:
                    tail[num+2] += 1
                else:
                    tail[num+2] = 1
            else:
                return False
        return True
    
    def isStraight(self, nums: List[int]) -> bool:
        zero_count = len(list(filter(lambda num: num==0, nums)))
        nonzero_nums = list(filter(lambda num: num!=0, nums))
        if len(set(nonzero_nums)) != len(nonzero_nums):
            return False
        if max(nonzero_nums)-min(nonzero_nums) - len(nonzero_nums) + 1 > zero_count:
            return False
        return True

    # TODO
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        result = []
        for i in range(len(words)):
            for j in range(i, len(words)):
                if len(words[i]) == len(words[j]):
                    if words[i][::-1] == words[j]:
                        result.append([i, j])
                        result.append([j, i])
                elif len(words[i]) > len(words[j]):
                    pass
    
    # (62) T: 86.08% S: 61.19%
    def uniquePaths(self, m: int, n: int) -> int:
        M, m = max(m, n)-1, min(m, n)-1
        result = 1
        for i in range(m):
            result *= (M+i+1)
        for i in range(m):
            result /= i+1
        return int(result)

    # (79) T: 5.02% S: 42.15%
    def existHelper(self, board: List[List[int]], word: str, position: List[int], map: List[List[bool]]) -> bool:
        if len(word) == 0:
            return True
        else:
            down, up, right, left = False, False, False, False
            if position[0]+1 < len(board) and map[position[0]+1][position[1]] == False and board[position[0]+1][position[1]] == word[0]:
                map[position[0]+1][position[1]] = True
                down = self.existHelper(board, word[1:], [position[0]+1, position[1]], map)
                map[position[0]+1][position[1]] = False
                if down:
                    return True
            if position[0]-1 >= 0 and map[position[0]-1][position[1]] == False and board[position[0]-1][position[1]] == word[0]:
                map[position[0]-1][position[1]] = True
                up = self.existHelper(board, word[1:], [position[0]-1, position[1]], map)
                map[position[0]-1][position[1]] = False
                if up:
                    return True
            if position[1]+1 < len(board[0]) and map[position[0]][position[1]+1] == False and board[position[0]][position[1]+1] == word[0]:
                map[position[0]][position[1]+1] = True
                right = self.existHelper(board, word[1:], [position[0], position[1]+1], map)
                map[position[0]][position[1]+1] = False
                if right:
                    return True
            if position[1]-1 >= 0 and map[position[0]][position[1]-1] == False and board[position[0]][position[1]-1] == word[0]:
                map[position[0]][position[1]-1] = True
                left = self.existHelper(board, word[1:], [position[0], position[1]-1], map)
                map[position[0]][position[1]-1] = False
                if left:
                    return True
            return False
    def exist(self, board: List[List[str]], word: str) -> bool:
        starts = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]:
                    starts.append([i, j])
        for start in starts:
            # print('------------------')
            map = [[False for i in range(len(board[0]))] for j in range(len(board))]
            map[start[0]][start[1]] = True
            if self.existHelper(board, word[1:], start, map) == True:
                return True
        return False
            
    # (118) T: 32.98% S: 43.39%
    def generate(self, numRows: int) -> List[List[int]]:
        result = [[1], [1, 1]]
        if numRows == 0:
            return []
        elif numRows == 1:
            return [[1]]
        elif numRows == 2:
            return [[1], [1, 1]]
        for i in range(2, numRows):
            temp = [1]
            for i in range(1, len(result[-1])):
                temp.append(result[-1][i-1]+result[-1][i])
            temp.append(1)
            result.append(temp)
        return result

    # (621) T: 61.97% S: 43.87%
    def leastInterval(self, tasks: List[str], n: int) -> int:
        if n == 0:
            return len(tasks)
        heap = [0] * 26
        for task in tasks:
            heap[ord(task) - ord('A')] += 1
        max_num = max(heap)
        min_count = (max_num-1) * (n+1) + len(list(filter(lambda s: s==max_num, heap)))
        if len(tasks) < min_count:
            return min_count
        else:
            return len(tasks)

    # (693) T: 94.88% S: 29.85%
    def hasAlternatingBits(self, n: int) -> bool:
        flag = -1
        while n != 0:
            if flag == -1:
                flag = n % 2
            else:
                if n % 2 == flag:
                    return False
                flag = n % 2
            n = int(n / 2)
        return True

    # (721, dp) T: 5.08% S: 45.05%
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        note = [[-1]*(len(s2)+1) for i in range(len(s1)+1)]
        note[0][0] = 0
        for i in range(len(s1)+1):
            for j in range(len(s2)+1):
                if j < len(s2) and note[i][j+1] == -1:
                    note[i][j+1] = note[i][j] + ord(s2[j])
                elif j < len(s2) and note[i][j+1] != -1:
                    note[i][j+1] = min(note[i][j+1], note[i][j] + ord(s2[j]))
                if i < len(s1) and note[i+1][j] == -1:
                    note[i+1][j] = note[i][j] + ord(s1[i])
                elif i < len(s1) and note[i+1][j] != -1:
                    note[i+1][j] = min(note[i+1][j], note[i][j] + ord(s1[i]))
                if i < len(s1) and j < len(s2) and s1[i] == s2[j]:
                        note[i+1][j+1] = note[i][j]
        return note[len(s1)][len(s2)]
                
    # (757) T: 100.00% S: 100.00%
    def intersectionSizeTwo(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        result = []
        noncontain_intervals = [intervals[0]]
        for i in range(1, len(intervals)):
            if intervals[i][0] > intervals[i-1][1]:
                noncontain_intervals.append(intervals[i])
            elif intervals[i][1] <= intervals[i-1][1]:
                noncontain_intervals.pop()
                length = len(noncontain_intervals)
                for j in range(length-1, -1, -1):
                    if intervals[i][1] <= noncontain_intervals[j][1]:
                        noncontain_intervals.pop()
                    else:
                        break
                noncontain_intervals.append(intervals[i])
            else:
                noncontain_intervals.append(intervals[i])
        # print(intervals)
        # print(noncontain_intervals)
        result.append(noncontain_intervals[0][1]-1)
        result.append(noncontain_intervals[0][1])
        for i in range(1, len(noncontain_intervals)):
            if noncontain_intervals[i][0] <= result[-2]:
                continue
            elif noncontain_intervals[i][0] <= result[-1]:
                result.append(noncontain_intervals[i][1])
            else:
                result.append(noncontain_intervals[i][1]-1)
                result.append(noncontain_intervals[i][1])
            # print(result)
        return len(result)
    
    # (842) T: 21.90% T: 13.00%
    def splitIntoFibonacciHelper(self, S: str, index: int, nums: List[int]) -> List[int]: 
        if S == "":
            return nums
        else:
            if index == len(S) + 1:
                return []
            elif int(S[0:index]) == nums[-1] + nums[-2] and not (S[0]=='0' and index != 1):
                if int(S[0:index]) >= 2147483647:
                    return []
                nums.append(int(S[0:index]))
                return self.splitIntoFibonacciHelper(S[index:], 1, nums)
            else:
                return self.splitIntoFibonacciHelper(S, index+1, nums)
    def splitIntoFibonacci(self, S: str) -> List[int]: 
        for i in range(1, len(S)):
            if S[0] == '0' and i != 1:
                    continue
            if int(S[0:i]) >= 2147483647:
                break
            for j in range(i+1,len(S)):
                if S[i] == '0' and j != i+1:
                    continue
                if int(S[i:j]) >= 2147483647:
                    break
                result = self.splitIntoFibonacciHelper(S[j:], 1, [int(S[0:i]), int(S[i:j])])
                if result != []:
                    return result
        return []

    # (860) T: 86.23% S: 33.77%
    def lemonadeChange(self, bills: List[int]) -> bool:
        dollar5, dollar10, dollar20 = 0, 0, 0
        for bill in bills:
            if bill == 5:
                dollar5 += 1
            elif bill == 10:
                if dollar5 == 0:
                    return False
                else:
                    dollar10 += 1
                    dollar5 -= 1
            else:
                if dollar10 == 0:
                    if dollar5 < 3:
                        return False
                    else:
                        dollar5 -= 3
                        dollar20 += 1
                else:
                    if dollar5 == 0:
                        return False
                    else:
                        dollar5 -= 1
                        dollar10 -= 1
                        dollar20 += 1
        return True
    # (861) T: 50.00% S: 23.08%
    def matrixScore(self, A: List[List[int]]) -> int:
        for i in range(len(A)):
            if A[i][0] == 0:
                for j in range(len(A[i])):
                    A[i][j] = abs(A[i][j] - 1)
        half_col = len(A) / 2
        for j in range(len(A[0])):
            temp = 0
            for i in range(len(A)):
                if A[i][j] == 1:
                    temp += 1
            if temp <= half_col:
                for i in range(len(A)):
                    A[i][j] = abs(A[i][j] - 1)
        result = 0
        for i in range(len(A)):
            A[i] = map(lambda x: str(x), A[i])
            result += int(str(''.join(A[i])), 2)
        return result

    # (1305) T: 32.00% S: 64.14%
    def getAllElementsHelper(self, root: TreeNode) -> List[int]:
        if root.left == None and root.right == None:
            return [root.val]
        elif root.right == None:
            temp = self.getAllElementsHelper(root.left)
            temp.extend([root.val])
            return temp
        elif root.left == None:
            temp = [root.val]
            temp.extend(self.getAllElementsHelper(root.right))
            return temp
        else:
            temp = self.getAllElementsHelper(root.left)
            temp.extend([root.val])
            temp.extend(self.getAllElementsHelper(root.right))
            return temp
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        result1, result2 = [], []
        if root1 != None:
            result1 = self.getAllElementsHelper(root1)
        if root2 != None:
            result2 = self.getAllElementsHelper(root2)
        result = []
        i, j = 0, 0
        while i != len(result1) or j != len(result2):
            if i == len(result1):
                result.append(result2[j])
                j += 1
            elif j == len(result2):
                result.append(result1[i])
                i += 1
            else:
                if result1[i] < result2[j]:
                    result.append(result1[i])
                    i += 1
                else:
                    result.append(result2[j])
                    j += 1
        return result


solution = Solution()
# print(solution.isPossible([1,2,3,3,4,4,5,5]))
# print(solution.isStraight([1,2,3,4,5]))
# print(solution.palindromePairs(["abcd","dcba","lls","s","sssll"]))

# print(solution.uniquePaths(5, 1)) # 62

# print(solution.exist([["A","B","C","E"],["S","F","E","S"],["A","D","E","E"]], "ABCEFSADEESE")) # 79

# print(solution.generate(5)) # 118

# print(solution.leastInterval(["A","A","A","B","B","B"], 2)) # 621

# print(solution.hasAlternatingBits(10)) # 693

# print(solution.minimumDeleteSum(s1 = "delete", s2 = "leet")) # 721

# print(solution.intersectionSizeTwo([[33,44],[42,43],[13,37],[24,33],[24,33],[25,48],[10,47],[18,24],[29,37],[7,34]])) # 757

# print(solution.splitIntoFibonacci("539834657215398346785398346991079669377161950407626991734534318677529701785098211336528511")) # 842

# print(solution.matrixScore([[0,0,1,1],[1,0,1,0],[1,1,0,0]])) # 861

# root1 = TreeNode(1)
# root1.left = TreeNode(0)
# root1.right = TreeNode(3)

# root2 = TreeNode(2)
# root2.left = TreeNode(1)
# root2.right = TreeNode(4)

# print(solution.getAllElements(root1, root2)) # 1305
from typing import List

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

    


solution = Solution()
# print(solution.isPossible([1,2,3,3,4,4,5,5]))
# print(solution.isStraight([1,2,3,4,5]))
# print(solution.palindromePairs(["abcd","dcba","lls","s","sssll"]))

# print(solution.generate(5)) # 118

# print(solution.leastInterval(["A","A","A","B","B","B"], 2)) # 621

# print(solution.minimumDeleteSum(s1 = "delete", s2 = "leet")) # 721

# print(solution.intersectionSizeTwo([[33,44],[42,43],[13,37],[24,33],[24,33],[25,48],[10,47],[18,24],[29,37],[7,34]])) # 757

# print(solution.matrixScore([[0,0,1,1],[1,0,1,0],[1,1,0,0]])) # 861
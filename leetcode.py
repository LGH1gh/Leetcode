import math
from typing import List
from datetime import datetime


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    # (1) T: 45.67% S: 14.61%
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            nums[i] = (nums[i], i)
        nums.sort()
        i, j = 0, len(nums)-1
        while i != j:
            if nums[i][0] + nums[j][0] == target:
                return [nums[i][1], nums[j][1]]
            if nums[i][0] + nums[j][0] > target:
                j -= 1
            else:
                i += 1

    # (2) T: 87.76% S: 6.60%
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        i = l3 = ListNode(-1)
        mark = False
        while l1 != None or l2 != None:
            if l1 == None and mark:
                i.next = ListNode((l2.val + 1) % 10)
                if l2.val + 1 < 10:
                    mark = False
                i, l2 = i.next, l2.next
            elif l1 == None and not mark:
                i.next = ListNode(l2.val)
                i, l2 = i.next, l2.next
            elif l2 == None and mark:
                i.next = ListNode((l1.val + 1) % 10)
                if l1.val + 1 < 10:
                    mark = False
                i, l1 = i.next, l1.next
            elif l2 == None and not mark:
                i.next = ListNode(l1.val)
                i, l1 = i.next, l1.next
            elif mark:
                i.next = ListNode((l1.val + l2.val + 1) % 10)
                if l1.val + l2.val + 1 < 10:
                    mark = False
                i, l1, l2 = i.next, l1.next, l2.next
            else:
                i.next = ListNode((l1.val + l2.val) % 10)
                if l1.val + l2.val >= 10:
                    mark = True
                i, l1, l2 = i.next, l1.next, l2.next
        if mark:
            i.next = ListNode(1)
        return l3.next
            
    # (3) T: 67.29% S: 5.22%
    def lengthOfLongestSubstring(self, s: str) -> int:
        i = 0
        vocab = set()
        result = 0
        for j in range(len(s)):
            # print(s[j])
            if s[j] not in vocab:
                vocab.add(s[j])
                result = max(result, len(vocab))
            else:
                while s[j] in vocab:
                    vocab.remove(s[i])
                    i += 1
                vocab.add(s[j])
                result = max(result, len(vocab))
        return result

    # (5) T: 56.48% S: 41.36%
    def longestPalindrome(self, s: str) -> str:
        result = (1, s[0])
        for i in range(1, len(s)-1):
            temp = 1
            while i - temp >= 0 and i + temp < len(s):
                if s[i - temp] == s[i + temp]:
                    temp += 1
                else:
                    break
            if result[0] < (temp-1)*2+1:
                result = ((temp-1)*2+1, s[i-temp+1:i+temp])
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                temp = 1
                while i-1-temp >= 0 and i+temp < len(s):
                    if s[i-1-temp] == s[i + temp]:
                        temp += 1
                    else:
                        break
                if result[0] < (temp-1)*2+2:
                    result = ((temp-1)*2+2, s[i-temp:i+temp])
        return result[1]

    # (6) T: 79.69% S: 29.66%
    def convert(self, s: str, numRows: int) -> str:
        result = ["" for _ in range(numRows)]
        if s == "" or numRows == 1:
            return s
        index = 0
        delta = 1
        for i in range(len(s)):
            result[index] += s[i]
            index = index + delta
            if index >= len(result):
                index = index - 2
                delta = -1
            if index < 0:
                index = index + 2
                delta = 1
        return "".join(result)

    # (7) T: 74.27% S: 16.90%
    def reverse(self, x: int) -> int:
        if x == 0:
            return 0
        sign = x / abs(x)
        x = int(str(reversed(str(abs(x)))))
        if (sign == 1 and x > 2**31 - 1) or (sign == -1 and x > 2**31):
            return 0
        return sign * x
        
    # (17) T: 77.90% S: 26.16%
    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits) == 0:
            return [] 
        def letterCombinationsHelper(digits: str, digit2vocab: List[str], result: List[str], temp: str):
            if digits == "":
                result.append(temp)
                return
            for i in range(len(digit2vocab[int(digits[0])])):
                letterCombinationsHelper(digits[1:], digit2vocab, result, temp+digit2vocab[int(digits[0])][i])
        digit2vocab = ['', '', 'abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        result = []
        letterCombinationsHelper(digits, digit2vocab, result, "")
        return result

    # (22) T: 71.53% S: 13.65%
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0:
            return []
        def generateParenthesis(n: int, left: int, right: int, result: List[str], temp: str):
            if right == n:
                result.append(temp)
                return
            if left < n:
                generateParenthesis(n, left+1, right, result, temp+'(')
            if right < left:
                generateParenthesis(n, left, right+1, result, temp+')')
        result = []
        generateParenthesis(n, 0, 0, result, "")
        return result

    # (39) T: 78.52% S: 5.67%
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        if target == 0:
            return []
        def combinationSumHelper(candidates: List[int], target: int, pos: int, result: List[List[int]], temp: List[int]):
            if target == 0:
                result.append(temp)
                return
            for i in range(pos, len(candidates)):
                if candidates[i] <= target:
                    combinationSumHelper(candidates, target-candidates[i], result, temp+[candidates[i]])
        result = []
        combinationSumHelper(candidates, target, 0, result, [])
        return result

    # (40) T: 89.60% S: 6.31%
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        if target == 0:
            return []
        def combinationSum2Helper(candidates: List[int], target: int, result: List[List[int]], temp: List[int]):
            if target == 0:
                result.append(temp)
                return
            for i in range(len(candidates)):
                if i != 0 and candidates[i-1] == candidates[i]:
                    continue
                if candidates[i] > target:
                    return
                combinationSum2Helper(candidates[i+1:], target-candidates[i], result, temp+[candidates[i]])
        candidates.sort()
        result = []
        combinationSum2Helper(candidates, target, result, [])
        return result

    # (41) T: 83.32% S: 11.19%
    def firstMissingPositive(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 1
        i = 0
        while i < len(nums):
            if nums[i] <= len(nums) and nums[i] >= 1 and nums[i]-1 != i and nums[nums[i]-1] != nums[i]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
            else:
                i += 1
        for i in range(len(nums)):
            if nums[i] != i + 1:
                return i + 1
        return len(nums)+1
        
    # (46) T: 73.67% S: 27.19%
    def permute(self, nums: List[int]) -> List[List[int]]:
        def permuteHelper(nums: List[int], result: List[List[int]], temp: List[int]):
            if len(nums) == 0:
                result.append(temp)
            for i in range(len(nums)):
                permuteHelper(nums[:i]+nums[i+1:], result, temp + [nums[i]])
        result = []
        permuteHelper(nums, result, [])
        return result

    # (47) T: 77.31% S: 21.21%
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        if len(nums) == 0:
            return []
        nums.sort()
        def permuteUniqueHelper(nums: List[int], result: List[List[int]], temp: List[int]):
            if len(nums) == 0:
                result.append(temp)
                return
            for i in range(len(nums)):
                if i != 0 and nums[i] == nums[i-1]:
                    continue
                permuteUniqueHelper(nums[:i]+nums[i+1:], result, temp+[nums[i]])
        result = []
        permuteUniqueHelper(nums, result, [])
        return result

    # (48) T: 56.92% S: 9.29%
    def rotate(self, matrix: List[List[int]]) -> None:
        for i in range(math.floor(len(matrix)/2)):
            for j in range(len(matrix)):
                matrix[i][j], matrix[len(matrix)-1-i][j] = matrix[len(matrix)-1-i][j], matrix[i][j]
        for i in range(len(matrix)):
            for j in range(i+1, len(matrix)):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # (49) T: 82.75% S: 20.70%
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        results = {}
        for s in strs:
            if ''.join(sorted(s)) not in results:
                results[''.join(sorted(s))] = [s]
            else:
                results[''.join(sorted(s))].append(s)
        results_list = []
        for result in results.values():
            results_list.append(result)
        return results_list
            
    # (62) T: 86.08% S: 61.19%
    def uniquePaths(self, m: int, n: int) -> int:
        M, m = max(m, n)-1, min(m, n)-1
        result = 1
        for i in range(m):
            result *= (M+i+1)
        for i in range(m):
            result /= i+1
        return int(result)

    # (66) T: 56.67% S: 19.44%
    def plusOne(self, digits: List[int]) -> List[int]:
        mark = False
        for i in range(len(digits)-1, -1, -1):
            if digits[i] != 9:
                digits[i] += 1
                mark = True
                break
            else:
                digits[i] = 0
        if mark == False:
            digits = [1] + digits
        return digits

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

    # (86) T: 12.87% S: 19.05%
    def partition(self, head: ListNode, x: int) -> ListNode:
        left_head = left_tail = ListNode(-1)
        right_head = right_tail = ListNode(-1)
        while head != None:
            if head.val < x:
                left_tail.next = head
                left_tail = left_tail.next
            else:
                right_tail.next = head
                right_tail = right_tail.next
            head = head.next
        if right_head != right_tail:
            left_tail.next = right_head.next
            left_tail = right_tail
        left_tail.next = None
        return left_head.next
        
    # (100) T: 7.89% S: 5.09%
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if p == None and q == None:
            return True
        if (p == None and q != None) or (p != None and q == None):
            return False
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    # (103) T: 47.81% S: 5.03%
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        result_list = []
        mark = False
        node_list = [root]
        while len(node_list) > 0:
            temp_result_list = []
            temp_node_list = []
            for i in range(len(node_list)-1, -1, -1):
                if node_list[i] == None:
                    continue
                else:
                    if mark == False:
                        temp_result_list.append(node_list[i].val)
                        temp_node_list.append(node_list[i].left)
                        temp_node_list.append(node_list[i].right)
                    else:
                        temp_result_list.append(node_list[i].val)
                        temp_node_list.append(node_list[i].right)
                        temp_node_list.append(node_list[i].left)
            result_list.append(temp_result_list)
            node_list = temp_node_list
            mark = not mark
        result_list.pop()
        return result_list   

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
    
    # (123) T: 13.93% S: 8.76%
    def maxProfit3(self, prices: List[int]) -> int:
        # length -> 交易日, 3 -> 交易次数, 2 -> 是否买入(0:未买入, 1买入)
        dp = [[[0 for _ in range(2)] for _ in range(3)] for _ in range(len(prices))]
        dp[0][1][1] = -prices[0]
        dp[0][0][1] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][2][0] = dp[i-1][2][0]
            dp[i][1][0] = max(dp[i-1][1][1]+prices[i], dp[i-1][1][0])
            dp[i][1][1] = max(dp[i-1][2][0]-prices[i], dp[i-1][1][1])
            dp[i][0][0] = max(dp[i-1][0][1]+prices[i], dp[i-1][0][0])
            dp[i][0][1] = max(dp[i-1][1][0]-prices[i], dp[i-1][0][1])

        return dp[-1][0][0]

    # (141) T: 14.27% S: 5.10%
    def hasCycle(self, head: ListNode) -> bool:
        if head == None:
            return False
        walker = head
        runner = head.next
        while walker != runner:
            walker = walker.next
            if runner == None:
                return False
            runner = runner.next
            if runner == None:
                return False
            runner = runner.next
        return True

    # (142) T: 73.98% S: 5.09%
    def detectCycle(self, head: ListNode) -> ListNode:
        if head == None:
            return None
        walker = head.next
        runner = head.next
        if runner == None:
            return None
        else:
            runner = runner.next
        while walker != runner:
            walker = walker.next
            if runner == None:
                return None
            runner = runner.next
            if runner == None:
                return None
            runner = runner.next
        marker = head
        while marker != walker:
            marker = marker.next
            walker = walker.next
        return marker

    # (189) T: 91.27% S: 18.87%
    def rotate(self, nums: List[int], k: int) -> None:
        nums[:] = nums[-k % len(nums):] + nums[:-k % len(nums)]

    # (213) T: 58.34% S: 14.25%
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        elif len(nums) == 0:
            return 0
        dp1 = [[0 for _ in range(len(nums)-1)] for _ in range(2)]
        dp2 = [[0 for _ in range(len(nums)-1)] for _ in range(2)]
        # print(dp1)
        dp1[0][0], dp1[1][0] = 0, nums[0]
        for i in range(1, len(nums)-1):
            dp1[0][i] = max(dp1[0][i-1], dp1[1][i-1])
            dp1[1][i] = dp1[0][i-1] + nums[i]
        dp2[0][0], dp2[1][0] = 0, nums[1]
        for i in range(2, len(nums)):
            dp2[0][i-1] = max(dp2[0][i-2], dp2[1][i-2])
            dp2[1][i-1] = dp2[0][i-2] + nums[i]
        return max(dp1[0][-1], dp1[1][-1], dp2[0][-1], dp2[1][-1])

    # (217) T: 62.53% S: 85.26%
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums  = sorted(nums)
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                return True
        return False

    # (221) T: 5.08% S: 5.22%
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        result_list = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == '1':
                    result_list.append([i, j])
        if len(result_list) == 0:
            return 0
        edge = 1
        while True:
            temp = []
            for i in result_list:
                mark = True
                for j in range(edge+1):
                    for k in range(edge+1):
                        if i[0]+j >= len(matrix) or i[1]+k >= len(matrix[0]) or matrix[i[0]+j][i[1]+k] != '1':
                            # print('(i, j)={}'.format(i))
                            mark = False
                # print('(i, j)={}, mark={}'.format(i, mark))
                if mark:
                    temp.append(i)
            if len(temp) == 0:
                return edge * edge
            result_list = temp
            edge += 1

    # (228) T: 90.75% S: 6.18%
    def summaryRanges(self, nums: List[int]) -> List[str]:
        if len(nums) == 0:
            return []
        result = []
        temp = begin = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == temp + 1:
                temp += 1
            elif temp == begin:
                result.append(str(begin))
                temp = begin = nums[i]
            else:
                result.append(f'{begin}->{temp}')
                temp = begin = nums[i]
        if temp == begin:
            result.append(f'{temp}')
        else:
            result.append(f'{begin}->{temp}')
        return result

    # (239) T: 15.71% S: 5.05%
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # print(nums)
        if len(nums) <= k:
            return [max(nums)]
        if k == 1:
            return nums
        stack = [(nums[0], 0)]
        for i in range(1, k):
            while len(stack) > 0 and stack[-1][0] < nums[i]:
                stack.pop()
            stack.append((nums[i], i))
        result = [stack[0][0]]
        shift = 0
        for i in range(k, len(nums)):
            # print(stack)
            while len(stack) > 0 and i-stack[-1][1] < k and stack[-1][0] < nums[i]:
                stack.pop()
            stack.append((nums[i], i))
            for j in range(shift, len(stack)):
                if i - stack[j][1] < k:
                    shift = j
                    # print(shift)
                    result.append(stack[shift][0])
                    break
        return result

    # (290) T: 14.16% S: 5.37%
    def wordPattern(self, pattern: str, s: str) -> bool:
        strList = s.split(' ')
        if len(strList) != len(pattern):
            return False
        patternDict = {}
        strDict = {}
        for i in range(len(strList)):
            if strList[i] not in patternDict and pattern[i] not in strDict:
                patternDict[strList[i]] = pattern[i]
                strDict[pattern[i]] = strList[i]
            else:
                if strList[i] not in patternDict or patternDict[strList[i]] != pattern[i]:
                    return False
                if pattern[i] not in strDict or strDict[pattern[i]] != strList[i]:
                    return False
        return True
        
    # (376) T: 32.00% S: 5.16%
    def wiggleMaxLength(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        refine_nums = [nums[0]]
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                continue
            refine_nums.append(nums[i])
        if len(refine_nums) == 1:
            return 1
        result_nums = [refine_nums[0]]
        for i in range(1, len(refine_nums)-1):
            if (refine_nums[i]-refine_nums[i-1]) * (refine_nums[i+1]-refine_nums[i]) < 0:
                result_nums.append(refine_nums[i])
        if len(result_nums) == 1:
            return 2
        if (result_nums[-1]-result_nums[-2]) * (refine_nums[-1]-result_nums[-1]) < 0:
            result_nums.append(refine_nums[-1])
        return len(result_nums)

    # (387) T: 51.34% S: 5.22%
    def firstUniqChar(self, s: str) -> int:
        str_dict = {}
        for i in s:
            if i in str_dict:
                str_dict[i] += 1
            else:
                str_dict[i] = 1
        for i in range(len(s)):
            if str_dict[s[i]] == 1:
                return i
        return -1

    # (389) T: 15.10% S: 5.13%
    def findTheDifference(self, s: str, t: str) -> str:
        strDict = {}
        for i in range(len(s)):
            if s[i] not in strDict:
                strDict[s[i]] = 1
            else:
                strDict[s[i]] += 1
        for i in range(len(t)):
            if t[i] not in strDict:
                return t[i]
            else:
                strDict[t[i]] += 1
        for i in strDict.keys():
            if strDict[i] % 2 == 1:
                return i

    # (394) T: 79.70% S: 5.13%
    def decodeString(self, s: str) -> str:
        while True:
            temp_str = ""
            num_str, num = "", 0
            mark = 0
            finish = True
            inner_str = ""
            for i in range(len(s)):
                print(inner_str)
                if mark == 0 and s[i].isalpha():
                    temp_str = temp_str + s[i]
                elif mark == 0 and s[i].isdigit():
                    num_str = num_str + s[i]
                elif mark == 0 and s[i] == '[':
                    finish = False
                    num = int(num_str)
                    mark += 1
                elif mark != 0 and s[i] == '[':
                    inner_str += '['
                    mark += 1
                elif mark != 1 and s[i] == ']':
                    inner_str += ']'
                    mark -= 1
                elif mark == 1 and s[i] == ']':
                    mark -= 1
                    temp_str += inner_str * num
                    inner_str = ""
                    num_str, num = "", 0
                else:
                    inner_str += s[i]
            s = temp_str
            if finish:
                break
        return s

    # (399) T: 52.57% S: 11.03%
    def calcEquationHelper(self, variable_dict: dict, begin: str, end: str) -> float:
        if end not in variable_dict or begin not in variable_dict:
            return -1
        value_dict = {}
        for key in variable_dict.keys():
            value_dict[key] = -1
        value_dict[begin] = 1
        queue = [begin]
        while value_dict[end] == -1 and len(queue) != 0:
            print(f'{value_dict}, {queue}')
            temp = []
            for i in range(len(queue)):
                for j in range(len(variable_dict[queue[i]])):
                    if value_dict[variable_dict[queue[i]][j][0]] == -1:
                        value_dict[variable_dict[queue[i]][j][0]] = value_dict[queue[i]] * variable_dict[queue[i]][j][1]
                        temp.append(variable_dict[queue[i]][j][0])
            queue = temp
            print(f'{value_dict}, {queue}')
        return value_dict[end]    
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        variable_dict = {}
        for i in range(len(equations)):
            if equations[i][0] in variable_dict:
                variable_dict[equations[i][0]].append((equations[i][1], values[i]))
            else:
                variable_dict[equations[i][0]] = [(equations[i][1], values[i])]
            if equations[i][1] in variable_dict:
                variable_dict[equations[i][1]].append((equations[i][0], 1/values[i]))
            else:
                variable_dict[equations[i][1]] = [(equations[i][0], 1/values[i])]
        print(variable_dict)
        result = []
        for query in queries:
            begin, end = query[0], query[1]
            result.append(self.calcEquationHelper(variable_dict, begin, end))
        return result

    # (404) T: 5.40% S: 10.08%
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        if root == None:
            return 0
        if root.left != None and root.left.left == None and root.left.right == None:
            return root.left.val + self.sumOfLeftLeaves(root.right)
        return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)

    # (435) T: 38.24% S: 10.70%
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        result = 0
        for i in range(1, len(intervals)):
            if intervals[i][0] >= intervals[i-1][1]:
                continue
            elif intervals[i][1] > intervals[i-1][1]:
                intervals[i][1] = intervals[i-1][1]
                result += 1
            else:
                result += 1
        return result

    # (473) T: 96.66% S: 24.07%
    def makesquareHelper(self, nums: List[int], vis: List[bool], pos: int, sum: int) -> bool:
        # print(sum)
        if sum == 0:
            return True
        for i in range(pos, len(nums)):
            if not vis[i] and nums[i] <= sum:
                vis[i] = True
                if self.makesquareHelper(nums, vis, i, sum-nums[i]):
                    return True
                vis[i] = False
        return False
    def makesquare(self, nums: List[int]) -> bool:
        if len(nums) < 4:
            return False
        total_edge = sum(nums)
        if total_edge % 4 != 0:
            return False
        edge = total_edge / 4
        nums.sort(reverse=True)
        vis = [False for _ in range(len(nums))]
        # print(nums)
        for i in range(4):
            print(vis)
            if not self.makesquareHelper(nums, vis, 0, int(edge)):
                return False
            # print('------------------------------------')
        return True

    # (507) T: 75.00% S: 6.27%
    def checkPerfectNumber(self, num: int) -> bool:
        sum_num = 0
        length = math.ceil(math.sqrt(num))
        for i in range(1, length):
            if num % i == 0:
                sum_num += (i + num//i)
        if sum_num == num * 2:
            return True
        return False

    # (509) T: 44.72 S: 5.67%
    def fib(self, n: int) -> int:
        n0, n1 = 0, 1
        if n == 0:
            return n0
        if n == 1:
            return n1
        for i in range(1, n):
            n0, n1 = n1, n0+n1
        return n1

    # (547) T: 57.07% S: 9.77%
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        result = 0
        visit_list = [False for _ in range(len(isConnected))]
        for i in range(len(visit_list)):
            print(visit_list)
            if visit_list[i] == True:
                continue
            visit_list[i] = True
            queue = [i]
            while len(queue) != 0:
                print(queue)
                temp = []
                for j in range(len(queue)):
                    for k in range(len(isConnected)):
                        if isConnected[queue[j]][k] == 1 and visit_list[k] == False:
                            visit_list[k] = True
                            temp.append(k)
                queue = temp
            result += 1
        return result

    # (605) T: 63.02% S: 5.01%
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        str_flowerbed = [str(x) for x in flowerbed]
        earth = ''.join(str_flowerbed)
        if earth[0] == '0':
            earth = '0' + earth
        if earth[-1] == '0':
            earth = earth + '0'
        earth = earth.split('1')
        def not_empty(s):
            return s and s.strip()
        earth = list(filter(not_empty, earth))
        result = 0
        for i in earth:
            result += int((len(i)-1)/2)
        return result >= n

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

    # (649) T: 75.00% S: 41.10%
    def predictPartyVictory(self, senate: str) -> str:
        mark = [0 for i in range(len(senate))]
        radiant, dire, radiant_ban, dire_ban = 0, 0, 0, 0
        for i in senate:
            if i == 'R':
                radiant += 1
            else:
                dire += 1
        while radiant != 0 and dire != 0:
            for i in range(len(senate)):
                if radiant == 0 or dire == 0:
                    break
                if mark[i] == 1:
                    continue
                if senate[i] == 'R' and radiant_ban != 0:
                    mark[i] = 1
                    radiant_ban -= 1
                elif senate[i] == 'R' and radiant_ban == 0:
                    dire_ban += 1
                    dire -= 1
                elif senate[i] == 'D' and dire_ban != 0:
                    mark[i] = 1
                    dire_ban -= 1
                elif senate[i] == 'D' and dire_ban == 0:
                    radiant_ban += 1
                    radiant -= 1
        if radiant == 0:
            return "Dire"
        else:
            return "Radiant"

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

    # (697) T: 46.77% S: 5.11%
    def findShortestSubArray(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 1
        nums_dict = {}
        for i in range(len(nums)):
            if nums[i] not in nums_dict:
                nums_dict[nums[i]] = [i]
            else:
                nums_dict[nums[i]].append(i)
        results = []
        for value in nums_dict.values():
            results += [len(value), max(value)-min(value)+1]
        if len(results) == 1:
            return results[0][1]
        results.sort(reverse=True)
        for i in range(1, len(results)):
            if results[i][0] != results[i-1][0]:
                return results[i-1][1]
        return results[-1][1]

    # (714) T: 68.62% S: 38.40%
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        if n < 2:
            return 0
        dp1 = [0 for _ in range(n)]
        dp2 = [0 for _ in range(n)]
        dp1[0] = -prices[0]
        for i in range(1,n):
            dp1[i] = max(dp1[i-1], dp2[i-1] - prices[i])
            dp2[i] = max(dp2[i-1], dp1[i-1] + prices[i] - fee)
        return dp2[n-1]

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

    # (738) T: 40.91% S: 5.70%          
    def monotoneIncreasingDigits(self, N: int) -> int:
        rawNum = str(N)
        mark = False
        for i in range(len(rawNum)):
            if rawNum[i] == '0':
                rawNum = rawNum[:i] + '0'*(len(rawNum)-i)
                mark = True
                break
        if mark:
            N = int(rawNum)-1

        strNum = str(N)
        for i in range(len(strNum)-1, 0, -1):
            if strNum[i] < strNum[i-1]:
                strNum = str(int(strNum[:i])-1) + '9'*(len(strNum)-i)
        return int(strNum)

    # (746) T: 97.30% S: 5.15%
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        cost.append(0)
        total_cost = [0] * len(cost)
        total_cost[0], total_cost[1] = cost[0], cost[1]
        for i in range(2, len(cost)):
            total_cost[i] = min(cost[i]+total_cost[i-1], cost[i]+total_cost[i-2])
        return total_cost[-1]

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
    
    # (781) T: 19.63% S: 5.55%
    def numRabbits(self, answers: List[int]) -> int:
        if len(answers) == 0:
            return 0
        answers.sort()
        result = 0
        temp_num = answers[0]
        temp_count = 1
        for i in range(1, len(answers)):
            print(f'{temp_num}, {temp_count}')
            if temp_count == temp_num + 1:
                result += (temp_num + 1)
                temp_count = 1
                temp_num = answers[i]
            elif temp_num != answers[i]:
                result += (temp_num + 1)
                temp_count = 1
                temp_num = answers[i]
            else:
                temp_count += 1
        return result + temp_num + 1
            
    # (830) T: 54.27% S: 13.68%
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        result = []
        count = 1
        temp = s[0]
        for i in range(1, len(s)):
            if s[i] == temp:
                count += 1
            elif count >= 3:
                result.append([i-count, i-1])
                count = 1
                temp = s[i]
            else:
                count = 1
                temp = s[i]
        if count >= 3:
            result.append([len(s)-count, len(s)-1])
        return result

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

    # (1283) T: 18.55% S: 42.63%
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        min_num, max_num = 1, max(nums)
        mid_num = int((min_num + max_num) / 2)
        while max_num - min_num >= 2:
            print(f'{min_num}, {mid_num}, {max_num}')
            temp = sum(map(lambda x: math.ceil(x/mid_num), nums))
            print(temp)
            if temp > threshold:
                min_num = mid_num
                mid_num = int((min_num + max_num) / 2)
            else:
                max_num = mid_num
                mid_num = int((min_num + max_num) / 2)
        if sum(map(lambda x: math.ceil(x/min_num), nums)) <= threshold:
            return min_num
        return max_num

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

    # (1423) T: 70.38% S: 6.02%
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        score1 = [0, cardPoints[0]]
        score2 = [0, cardPoints[-1]]
        for i in range(1, k):
            score1.append(score1[-1]+cardPoints[i])
            score2.append(score2[-1]+cardPoints[-i-1])
        # print(score1)
        # print(score2)
        result = 0
        for i in range(0, len(score1)):
            if result < score1[i] + score2[k-i]:
                result = score1[i] + score2[k-i]
        return result

    # (1536) T: 96.55% S: 15.79%
    def minSwaps(self, grid: List[List[int]]) -> int:
        counts = []
        for row in grid:
            count = 0
            for i in range(len(row)-1, -1, -1):
                if row[i] == 0:
                    count += 1
                else:
                    break
            counts.append(count)
        result = 0
        for i in range(len(counts)):
            if counts[i] >= len(counts) - i - 1:
                continue
            for j in range(i+1, len(counts)):
                if counts[j] >= len(counts) - i - 1:
                    result += (j - i)
                    for k in range(j, i, -1):
                        counts[k], counts[k-1] = counts[k-1], counts[k]
                    break
            if counts[i] < len(counts) - i - 1:
                return -1
        return result

    # (1539) T: 85.31% S: 46.46%
    def findKthPositive(self, arr: List[int], k: int) -> int:
        result = len(arr) + k
        for i in range(len(arr)-1, -1, -1):
            if arr[i] >= result:
                result -= 1
            else:
                break
        return result

    # (1626) T: 93.38% S: 6.60%
    def bestTeamScore(self, scores: List[int], ages: List[int]) -> int:
        people = list(zip(ages, scores))
        people.sort()
        dp = [lambda x: 0 for _ in range(len(people))]
        for i in range(len(people)):
            dp[i] = people[i][1]
            for j in range(i-1, -1, -1):
                if people[i][1] >= people[j][1] and dp[j]+people[i][1] > dp[i]:
                    dp[i] = people[i][1] + dp[j]
        # print(dp)
        return max(dp)

        

# (355) T: 80.57% S: 5.38%
class Twitter:
    def __init__(self):
        self.userList = {}
        self.tweetList = {}
        
    def postTweet(self, userId: int, tweetId: int) -> None:
        if userId not in self.tweetList:
            self.tweetList[userId] = [[tweetId, datetime.now()]]
        else:
            self.tweetList[userId].append([tweetId, datetime.now()])
        

    def getNewsFeed(self, userId: int) -> List[int]:
        tweets = []
        if userId not in self.userList:
            self.userList[userId] = set()
        if userId not in self.tweetList:
            self.tweetList[userId] = []
        for follower in self.userList[userId]:
            tweets.extend(self.tweetList[follower])
        tweets.extend(self.tweetList[userId])
        tweets = sorted(tweets, key=(lambda x: x[1]), reverse=True)
        result = []
        for tweet in tweets:
            result.append(tweet[0])
        return result[0:10]

    def follow(self, followerId: int, followeeId: int) -> None:
        if followeeId == followerId:
            return
        if followerId not in self.tweetList:
            self.tweetList[followerId] = []
        if followeeId not in self.tweetList:
            self.tweetList[followeeId] = []
        if followerId not in self.userList:
            self.userList[followerId] = set([followeeId])
        else:
            self.userList[followerId].add(followeeId)
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        """
        

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId == followerId:
            return
        if followerId not in self.userList:
            self.userList[followerId] = set()
            return
        if followeeId not in self.userList[followerId]:
            return
        self.userList[followerId].remove(followeeId)
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        """
        


solution = Solution()
# print(solution.lengthOfLongestSubstring("abcabcbb")) # 3
print(solution.firstMissingPositive([3,4,-1,1])) # 41
# print(solution.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])) # 49
# print(solution.uniquePaths(5, 1)) # 62
# print(solution.exist([["A","B","C","E"],["S","F","E","S"],["A","D","E","E"]], "ABCEFSADEESE")) # 79
# print(solution.generate(5)) # 118
# print(solution.maxProfit3([1,2,3,4,5]))
# print(solution.rob([0])) # 213
# print(solution.maximalSquare(matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]])) # 221
# print(solution.maxSlidingWindow([1,3,1,2,0,5], 3)) # 239
# print(solution.wordPattern("abba","dog cat cat fish")) # 290
# print(solution.wiggleMaxLength([1,2,3,4,5,6,7,8,9])) # 376
# print(solution.findTheDifference(s = "a", t = "aa")) # 389
# print(solution.decodeString("abc3[cd]xyz")) # 394
# print(solution.calcEquation([["x1","x2"],["x2","x3"],["x1","x4"],["x2","x5"]],[3.0,0.5,3.4,5.6],[["x4","x3"]])) # 399
# print(solution.eraseOverlapIntervals([ [1,2], [1,2], [1,2] ])) # 435
# print(solution.makesquare([10,6,5,5,5,3,3,3,2,2,2,2])) # 473
# print(solution.findCircleNum([[1,0,0,1],[0,1,1,0],[0,1,1,1],[1,0,1,1]])) # 547
# print(solution.canPlaceFlowers(flowerbed = [1,0,0,0,1], n = 2)) # 605
# print(solution.leastInterval(["A","A","A","B","B","B"], 2)) # 621
# print(solution.predictPartyVictory("RD")) # 649
# print(solution.hasAlternatingBits(10)) # 693
# print(solution.maxProfit(prices = [1, 3, 2, 8, 4, 9], fee = 2)) # 714
# print(solution.minimumDeleteSum(s1 = "delete", s2 = "leet")) # 721
# print(solution.monotoneIncreasingDigits(989998)) # 738
# print(solution.intersectionSizeTwo([[33,44],[42,43],[13,37],[24,33],[24,33],[25,48],[10,47],[18,24],[29,37],[7,34]])) # 757
# print(solution.minCostClimbingStairs(cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1])) #746
# print(solution.numRabbits([1, 1, 2])) # 781
# print(solution.splitIntoFibonacci("539834657215398346785398346991079669377161950407626991734534318677529701785098211336528511")) # 842
# print(solution.matrixScore([[0,0,1,1],[1,0,1,0],[1,1,0,0]])) # 861
# print(solution.maxScore(cardPoints = [1,79,80,1,1,1,200,1], k = 3)) # 1423
# print(solution.minSwaps(grid = [[0,0,1],[1,1,0],[1,0,0]])) # 1536
# print(solution.findKthPositive(arr = [1,2,3,4], k = 2)) # 1539
# print(solution.bestTeamScore([4,5,6,5], [2,1,2,1])) # 1626


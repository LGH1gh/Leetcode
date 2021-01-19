from typing import List
import math

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
    # (10-I) T: 92.74% S: 20.30%
    def fib(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        a, b = 0, 1
        for i in range(2, n+1):
            a, b = b, a+b
        return b % (10**9+7)
    # (10-II) T: 92.79% S: 12.91%
    def numWays(self, n: int) -> int:
        if n == 0 or n == 1:
            return 1
        a, b = 1, 1
        for i in range(2, n+1):
            a, b = b, a+b
        return b % (10**9+7)

    # (47) T: 97.59% S: 6.70%
    def maxValue(self, grid: List[List[int]]) -> int:
        value = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        value[0][0] = grid[0][0]
        for i in range(1, len(grid)):
            value[i][0] = value[i-1][0] + grid[i][0]
        for j in range(1, len(grid[0])):
            value[0][j] = value[0][j-1] + grid[0][j]
        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                value[i][j] = max(value[i-1][j], value[i][j-1]) + grid[i][j]
        return value[-1][-1]

    # (52) T: 44.39% S: 18.27%
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if headA == None or headB == None:
            return None
        headA_length, A_ptr = 1, headA
        headB_length, B_ptr = 1, headB
        while A_ptr.next != None:
            A_ptr = A_ptr.next
            headA_length += 1
        while B_ptr.next != None:
            B_ptr = B_ptr.next
            headB_length += 1
        if A_ptr != B_ptr:
            return None
        A_ptr, B_ptr = headA, headB
        if headA_length > headB_length:
            for _ in range(headA_length-headB_length):
                A_ptr = A_ptr.next
        elif headA_length < headB_length:
            for _ in range(headB_length-headA_length):
                B_ptr = B_ptr.next
        while A_ptr != B_ptr:
            A_ptr, B_ptr = A_ptr.next, B_ptr.next
        return A_ptr.val

    # (55) T: 19.63% S: 72.71%
    def isBalancedHeight(self, root:TreeNode) -> int:
        if root.left == None and root.right == None:
            return 1
        elif root.left == None:
            return 1 + self.isBalancedHeight(root.right)
        elif root.right == None:
            return 1 + self.isBalancedHeight(root.left)
        else:
            return 1 + max(self.isBalancedHeight(root.right), self.isBalancedHeight(root.left))
    def isBalanced(self, root: TreeNode) -> bool:
        if root == None:
            return True
        if root.left == None and root.right == None:
            return True
        elif root.left == None:
            return self.isBalancedHeight(root.right) < 2
        elif root.right == None:
            return self.isBalancedHeight(root.left) < 2
        else:
            return abs(self.isBalancedHeight(root.left)-self.isBalancedHeight(root.right)) < 2 and self.isBalanced(root.left) and self.isBalanced(root.right)

    # (58) T: 40.94% S: 17.03%
    def reverseLeftWords(self, s: str, n: int) -> str:
        s = ''.join(reversed(s))
        s = ''.join(reversed(s[:-n])) + ''.join(reversed(s[-n:]))
        return s

    # (64) T: 36.67% S: 12.08%
    def sumNums(self, n: int) -> int:
        return n > 0 and self.sumNums(n-1) + n


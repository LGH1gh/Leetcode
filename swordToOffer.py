class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:

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
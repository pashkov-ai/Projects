"""
101. Symmetric Tree
Easy

Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3



But the following [1,2,2,null,3,null,3] is not:

    1
   / \
  2   2
   \   \
   3    3



Note:
Bonus points if you could solve it both recursively and iteratively.
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class RecursiveSolution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if root is None:
            return True
        return self.isSymmetricSubtrees(root.left, root.right)

    def isSymmetricSubtrees(self, p: TreeNode, q: TreeNode) -> bool:
        p_none, q_none = p is None, q is None
        if p_none and q_none:
            return True
        if p_none or q_none:
            return False
        if p.val != q.val:
            return False
        return self.isSymmetricSubtrees(p.left, q.right) and self.isSymmetricSubtrees(p.right, q.left)


from collections import deque


class IterativeSolution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if root is None:
            return True
        deq = deque()
        deq.append(root.left)
        deq.appendleft(root.right)
        while len(deq) > 0:
            if not self.isSymmetricSubtrees(deq):
                return False
        return True

    def isSymmetricSubtrees(self, deq: deque) -> bool:
        p, q = deq.pop(), deq.popleft()
        p_none, q_none = p is None, q is None
        if p_none and q_none:
            return True
        if p_none or q_none:
            return False
        deq.append(p.left)
        deq.appendleft(q.right)
        deq.append(p.right)
        deq.appendleft(q.left)
        return p.val == q.val

class Tree(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    def inOrder(self, root):   #中序
        if not root:
            return []

        left = self.inOrder(root.left)
        right = self.inOrder(root.right)

        return left + [root.val] + right

    def isValidBST(self, root):
        orderlist = self.inOrder(root)
        for i in range(1, len(orderlist) - 1):
            if orderlist[i] <= orderlist[i-1]:
                return False

        return True


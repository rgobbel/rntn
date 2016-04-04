from random import randint

class Queue:

    def __init__(self):
        self.l = []

    def push(self, val):
        self.l.insert(0, val)

    def pop(self):
        return self.l.pop()

    def len(self):
        return len(self.l)


class BinaryTree:

    def __init__(self, val=None):
        self._left = None
        self._right = None
        self._val = val

    @staticmethod
    def build(vals):
        root = BinaryTree(vals[0])
        if len(vals) > 1:
            for val in vals[1:]:
                root.add(val)
        return root

    def add_left(self, val):
        return self.add(val, 'left')

    def add_right(self, val):
        return self.add(val, 'right')

    def add(self, val, lr=None):
        if lr == 'left':
            if self.left is not None:
                result = self.left.add(val)
            else:
                result = self.left = BinaryTree(val)
        elif lr == 'right':
            if self.right is not None:
                result = self.right.add(val)
            else:
                result = self.right = BinaryTree(val)
        else:
            dir = 'left' if randint(0, 1) == 0 else 'right'
            result = self.add(val, dir)
        return result

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, val):
        self._left = val

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, val):
        self._right = val

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val = val

    def hprint(self, level=0):
        if self.right is not None: self.right.hprint(level+1)
        print('  ' * level, end='')
        print(self.val)
        if self.left is not None: self.left.hprint(level+1)

    def bforder(self):
        queue = Queue()
        queue.push(self)
        result = []
        while queue.len() > 0:
            item = queue.pop()
            result.append(item.val)
            if item.left is not None:
                queue.push(item.left)
            if item.right is not None:
                queue.push(item.right)
        return result

    def postorder(self):
        result = []
        if self.left is not None:
            result += self.left.postorder()
        if self.right is not None:
            result += self.right.postorder()
        result.append(self.val)
        return result


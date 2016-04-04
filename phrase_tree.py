class PhraseNode:

    def __init__(self, idx, iparent=None, token=None):
        self._idx = idx
        self._iparent = iparent if iparent is not None and iparent >= 0 else None
        self._token = token
        self._left = None
        self._right = None
        self._parent = None
        self._sentiment = None
        self._phrase_id = None
        self._word_vec = None

    def __repr__(self):
        if self.token is not None:
            return '{}:{}'.format(self.idx, self.token)
        elif self.left is not None:
            lstr = self.left.idx
        else:
            lstr = '*'
        if self.right is not None:
            rstr = self.right.idx
        else:
            rstr = '*'
        return '{}:[{},{}]'.format(self._idx, lstr, rstr)

    def fix_pointers(self):
        if self._token is not None:
            return
        if self.left.lowest_leaf_index > self.right.lowest_leaf_index:
            self._left, self._right = self._right, self._left
        self._left.fix_pointers()
        self._right.fix_pointers()

    @property
    def is_root(self):
        return self.iparent is None or self.iparent < 0

    @property
    def is_leaf(self):
        return self.token is not None

    @property
    def idx(self):
        return self._idx

    @property
    def iparent(self):
        return self._iparent

    @iparent.setter
    def iparent(self, val):
        self._iparent = val

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, par):
        self._parent = par

    @property
    def token(self):
        return self._token

    @token.setter
    def token(self, s):
        self._token = s

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, node):
        self._left = node

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, node):
        self._right = node

    @property
    def phrase(self):
        ll = sorted(self.leaves, key=lambda x: x.idx)
        return ' '.join([x.token for x in ll])

    @property
    def phrase_id(self):
        return self._phrase_id

    @phrase_id.setter
    def phrase_id(self, val):
        self._phrase_id = val

    @property
    def word_vec(self):
        return self._word_vec

    @word_vec.setter
    def word_vec(self, val):
        self._word_vec = val

    @property
    def sentiment(self):
        return self._sentiment

    @sentiment.setter
    def sentiment(self, val):
        self._sentiment = val

    @property
    def leaves(self):
        if self.is_leaf:
            return [self]
        else:
            return self.left.leaves + self.right.leaves

    @property
    def lowest_leaf_index(self):
        if self.is_leaf:
            return self.idx
        else:
            return min(self.left.lowest_leaf_index, self.right.lowest_leaf_index)

    @property
    def n_chunks(self):
        if self.is_leaf:
            return 1
        else:
            return self._left.n_chunks + 1 + self._right.n_chunks


class SentenceTree:

    def __init__(self, idx, tokens, ppointers, sentence):
        self._idx = idx
        self._sentence = sentence
        self._nodes = [PhraseNode(i, ppointers[i]-1) for i in range(len(ppointers))]
        for i in range(len(ppointers)):
            node = self.nodes[i]
            if i < len(tokens):
                node.token = tokens[i]
            pp = ppointers[i]
            if pp != 0:
                node.parent = self.nodes[pp-1]
                if node.parent.left is None:
                    node.parent.left = node
                else:
                    node.parent.right = node
            else:
                self._root = node

    def __repr__(self):
        return '{}:{}'.format(self.idx, self.root.phrase)

    @property
    def root(self):
        return self._root

    @property
    def idx(self):
        return self._idx

    @property
    def nodes(self):
        return self._nodes

    @property
    def leaves(self):
        return [node for node in self.nodes if node.is_leaf]

    @property
    def sentence(self):
        return self._sentence

    @property
    def n_chunks(self):
        return self._root.n_chunks

    def fix_pointers(self):
        self.root.fix_pointers()

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        self.iter_index += 1
        if self.iter_index > len(self):
            raise StopIteration
        else:
            return self[self.iter_index-1]

    @classmethod
    def load_trees(cls, tree_data, sentences):
        trees = [cls(i, tokens, tree, sentences[i])
                 for (i, (tokens, tree)) in enumerate(tree_data)]
        return trees


class SentenceSet:

    def __init__(self, trees=None):
        self.idx_dict = {}
        if trees:
            for tree in trees:
                self.add(tree)

    def add(self, tree):
        self.idx_dict[tree.idx] = tree

    def __getitem__(self, idx):
        return self.idx_dict[idx]

    def __len__(self):
        return len(self.idx_dict)

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        self.iter_index += 1
        if self.iter_index > len(self):
            raise StopIteration
        else:
            return self[self.iter_index-1]

    def fix_pointers(self):
        for tree in self:
            tree.fix_pointers()

    @property
    def n_chunks(self):
        return sum(sentence.n_chunks for sentence in self)

    @property
    def phrases(self):
        return [tree.phrase for tree in
                [item for sublist in
                 [sentence.nodes for sentence in self]
                 for item in sublist]]

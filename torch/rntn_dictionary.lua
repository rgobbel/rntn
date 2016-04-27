require 'csv'

local DictItem = {}

function DictItem.new() end

function rntn.dict.
class DictItem:

    def __init__(self, phrase, phrase_id, sentiment=None):
        self._phrase = phrase
        self._phrase_id = phrase_id
        self._sentiment = sentiment

    @property
    def phrase_id(self):
        return self._phrase_id

    @property
    def phrase(self):
        return self._phrase

    @property
    def sentiment(self):
        return self._sentiment

    @sentiment.setter
    def sentiment(self, val):
        self._sentiment = val

    @property
    def sentiment_class(self):
        if self.sentiment   <= 0.2: return 0
        elif self.sentiment <= 0.4: return 1
        elif self.sentiment <= 0.6: return 2
        elif self.sentiment <= 0.8: return 3
        else:                       return 4

    @property
    def sentiment_1hot(self):
        return [1.0 if self.sentiment_class == i else 0.0 for i in range(5)]


class barsep(csv.Dialect):
    """Bar-separated files."""
    delimiter = '|'
    quoting = csv.QUOTE_NONE
    lineterminator = '\r\n'
csv.register_dialect('barsep', barsep)

class Dictionary:

    def __init__(self, items=None):
        self._idx_dict = {}
        self._str_dict = {}
        if items is not None:
            for item in items:
                self.add(item)

    def add(self, item):
        self._idx_dict[item.phrase_id] = item
        self._str_dict[item.phrase] = item
        return self

    def __getitem__(self, item):
        if type(item) is int:
            return self._idx_dict[item]
        elif type(item) is str:
            return self._str_dict[item]
        else:
            raise TypeError

    def __len__(self):
        return len(self._idx_dict)

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        self.iter_index += 1
        if self.iter_index > len(self):
            raise StopIteration
        else:
            return self[self.iter_index-1]

    def dump(self, dict_file, sentiment_file):
        with open(dict_file, 'w') as f_dict:
            with open(sentiment_file, 'w') as f_sent:
                f_sent.write('phrase ids|sentiment values\n')
                for item in self:
                    f_dict.write('{}|{}\n'.format(item.phrase, item.phrase_id))
                    if item.sentiment is not None:
                        f_sent.write('{}|{}\n'.format(item.phrase_id, item.sentiment))

    @classmethod
    def load(cls, dict_filename, sentiment_filename):
        result = cls()
        with open(sentiment_filename, 'r') as f_sentiment:
            if PY3:
                csv.register_dialect('barsep', delimiter='|', quoting=csv.QUOTE_NONE)
            fieldnames = ['phrase ids', 'sentiment values']
            reader = csv.DictReader(f_sentiment, dialect='barsep', fieldnames=fieldnames)
            _ = reader.next()
            sentiment_map = {row['phrase ids']: float(row['sentiment values']) for row in reader}
        with open(dict_filename, 'r') as f_dict:
            for line in f_dict.readlines():
                phrase, phrase_id = line.strip().split('|')
                result.add(DictItem(phrase, int(phrase_id), sentiment_map[phrase_id]))
        return result

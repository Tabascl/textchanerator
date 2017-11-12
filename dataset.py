import random
import numpy as np

SKIP = 2

class DataSet():
    def __init__(self, text):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.char_size = len(self.chars)

        self.char2id = dict((c, i) for i, c in enumerate(self.chars))
        self.id2char = dict((i, c) for i, c in enumerate(self.chars))

    def next_batch(self, batch_size, len_per_section):
        '''Return the next `batch_size` examples from this data set.'''
        max_ix = len(self.text) - batch_size - len_per_section * SKIP
        start = random.randint(0, max_ix)
        batchtext = self.text[start:start +  len_per_section + batch_size * SKIP]

        sections = []
        next_chars = []
        for i in range(0, len(batchtext) - len_per_section, SKIP):
            sections.append(batchtext[i: i + len_per_section])
            next_chars.append(batchtext[i + len_per_section])

        X = np.zeros((len(sections), len_per_section, self.char_size))
        y = np.zeros((len(sections), self.char_size))

        for i, section in enumerate(sections):
            for j, char in enumerate(section):
                X[i, j, self.char2id[char]] = 1
                y[i, self.char2id[next_chars[i]]] = 1
        
        return X, y

if __name__ == '__main__':
    text = open('cleaned_posts.txt', 'r').read()
    dataset = DataSet(text)
    for _ in range(10):
        sth = dataset.next_batch(512, 50)

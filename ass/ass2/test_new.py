from functools import partial

from nltk import FreqDist
from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.util import skipgrams
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline, padded_everygrams
from nltk.lm.preprocessing import flatten
from nltk.lm import MLE

# text = 'happy'

# bigrams_list = list(ngrams(text, n=2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
# vocab = list(pad_both_ends(text, n=2))
# print(bigrams_list)
# print(vocab)
# print(list(skipgrams(text, n=2, k=1, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')))

# training_ngrams, padded_sentences = padded_everygram_pipeline(2, text)
# for ngramlize_sent in training_ngrams:
#     print(list(ngramlize_sent))
#     print()
# print('#############')
# list(padded_sentences)


model = MLE(3)

text = ['abc', 'abcdef']

# </s><>
#

# def create_everygrams_reverse(order, text):
#     padding_fn = partial(pad_both_ends, n=order)
#
#     return (
#         (everygrams(list(reversed(list(padding_fn(sent)))), max_len=order) for sent in text),
#         flatten(map(padding_fn, text)),
#     )
#
# #
# train, vocab = create_everygrams_reverse(3, text)
# #
# model.fit(train, vocab)
# #
# print(model.counts[['f']]['e'])

# for item in train:
#     print(item)

# train, vocab = padded_everygram_pipeline(3, text)
#
# model.fit(train, vocab)
# print(model.vocab.counts)
# print(model.counts['a'])
# # print(model.counts.max())
#
# print(len(model.counts[['<s>', '<a>']]))
# print(model.score('a'))
# print(model.counts[['d']])

test = FreqDist()

for i in range(5):
    test['a'] += 0.1
    test['b'] += 1

print(test.items())

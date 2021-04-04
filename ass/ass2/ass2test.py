from functools import partial
from nltk.lm.preprocessing import flatten
from nltk import everygrams
from nltk.corpus import brown
import numpy as np
import re
import string
import copy
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE
from nltk.probability import FreqDist
from itertools import count

letter_lowercase = list(string.ascii_lowercase)

np.random.seed(0)

# training_set stores the rest word types for training
training_set = []
# test_set stores 1000 word types for testing
test_set = []

words = list(set([word.lower() for word in brown.words() if not re.search(r'[^a-zA-Z]', word)]))

np.random.shuffle(list(words))

test_set += words[0:1000]
training_set += words[1000:]


def hangman(secret_word, guesser, max_mistakes=8, verbose=True, **guesser_args):
    """
        This function plays the hangman game with the provided guesser and returns the number of incorrect guesses.

        secret_word: a string of lower-case alphabetic characters, i.e., the answer to the game
        guesser: a function which guesses the next character at each stage in the game
            The function takes a:
                mask: what is known of the word, as a string with _ denoting an unknown character
                guessed: the set of characters which already been guessed in the game
                guesser_args: additional (optional) keyword arguments, i.e., name=value
        max_mistakes: limit on length of game, in terms of number of allowed mistakes
        verbose: silent or verbose diagnostic prints
        guesser_args: keyword arguments to pass directly to the guesser function
    """
    secret_word = secret_word.lower()
    mask = ['_'] * len(secret_word)
    guessed = set()
    if verbose:
        print("Starting hangman game. Target is", ' '.join(mask), 'length', len(secret_word))

    mistakes = 0
    while mistakes < max_mistakes:
        if verbose:
            print("You have", (max_mistakes - mistakes), "attempts remaining.")
        guess = guesser(mask, guessed, **guesser_args)

        if verbose:
            print('Guess is', guess)
        if guess in guessed:
            if verbose:
                print('Already guessed this before.')
            mistakes += 1
        else:
            guessed.add(guess)
            if guess in secret_word and len(guess) == 1:
                for i, c in enumerate(secret_word):
                    if c == guess:
                        mask[i] = c
                if verbose:
                    print('Good guess:', ' '.join(mask))
            else:
                if len(guess) != 1:
                    print('Please guess with only 1 character.')
                if verbose:
                    print('Sorry, try again.')
                mistakes += 1

        if '_' not in mask:
            if verbose:
                print('Congratulations, you won.')
            return mistakes

    if verbose:
        print('Out of guesses. The word was', secret_word)
    return mistakes


def test_guesser(guesser, test=test_set):
    """
        This function takes a guesser and measures the average number of incorrect guesses made over all the words in the test_set.
    """
    total = 0
    for word in test:
        total += hangman(word, guesser, 26, False)
    return total / float(len(test))


# q6

def create_everygrams_reverse(order, text):
    padding_fn = partial(pad_both_ends, n=order)

    return (
        (everygrams(list(reversed(list(padding_fn(sent)))), max_len=order) for sent in text),
        flatten(map(padding_fn, text)),
    )


model_left_context = MLE(3)
train_left, vocab_left = padded_everygram_pipeline(3, training_set)
model_left_context.fit(train_left, vocab_left)

counter = count()


def my_amazing_ai_guesser(mask, guessed):
    if not guessed:
        output = next(counter)
        print('Word Number')
        print(output)

    letter_prob = FreqDist()

    chars = ['<s>', '<s>'] + mask + ['</s>', '</s>']

    unigram_flag_left = False

    for index in range(2, len(chars) - 2):

        if chars[index] != '_':
            continue

        # left context calculation

        if len(model_left_context.counts[[chars[index - 2], chars[index - 1]]]):  # trigram model

            for letter in letter_lowercase:
                letter_prob[letter] += model_left_context.score(letter, [chars[index - 2], chars[index - 1]])

        elif len(model_left_context.counts[[chars[index - 1]]]):  # bigram model

            for letter in letter_lowercase:
                letter_prob[letter] += model_left_context.score(letter, [chars[index - 1]])

        else:  # unigram model

            unigram_flag_left = True

    if unigram_flag_left:
        for letter in letter_lowercase:
            letter_prob[letter] += model_left_context.score(letter)

    guess_sorted = [pair[0] for pair in sorted(letter_prob.items(), key=lambda item: item[1], reverse=True)
                    if pair[0] not in guessed]

    return guess_sorted[0]


print("Start guessing!")
result = test_guesser(my_amazing_ai_guesser)
print("Testing my amazing AI guesser using every word in test set")
print("Average number of incorrect guesses: ", result)

# num = hangman('happy', my_amazing_ai_guesser, 26, False)
# print(num)

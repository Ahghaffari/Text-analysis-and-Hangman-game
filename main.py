#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Created on Mon Mar 29 11:19:02 2021
__author__: Amirhossein Ghaffari
__python_version__: 3.9.2
__discription__: This is the main file of my final project code that runs to
                 play hangman game.
"""

# Import packages
from nltk.corpus.reader.conll import ConllCorpusReader
import io
from contextlib import redirect_stdout
from nltk.text import Text
import string
import pyconll
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from tabulate import tabulate
import random
import re


def data_visualizer_preprocess(sentences_dataset, tags, words_all, tags_all, sentences_length, words_in_every_sentence,
                               elements):
    # Words count and sort the list ascending
    words_count = Counter(words_all)
    words_count_sorted = dict(sorted(words_count.items(), key=lambda item: item[1]))

    # 3.1 - 100 most occurrence words visualize
    most_occurance_words = list(words_count_sorted.keys())[-100:]
    words_occurance = list(words_count_sorted.values())[-100:]
    plt.figure(figsize=(20, 3))
    plt.bar(most_occurance_words, words_occurance, align='edge', width=0.3)
    plt.ylabel('Occurrences')
    plt.xlabel('Words')
    plt.title('Occurrences by word')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        '3_1.png',
        format='png'
    )
    plt.show()

    # 3.2 - Words less than 10 times
    occurrence = list(words_count_sorted.values())[0]
    cnt = 0
    words_less_ten = ""
    while occurrence <= 10:
        words_less_ten = words_less_ten + list(words_count_sorted.values())[cnt] \
                         * (list(words_count_sorted.keys())[cnt] + " ")
        cnt += 1
        occurrence = list(words_count_sorted.values())[cnt]
    wordcloud = WordCloud().generate(words_less_ten)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(
        '3_2.png',
        format='png'
    )
    plt.show()

    # 3.3 - POS tags distribution
    tags_count = Counter(tags_all)
    tags_count_sorted = dict(sorted(tags_count.items(), key=lambda item: item[1]))
    most_occurance_tags = list(tags_count_sorted.keys())
    most_occurance_tags = ['None' if elem is None else elem for elem in most_occurance_tags]
    tags_occurance = np.asarray(list(tags_count_sorted.values()))
    tags_occurance = tags_occurance * 100 / np.sum(tags_occurance)
    plt.figure()
    plt.bar(most_occurance_tags, tags_occurance, align='edge', width=0.3)
    plt.ylabel('Distribution(percent)')
    plt.xlabel('POS')
    plt.title('Distribution of POS')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        '3_3.png',
        format='png'
    )
    plt.show()

    # 3.4 - Distribution of sentence length
    sentence_len_count = Counter(sentences_length)
    sentence_len_count_sorted = dict(sorted(sentence_len_count.items(), key=lambda item: item[1]))
    most_occurance_len = list(sentence_len_count_sorted.keys())
    most_occurance_len = ['None' if elem is None else elem for elem in most_occurance_len]
    sentence_len_occurance = np.asarray(list(sentence_len_count_sorted.values()))
    sentence_len_occurance_edited = sentence_len_occurance * 100 / np.sum(sentence_len_occurance)
    plt.figure()
    plt.bar(most_occurance_len, sentence_len_occurance_edited, align='edge', width=0.3)
    plt.ylabel('Distribution(percent)')
    plt.xlabel('Sentence length')
    plt.title('Distribution of Sentence length')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        '3_4.png',
        format='png'
    )
    plt.show()

    # 3.5 - Corpus size; tagset size; max, min and mean sentence length
    tagset_size = len(tags_count_sorted)
    sentences_length = np.asarray(sentences_length)
    max_sentence_len = np.max(sentences_length)
    min_sentence_len = np.min(sentences_length)
    mean_sentence_len = np.mean(sentences_length)
    corpus_size = len(sentences_dataset)
    table = [["Corpus size", "Tagset size", "Max sentence length", "Min sentence length", "Mean sentence length"],
             [corpus_size, tagset_size, max_sentence_len, min_sentence_len, mean_sentence_len]
             ]
    table_out = tabulate(table, headers='firstrow')
    print(table_out)
    # Write specifications table to a text file
    f = open("3_5.txt", "w")
    f.write(table_out)
    f.close()

    # 5.1 - Data preprocessing part 1
    cnt = 0
    for i in range(corpus_size):
        if sentences_length[i] < 5 or sentences_length[i] > 20:
            del sentences_dataset[i - cnt]
            del words_in_every_sentence[i - cnt]
            del elements[i - cnt]
            del tags[i - cnt]
            cnt += 1

    # 5.2 - Data preprocessing part 2
    words_less_ten = words_less_ten.split()
    loop_cnt = len(sentences_dataset)
    cnt = 0
    for i in range(loop_cnt):
        if any(item in words_in_every_sentence[i - cnt] for item in words_less_ten):
            del sentences_dataset[i - cnt]
            del words_in_every_sentence[i - cnt]
            del elements[i - cnt]
            del tags[i - cnt]
            cnt += 1

    # 5.3 - Data preprocessing part 3
    for i in range(len(sentences_dataset)):
        sentences_dataset[i] = sentences_dataset[i].lower()

    return sentences_dataset, words_in_every_sentence, tags, elements


def sentence_guessing(sentences_dataset, words_dataset, pos, elements, words, deadline_cnt_const, hint_cnt_const):
    print("-" * 100)
    input("\nWelcome to the sentence guessing game! Your challenge is"
          "to guess the words and complete the sentence.\nReady to play?"
          "\nPlease press Enter key to start this challenging game...")

    score = 0
    # Pick a random sentence and show words locations to player
    num = random.randint(0, len(sentences_dataset) - 1)
    selected_sentence = sentences_dataset[num]
    selected_words = words_dataset[num]
    selected_pos = pos[num]
    selected_element = elements[num]
    show_sentence = sentence_hide(selected_sentence)

    while True:
        guessed_word = input("The sentence has {0} words. What`s the first one?\n{1}\n".format(len(selected_words),
                                                                                               show_sentence))
        if guessed_word.isalpha() or guessed_word == "?":
            break
        else:
            print("Please enter valid input only with alphabets.\n")

    for words_cnt in range(len(selected_words)):
        deadline_cnt = 1
        hint_cnt = 0
        while True:
            print("-" * 100)
            if guessed_word == "?" and hint_cnt < hint_cnt_const:
                hint_cnt += 1
                if hint_cnt == 1:
                    for index_elem in range(len(selected_element)):
                        if selected_element[index_elem] == selected_words[words_cnt]:
                            break

                    print("It is a {0} with {1} letters.\n".format(selected_pos[index_elem],
                                                                   len(selected_words[words_cnt])))

                elif hint_cnt == 2:
                    text = Text(words)
                    similar_words = get_similar(selected_words[words_cnt], text)
                    try:
                        print("Words in similar contexts are {0}, {1}, {2}, {3} and {4}".format(similar_words[0],
                                                                                                similar_words[1],
                                                                                                similar_words[2],
                                                                                                similar_words[3],
                                                                                                similar_words[4]))
                    except:
                        print("There is not 5 similar words in the context!")

                else:
                    print("It starts with {0} and ends with {1}".format(selected_words[words_cnt][0],
                                                                        selected_words[words_cnt][-1]))

                while True:
                    guessed_word = input("So Whats the next word?\n{0}\n".format(show_sentence))
                    if guessed_word.isalpha() or guessed_word == "?":
                        break
                    else:
                        print("Please enter valid input only with alphabets.\n")

            elif guessed_word == "?":
                while True:
                    guessed_word = input("You`ve used all {0} available hints for this word."
                                         "\nPlease try to guess it yourself:\n".format(hint_cnt_const))
                    if guessed_word.isalpha() or guessed_word == "?":
                        break
                    else:
                        print("Please enter valid input only with alphabets.\n")

            else:
                if guessed_word.lower() == selected_words[words_cnt]:
                    print("Your guess: {0}\n".format(guessed_word))

                    if hint_cnt == 0:
                        score += 30

                    elif hint_cnt == 1:
                        score += 20

                    elif hint_cnt == 2:
                        score += 15

                    elif hint_cnt == 3:
                        score += 10

                    index = show_sentence.find("_")
                    show_sentence = show_sentence.replace(show_sentence[index], selected_words[words_cnt], 1)
                    index = show_sentence.find("_")

                    if index != -1:
                        while True:
                            guessed_word = input("Great! Whats the next word?\n{0}\n".format(show_sentence))
                            if guessed_word.isalpha() or guessed_word == "?":
                                break
                            else:
                                print("Please enter valid input only with alphabets.\n")
                        break

                    else:
                        print("Congratulations! Your final score is {0}.\nYour completed statement is:\n{1}\n"
                              "See you next time :)"
                              .format(score, show_sentence))
                        break

                elif deadline_cnt < deadline_cnt_const:
                    print("Your guess: {0}\n".format(guessed_word))
                    while True:
                        guessed_word = input("Wrong!Your wrong attempts for this word: {0}\nYou have {1} left."
                                             "\nTry another word or ask for a hint with \'?\':\n"
                                             .format(deadline_cnt, deadline_cnt_const - deadline_cnt))
                        if guessed_word.isalpha() or guessed_word == "?":
                            break
                        else:
                            print("Please enter valid input only with alphabets.\n")

                    deadline_cnt += 1

                else:
                    print("Your guess: {0}\n".format(guessed_word))
                    score -= 10
                    index = show_sentence.find("_")
                    show_sentence = show_sentence.replace(show_sentence[index], selected_words[words_cnt], 1)
                    index = show_sentence.find("_")
                    if index != -1:
                        while True:
                            guessed_word = input("Wrong!It was the last Attempt! I show you the answer, Whats the next "
                                                 "word?\n{0}\n".format(show_sentence))
                            if guessed_word.isalpha() or guessed_word == "?":
                                break
                            else:
                                print("Please enter valid input only with alphabets.\n")

                    else:
                        print("Wrong! You guessed the last word wrong! I show you the answer, You finished the game "
                              "successfully, your score is {0}. Your completed statement is:\n{1}\nSee you later :)"
                              .format(score, show_sentence))
                    break


def sentence_hide(word):
    word = re.sub('[A-Za-z]', '_', word)
    # word = re.sub('\d', '_', word)
    show_sentence = ""
    last_char = "S"
    for i in range(len(word)):
        if word[i] != last_char:
            show_sentence = show_sentence + word[i]
            last_char = word[i]
    return show_sentence


def get_similar(word, text):
    with io.StringIO() as f, redirect_stdout(f):
        text.similar(word, num=5)
        result = f.getvalue().replace('\n', ' ').strip(' ').split(' ')
        if result == ['No', 'matches']:
            result = []
    return result


# Load input file
train = pyconll.load_from_file('en_ewt-ud-train.conllu')
# with open('en_ewt-ud-train.conllu', 'r') as rfile:
#     with open('en_ewt-ud-train_preproc.conllu', 'w') as wfile:
#         for line in rfile.readlines():
#             if line[0] != '#':
#                 wfile.write(line)
corpus = ConllCorpusReader('./', ['en_ewt-ud-train.conllu'], ['words', 'pos', 'tree', 'chunk', 'ne', 'srl', 'ignore'])

# Declare parameters
sentences = []
words_in_sentences = []
words = []
upos = []
upos_all = []
sentences_len = []
elements_in_sentences = []
GUESSING_DEADLINE_CNT = 5
GUESSING_HINT_CNT = 3
translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

# Read sentences and part of speech tags and append needed data to some lists
for sentence_cnt in range(len(train)):
    elements_in_sentences.append(re.findall(r"[\w']+|[.,!?;:`']+|[\b.*?\S.*?(?:\b|$)]",
                                            train._sentences[sentence_cnt].text.lower()))
    words_in_sentences.append(train._sentences[sentence_cnt].text.lower().translate(translator).split())
    sentences.append(train._sentences[sentence_cnt].text)
    words = words + train._sentences[sentence_cnt].text.lower().translate(translator).split()
    upos.append([])
    sentences_len.append(len(train._sentences[sentence_cnt]._tokens))
    for upos_cnt in range(len(train._sentences[sentence_cnt]._tokens)):
        upos[sentence_cnt].append(train._sentences[sentence_cnt]._tokens[upos_cnt].upos)
        upos_all.append(train._sentences[sentence_cnt]._tokens[upos_cnt].upos)

# Visualize data and preprocess data for game
sentences_processed, words_in_sentences_processed, pos_processed, elements_processed = data_visualizer_preprocess(
    sentences, upos, words, upos_all, sentences_len, words_in_sentences, elements_in_sentences)

# Game
sentence_guessing(sentences_processed, words_in_sentences_processed, pos_processed, elements_processed, words,
                  GUESSING_DEADLINE_CNT, GUESSING_HINT_CNT)

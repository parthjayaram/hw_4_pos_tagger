from nltk import word_tokenize
from collections import OrderedDict
import os
import json
import numpy as np
from pprint import pprint
from nltk.tokenize import sent_tokenize
import re

class viterbi_pos_tagger:

    def __init__(self, training_file, oov_flag):
        train_raw = open(os.path.join("files", training_file), 'r').readlines()
        self._train_bigram(train_raw, oov_flag)
        self.train_trigram(train_raw)

    def _train_bigram(self, train_raw, oov_flag):
        """

        :param train_raw: array of lines in training corpus with POS labeled
        :return:
        """
        # initialize likelihood and transition dictionary
        self._LIKELIHOOD = {}
        self._STATE = {}
        previous = 'sent_begin'

        sent_end = '.'
        sent_begin = '\n'
        # iterate over training corpus line by line
        for line in train_raw:
            line_array = line.split("\t")
            # determine word and pos tag from training corpus
            if len(line_array) > 1:
                line_array[1] = line_array[1].rstrip()
            if line_array[0] == sent_begin:
                POS = 'sent_begin'
                WORD = 'sent_begin'
            elif line_array[1] == sent_end:
                WORD = line_array[0]
                POS = '.'
            else:
                WORD = line_array[0]
                POS = line_array[1]
            # populate likelihood dictionary
            if not self._LIKELIHOOD.get(POS):
                self._LIKELIHOOD[POS] = {}
            if not self._LIKELIHOOD[POS].get(WORD):
                self._LIKELIHOOD[POS][WORD] = 1
            else:
                self._LIKELIHOOD[POS][WORD] += 1
            #populate state dictionary
            if not self._STATE.get(previous):
                self._STATE[previous] = {}
            if not self._STATE[previous].get(POS):
                self._STATE[previous][POS] = 1
            else:
                self._STATE[previous][POS] += 1
            previous = POS

        # get all words which appear only once in a particular tag, label them as OOV and note their tags
        oov_words_in_likelihood = []
        for POS in self._LIKELIHOOD:
            for word in self._LIKELIHOOD[POS]:
                if self._LIKELIHOOD[POS][word] == 1:
                    result_tuple = (POS, word)
                    oov_words_in_likelihood.append(result_tuple)
                    # oov_word = self.oov_identifier(word)
                    # self._LIKELIHOOD[POS][oov_word] = self._LIKELIHOOD[POS].pop(word)

        if oov_flag:
            for oov_tuple in oov_words_in_likelihood:
                pos = oov_tuple[0]
                word = oov_tuple[1]
                oov_equivalent = self.oov_identifier(word)
                print pos + " " + word + " " + oov_equivalent
                if self._LIKELIHOOD[pos].get(oov_equivalent):
                    self._LIKELIHOOD[pos][oov_equivalent] += self._LIKELIHOOD[pos].pop(word)
                else:
                    self._LIKELIHOOD[pos][oov_equivalent] = self._LIKELIHOOD[pos].pop(word)

        # convert to probability
        self._LIKELIHOOD_PROBS = self._LIKELIHOOD
        for POS in self._LIKELIHOOD_PROBS:
            pos_total = sum(self._LIKELIHOOD_PROBS[POS].itervalues())
            for word in self._LIKELIHOOD_PROBS[POS]:
                self._LIKELIHOOD_PROBS[POS][word] = self._LIKELIHOOD_PROBS[POS][word] / float(pos_total)

        # convert to probability
        self._STATE_PROBS = self._STATE
        for previous in self._STATE_PROBS:
            previous_total = sum(self._STATE_PROBS[previous].itervalues())
            for pos in self._STATE_PROBS[previous]:
                self._STATE_PROBS[previous][pos] = self._STATE_PROBS[previous][pos] / float(previous_total)

        json.dump(self._LIKELIHOOD_PROBS, file('LIKELIHOOD_PROBS_w_OOV.txt', 'w'))
        json.dump(self._STATE_PROBS, file('STATE_PROBS.txt', 'w'))

        # populate tags array to keep track of specific tags
        self._tags = []
        for previous in self._STATE_PROBS:
            # print previous
            self._tags.append(previous)
        #reshuffle tags so that sent_begin is at index 0 and sent_end is at index -1
        # print "Unmodified" + str(len(self._tags))
        temp = self._tags[0]
        self._tags[self._tags.index('sent_begin')] = temp
        self._tags[0] = 'sent_begin'

        temp = self._tags[-1]
        self._tags[self._tags.index('.')] = temp
        self._tags[-1] = '.'

        # now to initialize the transition matrix
        self._transition_matrix = np.zeros((len(self._tags), len(self._tags)))
        for i in range(0, len(self._tags)):
            for j in range(0, len(self._tags)):
                if self._STATE_PROBS[self._tags[i]].get(self._tags[j]):
                    # print "For tag " + tags[i] + " The transition probability to " + tags[j] + " is " + str(STATE_PROBS[tags[i]].get(tags[j]))
                    self._transition_matrix[i][j] = self._STATE_PROBS[self._tags[i]].get(self._tags[j])
                else:
                    # print "For tag " + tags[i] + " The transition probability to " + tags[j] + " is " + str(0.00)
                    self._transition_matrix[i][j] = 0.0


    def oov_identifier(self, word):
        if word[0].isupper():
            return 'start_cap'
        elif re.match('.*[0-9].*', word):
            return 'misc_num'
        elif re.match('[0-9]{2}', word):
            return 'two_digit_num'
        elif re.match('[0-9]{4}', word):
            return 'four_digit_num'
        elif word == word.upper():
            return 'all_cap'
        elif re.match('[0-9]{2}-[0-9]{2,4}', word):
            return 'digit_dash'
        elif re.match('([A-Z]+[0-9]+)|([0-9]+[A-Z]+)', word):
            return 'alpha_numeric'
        else:
            return 'oov'

    def train_trigram(self, train_raw):
        """

        :param train_raw:
        :return:
        """
        # initialize likelihood and transition dictionary
        self._STATE_TRIGAM = {}

        sent_end = '.'
        sent_begin = '\n'
        # iterate over training corpus line by line
        train_raw.insert(0, '\n')
        for line_index in range(2,len(train_raw)):
            line_k_array = train_raw[line_index].split("\t")
            line_u_array = train_raw[line_index - 1].split("\t")
            line_v_array = train_raw[line_index - 2].split("\t")

            # parse current word and pos
            if len(line_k_array) > 1:
                current_word = line_k_array[0]
                current_pos = line_k_array[1]
            else:
                current_word = 'sent_begin'
                current_pos = 'sent_begin'
            # parse state u
            if len(line_u_array) > 1:
                word_u = line_u_array[0]
                pos_u = line_u_array[1]
            else:
                word_u = 'sent_begin'
                pos_u = 'sent_begin'
            # parse state v
            if len(line_v_array) > 1:
                word_v = line_v_array[0]
                pos_v = line_v_array[1]
            else:
                word_v = 'sent_begin'
                pos_v = 'sent_begin'
            # populate trigram state dictionary
            if not self._STATE_TRIGAM.get(pos_v):
                self._STATE_TRIGAM[pos_v] = {}
            if not self._STATE_TRIGAM[pos_v].get(pos_u):
                self._STATE_TRIGAM[pos_v][pos_u] = {}

            if not self._STATE_TRIGAM[pos_v][pos_u].get(current_pos):
                self._STATE_TRIGAM[pos_v][pos_u][current_pos] = 1
            else:
                self._STATE_TRIGAM[pos_v][pos_u][current_pos] += 1

        # convert them to probabilities
        self._STATE_TRIGAM_PROBS = self._STATE_TRIGAM
        for STATE_V in self._STATE_TRIGAM_PROBS:
            for STATE_U in self._STATE_TRIGAM_PROBS[STATE_V]:
                state_u_total = float(sum(self._STATE_TRIGAM_PROBS[STATE_V][STATE_U].itervalues()))
                for STATE_K in self._STATE_TRIGAM_PROBS[STATE_V][STATE_U]:
                    self._STATE_TRIGAM_PROBS[STATE_V][STATE_U][STATE_K] /= state_u_total

        # populate trigram tags array to keep track of specific tags
        self._tags_trigram = []
        for STATE_V in self._STATE_TRIGAM_PROBS:
            # print previous
            self._tags.append(STATE_V)
        # reshuffle tags so that sent_begin is at index 0 and sent_end is at index -1
        # print "Unmodified" + str(len(self._tags))
        temp = self._tags_trigram[0]
        self._tags_trigram[self._tags_trigram.index('sent_begin')] = temp
        self._tags_trigram[0] = 'sent_begin'

        temp = self._tags_trigram[-1]
        self._tags_trigram[self._tags_trigram.index('.')] = temp
        self._tags_trigram[-1] = '.'

        # initialize the trigram transition matrix
        len_tags = len(self._tags_trigram)
        self._trigram_transition_matrix = np.zeros((len_tags, len_tags, len_tags))
        for i in range(0, len_tags):
            for j in range(0, len_tags):
                for k in range(0, len_tags):
                    if self._STATE_TRIGAM_PROBS[self._tags_trigram[i]][self._tags_trigram[j]].get(self._tags_trigram[k]):
                        self._trigram_transition_matrix[i][j][k] = self._STATE_TRIGAM_PROBS[self._tags_trigram[i]][self._tags_trigram[j]].get(self._tags_trigram[k])
                    else:
                        self._transition_matrix[i][j][k] = 0.0


    def predict(self, sentence_tokens):
        sentence_tokens.insert(0, 'sent_begin')
        viterbi = np.zeros((len(self._tags), len(sentence_tokens)))
        backpointers = np.zeros((len(self._tags), len(sentence_tokens)), dtype=np.int8)
        viterbi[0][0] = 1.0

        for i in range(1, len(self._tags)):
            backpointers[i][1] = 0
            prev_state_index = 0  # because this is initialization, max prob is at 0,0 index of 1
            # print "For tag " + tags[prev_state_index] + " The transition probability to " + tags[i] + " is " + str(
            #     transition_matrix[prev_state_index][i])
            if self._LIKELIHOOD_PROBS[self._tags[i]].get(sentence_tokens[1]):
                viterbi[i][1] = self._transition_matrix[prev_state_index][i] * self._LIKELIHOOD_PROBS[self._tags[i]].get(
                    sentence_tokens[1])
            else:
                word = sentence_tokens[1]
                oov_equivalent = self.oov_identifier(word)
                if self._LIKELIHOOD_PROBS[self._tags[i]].get(oov_equivalent):
                    viterbi[i][1] = self._transition_matrix[prev_state_index][i] * \
                                    self._LIKELIHOOD_PROBS[self._tags[i]].get(oov_equivalent)
                else:
                    viterbi[i][1] = self._transition_matrix[prev_state_index][i] * 0.0

        for word_index in range(2, len(sentence_tokens)):
            prev_word_column = word_index - 1
            prev_max_index = np.argmax(viterbi, axis=0)[prev_word_column]
            # print prev_max_index
            for state_index in range(1, len(self._tags)):
                backpointers[state_index][word_index] = prev_max_index
                if self._LIKELIHOOD_PROBS[self._tags[state_index]].get(sentence_tokens[word_index]):
                    viterbi[state_index][word_index] = viterbi[prev_max_index][prev_word_column] * \
                                                       self._transition_matrix[prev_max_index][state_index] * \
                                                       self._LIKELIHOOD_PROBS[self._tags[state_index]].get(
                                                           sentence_tokens[word_index])
                else:
                    word = sentence_tokens[word_index]
                    oov_equivalent = self.oov_identifier(word)

                    if self._LIKELIHOOD_PROBS[self._tags[state_index]].get(oov_equivalent):
                        viterbi[state_index][word_index] = viterbi[prev_max_index][prev_word_column] * \
                                                       self._transition_matrix[prev_max_index][state_index] * \
                                                       self._LIKELIHOOD_PROBS[self._tags[state_index]].get(oov_equivalent)
                    else:
                        viterbi[state_index][word_index] = viterbi[prev_max_index][prev_word_column] * \
                                                       self._transition_matrix[prev_max_index][state_index] * 0.0
        return self._get_tags(backpointers, sentence_tokens)

    def _get_tags(self, backpointers, sentence_tokens):
        return_tags = []
        for i in range(1, len(sentence_tokens) + 1):
            if i == len(sentence_tokens):
                tag = (sentence_tokens[-1], self._tags[-1])
                # print tag

            else:
                # print backpointers[1][i]
                tag = (sentence_tokens[i - 1], self._tags[backpointers[1][i]])
                # print tag
            return_tags.append(tag)

        return return_tags

    def predict_corpus(self, corpus_filename, result_filename):
        test_raw = open(os.path.join("files", corpus_filename), 'r').readlines()
        sentences = []
        sentence = []
        for line in test_raw:
            if line == '\n':
                sentences.append(sentence)
                # print sentence
                sentence = []
            else:
                # print line.rstrip()
                sentence.append(line.rstrip())

        # print sentences
        first_run = True
        with open(result_filename, 'w') as result_file:
            for sentence in sentences:
                result = self.predict(sentence)
                for word in result:
                    # avoid printing the leading new line
                    if first_run and word[0] == 'sent_begin':
                        first_run = False
                        continue
                    if word[0] == 'sent_begin':
                        result_file.write('\n')
                    else:
                        word_w_pos = '\t'.join(word) + '\n'
                        result_file.write(word_w_pos)
            result_file.write('\n')


def train_test_model():
    viterbi_model = viterbi_pos_tagger("WSJ_02-21.pos")
    viterbi_model.predict_corpus("WSJ_24.words","WSJ_24_self_tagged_w_oov.pos")




def printMatrixE(a):
   print "Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]"
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(0,rows):
      for j in range(0,cols):
         print("%6.25f" %a[i,j]),
      print
   print




if __name__ == "__main__":
    train_test_model()



train_raw = open( os.path.join("files", "WSJ_02-21.pos"), 'r').readlines()
    # initialize likelihood and transition dictionary
    LIKELIHOOD = {}
    STATE = {}
    i = 0
    previous = 'sent_begin'

    #pos tag for sent_end
    sent_end = '.'
    sent_begin = '\n'

    #iterate over training corpus line by line
    for line in train_raw:
        line_array = line.split("\t")

        if len(line_array) > 1:
            line_array[1] = line_array[1].rstrip()
        if line_array[0] == sent_begin:
            POS = 'sent_begin'
            WORD = 'sent_begin'
        elif line_array[1] == sent_end:
            WORD = line_array[0]
            POS = 'sent_end'
        else:
            WORD = line_array[0]
            POS = line_array[1]
        # populate the likelihood dictionary
        if not LIKELIHOOD.get(POS):
            LIKELIHOOD[POS] = {}
        if not LIKELIHOOD[POS].get(WORD):
            LIKELIHOOD[POS][WORD] = 1
        else:
            LIKELIHOOD[POS][WORD] += 1
        # populate the transition dictioary
        if not STATE.get(previous):
            STATE[previous] = {}
        if not STATE[previous].get(POS):
            STATE[previous][POS] = 1
        else:
            STATE[previous][POS] += 1

        previous = POS


    json.dump(LIKELIHOOD, file('LIKELIHOOD.txt', 'w'))
    json.dump(STATE, file('STATE.txt', 'w'))

    LIKELIHOOD_PROBS = LIKELIHOOD
    for POS in LIKELIHOOD_PROBS:
        pos_total = sum(LIKELIHOOD_PROBS[POS].itervalues())
        for word in LIKELIHOOD[POS]:
            LIKELIHOOD_PROBS[POS][word] = LIKELIHOOD_PROBS[POS][word]/float(pos_total)

    STATE_PROBS = STATE
    for previous in STATE_PROBS:
        previous_total = sum(STATE_PROBS[previous].itervalues())
        for pos in STATE_PROBS[previous]:
            STATE_PROBS[previous][pos] = STATE_PROBS[previous][pos]/float(previous_total)

    json.dump(LIKELIHOOD_PROBS, file('LIKELIHOOD_PROBS.txt', 'w'))
    json.dump(STATE_PROBS, file('STATE_PROBS.txt', 'w'))

    tags = []
    for previous in STATE_PROBS:
        tags.append(previous)

    # print "Unmodified" + str(len(tags))
    temp = tags[0]
    tags[tags.index('sent_begin')] = temp
    tags[0] = 'sent_begin'

    temp = tags[-1]
    tags[tags.index('sent_end')] = temp
    tags[-1] = 'sent_end'
    # print "Modified" + str(len(tags))
    # print tags

    transition_matrix = np.zeros((len(tags), len(tags)))

    for i in range(0, len(tags)):
        for j in range(0, len(tags)):
            if STATE_PROBS[tags[i]].get(tags[j]):
                # print "For tag " + tags[i] + " The transition probability to " + tags[j] + " is " + str(STATE_PROBS[tags[i]].get(tags[j]))
                transition_matrix[i][j] = STATE_PROBS[tags[i]].get(tags[j])
            else:
                # print "For tag " + tags[i] + " The transition probability to " + tags[j] + " is " + str(0.00)
                transition_matrix[i][j] = 0.0
    # pprint(transition_matrix)

    # viterbi_forward("The quick brown fox jumped.", tags, transition_matrix, LIKELIHOOD_PROBS)

    test_filename = "WSJ_24.words"
    test_raw = open(os.path.join("files", test_filename), 'r').readlines()
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
    with open('WSJ_24_self_tagged.pos', 'w') as result_file:
        for sentence in sentences:
            result = viterbi_forward(sentence, tags, transition_matrix, LIKELIHOOD_PROBS)
            for word in result:
                if word[0] == 'sent_begin':
                    result_file.write('\n')
                else:
                    word_w_pos = '\t'.join(word) + '\n'
                    result_file.write(word_w_pos)


def viterbi_forward(sentence, tags, transition_matrix, likelihood_table):
    if type(sentence) == 'str':
        sentence_tokens = word_tokenize(sentence)
    else:
        sentence_tokens = sentence
    sentence_tokens.insert(0, 'sent_begin')
    viterbi = np.zeros((len(tags), len(sentence_tokens)))
    backpointers = np.zeros((len(tags), len(sentence_tokens)), dtype=np.int8)
    viterbi[0][0] = 1.0

    for i in range(1, len(tags)):
        backpointers[i][1] = 0
        prev_state_index = 0 # because this is initialization

        # print "For tag " + tags[prev_state_index] + " The transition probability to " + tags[i] + " is " + str(
        #     transition_matrix[prev_state_index][i])
        if likelihood_table[tags[i]].get(sentence_tokens[1]):
            viterbi[i][1] = transition_matrix[prev_state_index][i] * likelihood_table[tags[i]].get(sentence_tokens[1])
        else:
            viterbi[i][1] = transition_matrix[prev_state_index][i] * 0.00




    for word_index in range(2, len(sentence_tokens)):
        prev_word_column = word_index - 1
        prev_max_index = np.argmax(viterbi, axis=0)[prev_word_column]
        # print prev_max_index
        for state_index in range(1, len(tags)):
            backpointers[state_index][word_index] = prev_max_index
            if likelihood_table[tags[state_index]].get(sentence_tokens[word_index]):
                viterbi[state_index][word_index] = viterbi[prev_max_index][prev_word_column] * \
                                                   transition_matrix[prev_max_index][state_index] * \
                                                   likelihood_table[tags[state_index]].get(sentence_tokens[word_index])
            else:
                viterbi[state_index][word_index] = viterbi[prev_max_index][prev_word_column] * \
                                                   transition_matrix[prev_max_index][state_index] * \
                                                   0.0

    return viterbi_backward(sentence_tokens, tags, backpointers)

def viterbi_backward(sentence_tokens, tags, backpointers):
    return_tags = []
    for i in range(1, len(sentence_tokens)+1):
        if i == len(sentence_tokens):
            tag = (sentence_tokens[-1], tags[-1])
            # print tag

        else:
            # print backpointers[1][i]
            tag = (sentence_tokens[i-1], tags[backpointers[1][i]])
            # print tag
        return_tags.append(tag)

    return return_tags

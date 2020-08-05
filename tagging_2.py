# -*- coding: utf-8 -*-

import os
import sys
import pandas
from pandas import DataFrame
import six
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.chunk import conlltags2tree, tree2conlltags
import codecs
from nltk.tokenize import RegexpTokenizer
from nltk.corpus.reader import ConllChunkCorpusReader 
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def remove_duplicate(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
input_dir = "all_text_files"

file_content = []
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        with codecs.open(os.path.join(root, filename), 'r', encoding='utf-8', 
            errors='ignore') as fd:
            for line in fd:
                file_content.append(line)
print (len(file_content))


required_sent_list_1 = []
for lc_1 in range (0, len(file_content)):
#for lc_1 in range (0, 1):
    file_content_sentences = sent_tokenize(file_content[lc_1])
    #print (file_content_sentences)

    for lc_2 in range (0, len(file_content_sentences)):
        word_list = word_tokenize(file_content_sentences[lc_2])
        pos_tag_result = pos_tag(word_list) 
        #print ("pos_tag_result: ", pos_tag_result)

        ne_tree = ne_chunk(pos_tag_result)
        iob_tagged = tree2conlltags(ne_tree)
    
        for lc_3 in range (0, len(iob_tagged)):
            if (iob_tagged[lc_3][0] == 'v.'):
                if ((iob_tagged[lc_3][1] == 'NNP') or (iob_tagged[lc_3][1] == 'NN') or
                    (iob_tagged[lc_3][1] == 'JJ') or (iob_tagged[lc_3][1] == 'FW') or
                    (iob_tagged[lc_3][1] == 'VBP')):
                    if (iob_tagged[lc_3+1][1] == 'NNP'):
                        required_sent_list_1.append(file_content_sentences[lc_2])
print (len(required_sent_list_1))

required_sent_list_2 = remove_duplicate(required_sent_list_1)
print (len(required_sent_list_2))

result_1 = "my_analysis_2.txt"
if(os.path.isfile(result_1)):
    os.remove(result_1)

for i in range (0, len(required_sent_list_2)):
    with open(result_1, "a", encoding='utf8') as fd_1:
        fd_1.write("count: {}, sentence: {}".format(i, required_sent_list_2[i]))
        fd_1.write("\n")
    fd_1.close()

  


        

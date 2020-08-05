import os
import sys
import pandas
from pandas import DataFrame
import numpy
import itertools
import collections
import heapq

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Sequential
from keras.utils import to_categorical

from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
global data
data = 0

NEWLINE = "\n"

ABC = "type-1"
DEF = "type-2"
GHI = "type-3"
JKL = "type-4"
UNK = "type-5"
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
SOURCES = [
    ('data/set_1',    ABC),
    ('data/set_2',    DEF),
    ('data/set_3a',   GHI),
    ('data/set_3b',   GHI),
    ('data/set_3c',   GHI),
    ('data/set_4',    JKL),
    ('data/set_5',    UNK),
]

SKIP_FILES = {'cmds'}
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []

                    f = open(file_path, encoding="latin-1")
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)

    return data_frame
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def load_training_data():
    global data, test

    # loading training data
    data = DataFrame({'text': [], 'class': []})
    for path, classification in SOURCES:
        data = data.append(build_data_frame(path, classification))

    data = data.reindex(numpy.random.permutation(data.index))
    print('Total number of emails: ', len(data))
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
########## Data Processing starts here #########################################
#------------------------------------------------------------------------------#
load_training_data()
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
text_words = []
for i in range (0, len(data)):
    temp_data = data['text'][i]
    temp_data_word = text_to_word_sequence(temp_data,
                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                     lower=True,
                     split=" ")
    text_words.append(temp_data_word)
#print (text_words)

word_counts = collections.Counter(itertools.chain(*text_words))
#print (word_counts)
vocabulary_size = len(word_counts)
print ("vocabulary_size = %s" %vocabulary_size)

"""
## for debugging only
temp_list = []
for i in range (0, len(text_words)):
    temp_list.append(len(text_words[i]))
print (heapq.nlargest(10, temp_list))
print (numpy.mean(temp_list))
"""

# sequence_length is the length of the sentence to be considered.
#sequence_length =  len(max(text_words, key=len))
sequence_length = 2000
print ("sequence_length = %s" %sequence_length)


# training - testing splitting
train_data, test_data, train_tags, test_tags = train_test_split(
        data['text'],
        data['class'],
        test_size=0.2,
        random_state=42)

#print (train_data)

print ("Num Training Mails = %s" %len(train_data))
print ("Num Testing Mails = %s" %len(test_data))
print ("Sum of Training and Testing mails = %s" %(len(train_data) +
        len(test_data)))
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
## Data Encoding
tokenizer = Tokenizer(num_words=sequence_length, 
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=" ",
                      char_level=False,
                      oov_token=None)


tokenizer.fit_on_texts(train_data) # only fit on train


# using mode = "tfidf"; other mode is also available
x_train = tokenizer.texts_to_matrix(train_data, mode="tfidf")
#print (x_train)


x_test = tokenizer.texts_to_matrix(test_data, mode="tfidf")
#print (x_test)


# To get the number of classes automatically
# the encoder can be also be used to get the prediction result in text
encoder = LabelEncoder()

encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = numpy.max(y_train) + 1
print ("Number of Classes found = %s" %num_classes)

# One Hot Encoding for the label
# This is different from keras.preprocessing text.one_hot
y_train = to_categorical(y_train, num_classes)
#print (y_train)

y_test = to_categorical(y_test, num_classes)
#print (y_test) 

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
########## Data Processing ends here ###########################################
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#### LSTM Starts here 
NUM_EPOCHS = 100
BATCH_SIZE = 32
DROP = 0.2

# Embedding hyper-parameters
EMBEDDING_VECTOR_SIZE = 512

# LSTM hyper-parameters:
HIDDEN_LAYER_SIZE = 50


## model description starts here
#------------------------------------------------------------------------------#
# creating a sequential model
training_model = Sequential()


# adding Embedding Layer
## Note: Keras offers an Embedding layer that can be used for neural networks 
##       on text data.
##       The Embedding layer can only be used as the first layer in a model.
##
##       It must specify 3 arguments:
##       input_dim: This is the size of the vocabulary in the text data. 
##       output_dim: This is the size of the vector space in which words will 
##                   be embedded. It defines the size of the output vectors 
##                   from this layer for each word. 
##       input_length: This is the length of input sequences, as like any input 
##                     layer of a Keras model. 

training_model.add(Embedding(input_dim=vocabulary_size,
                             output_dim=EMBEDDING_VECTOR_SIZE,
                             input_length=sequence_length))


# adding a dropout to tackle any over-fitting
training_model.add(Dropout(DROP))


# adding LSTM 
training_model.add(LSTM(HIDDEN_LAYER_SIZE))


# adding a dropout
training_model.add(Dropout(DROP))


# adding Dense
training_model.add(Dense(units=num_classes, activation='softmax'))



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# for multiclass classification problem
training_model.compile(optimizer = sgd, 
                       loss = "categorical_crossentropy",
                       metrics = ["accuracy"])

#print(training_model.summary())


print ("########  Training is starting now:  ########\n")

checkpoint = ModelCheckpoint('LSTM_weights_{epoch:03d}_{val_acc:.4f}.hdf5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')

# Model fitting
training_model.fit(x_train, y_train,  
                   batch_size=BATCH_SIZE, 
                   epochs=NUM_EPOCHS, 
                   verbose=1,
                   callbacks=[checkpoint],
                   validation_data=(x_test, y_test))


# Evaluate
score = training_model.evaluate(x_test, y_test, 
                                batch_size=BATCH_SIZE, 
                                verbose=1)

score_file = "LSTM_score.txt"
if(os.path.isfile(score_file)):
    os.remove(score_file)
score_file_fd = open(score_file, "w")

score_file_fd.write("Test score: [%s]\n" %score[0])
score_file_fd.write("Test accuracy: [%s]\n" %score[1])

score_file_fd.close()
#### LSTM Ends here
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#    
# It may happen: model.fit did not end the session cleanly. So, clearing the 
# session manually might be required.
# example of session object: which can raise error
# <tensorflow.python.client.session.Session object at 0x7f0896aabe48>
K.clear_session()
#------------------------------------------------------------------------------#



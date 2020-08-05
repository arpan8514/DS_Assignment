import os
import sys
import pandas
from pandas import DataFrame
import numpy
import itertools
import collections

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing import text
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model
from keras.utils import to_categorical

from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input, Bidirectional
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

load_training_data()

text_words = []
for i in range (0, len(data)):
    temp_data = data['text'][i]
    temp_data_word = text.text_to_word_sequence(temp_data,
                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                     lower=True,
                     split=" ")
    text_words.append(temp_data_word)

# sequence_length is the length of the sentence to be considered.
sequence_length =  len(max(text_words, key=len))

#sequence_length = 5000
print ("sequence_length = %s" %sequence_length)

word_counts = collections.Counter(itertools.chain(*text_words))
#print (word_counts)
vocabulary_size = len(word_counts)
print ("vocabulary_size = %s" %vocabulary_size)

# training - testing splitting
train_data, test_data, train_tags, test_tags = train_test_split(
        data['text'],
        data['class'],
        test_size=0.2,
        random_state=42)

print ("Num Training Mails = %s" %len(train_data))
#print (type(test_data))
print ("Num Testing Mails = %s" %len(test_data))
print ("Sum of Training and Testing mails = %s" %(len(train_data) +
        len(test_data)))

tokenize = text.Tokenizer(num_words=sequence_length, char_level=False)

tokenize.fit_on_texts(train_data) # only fit on train
x_train = tokenize.texts_to_matrix(train_data)
#print (x_train)

x_test = tokenize.texts_to_matrix(test_data)

encoder = LabelEncoder()

encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = numpy.max(y_train) + 1
print ("Number of Classes found = %s" %num_classes)
y_train = to_categorical(y_train, num_classes)
#print (y_train)
y_test = to_categorical(y_test, num_classes)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

#### Data Processing ends here #################################################

#### LSTM Starts here ##########################################################
NUM_EPOCHS = 40
BATCH_SIZE = 32
DROP = 0.4

# Embedding hyper-parameters
EMBEDDING_VECTOR_SIZE = 32

# LSTM hyper-parameters:
HIDDEN_LAYER_SIZE = 100

## model description starts here
#------------------------------------------------------------------------------#
# creating a tensor input
input_Tensor = Input(shape=(sequence_length,), dtype='int32')


# Adding Embedding Layer
embedding_Tensor = Embedding(input_dim=vocabulary_size,
                             output_dim=EMBEDDING_VECTOR_SIZE,
                             input_length=sequence_length)(input_Tensor)

lstm_Tensor = Bidirectional(LSTM(HIDDEN_LAYER_SIZE))(embedding_Tensor)


# Adding Dropout Layer
dropout_Tensor = Dropout(DROP)(lstm_Tensor)


# Adding Dense Layer
output_Tensor = Dense(units=num_classes, activation='softmax')(dropout_Tensor)


# keras model creation
training_model = Model(inputs=input_Tensor, outputs=output_Tensor)


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# for multiclass classification problem
training_model.compile(optimizer = sgd, 
                       loss = "categorical_crossentropy",
                       metrics = ["accuracy"])


print ("########  Training is starting now:  ########\n")

checkpoint = ModelCheckpoint(
        'Bidirectional_LSTM_weights_{epoch:03d}_{val_acc:.4f}.hdf5',
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


# Evaluating the training model
score = training_model.evaluate(x_test, y_test, 
                                batch_size=BATCH_SIZE, 
                                verbose=1)

score_file = "Bidirectional_LSTM_score.txt"
if(os.path.isfile(score_file)):
    os.remove(score_file)
score_file_fd = open(score_file, "w")

score_file_fd.write("Test score: [%s]\n" %score[0])
score_file_fd.write("Test accuracy: [%s]\n" %score[1])

score_file_fd.close()

#### LSTM Ends here ##########################################################

    
# It may happen: model.fit did not end the session cleanly. So, clearing the 
# session manually might be required.
# example of session object: which can raise error
# <tensorflow.python.client.session.Session object at 0x7f0896aabe48>
K.clear_session()




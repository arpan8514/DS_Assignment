import os
import sys
from pandas import DataFrame
import numpy
import collections
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model, load_model

from keras.layers import Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Flatten
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
global data, test
data = 0
test = 0

global vocabulary_size
vocabulary_size = 0

NEWLINE = "\n"

ABC = "category-1"
DEF = "category-2"
GHI = "category-3"
JKL = "category-4"
XYZ = "category-5"
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
SOURCES = [
    ('data/set_1',    ABC),
    ('data/set_2',    DEF),
    ('data/set_3a',   GHI),
    ('data/set_3b',   GHI),
    ('data/set_3c',   GHI),
    ('data/set_4',    JKL),
    ('data/set_5',    XYZ),
]

TST = "test"

TESTS = [
    ('test/tc_1', TST)
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
def load_training_testing_data():
    global data, test
    
    # loading training data
    data = DataFrame({'text': [], 'class': []})
    for path, classification in SOURCES:
        data = data.append(build_data_frame(path, classification))

    data = data.reindex(numpy.random.permutation(data.index))
    print('Total number of emails: ', len(data))

    # loading final test data
    test = DataFrame({'text': []})
    for path, classification in TESTS:
        #print (path)
        test = test.append(build_data_frame(path, classification))
    print('\nTotal number of tests: ', len(test))
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#    
########## Data Processing starts here #########################################
load_training_testing_data()
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
sequence_length =  len(max(text_words, key=len))
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
#### model creation starts here  ###############################################
BATCH_SIZE = 32
NUM_EPOCH = 50

# Embedding hyper-parameters
EMBEDDING_VECTOR_SIZE = 128

# Convolution hyper-parameters
KERNEL_SIZE = 4
NUM_FILTERS = 32


## model description starts here

# creating a tensor input
input_Tensor = Input(shape=(sequence_length,), dtype='int32')

# Adding Embedding Layer
embedding_Tensor = Embedding(input_dim=vocabulary_size,
                             output_dim=EMBEDDING_VECTOR_SIZE,
                             input_length=sequence_length)(input_Tensor)
#print (embedding_Tensor.shape)


# Adding a series/cascade of Convolution and Max-Pool layers
#----------------------------------------------------------#
conv_level_1_Tensor = Conv1D(filters=NUM_FILTERS,
                       kernel_size=KERNEL_SIZE, 
                       strides=1,
                       padding="valid",
                       activation="relu")(embedding_Tensor)

maxpool_level_1_Tensor = MaxPooling1D(pool_size=KERNEL_SIZE,
                                strides=1,
                                padding="valid")(conv_level_1_Tensor)
#----------------------------------------------------------#

#----------------------------------------------------------#
conv_level_2_Tensor = Conv1D(filters=NUM_FILTERS,
                             kernel_size=KERNEL_SIZE,
                             strides=1,
                             padding="valid",
                             activation="relu")(maxpool_level_1_Tensor)

maxpool_level_2_Tensor = MaxPooling1D(pool_size=KERNEL_SIZE,
                                      strides=1,
                                      padding="valid")(conv_level_2_Tensor)
#----------------------------------------------------------#

#----------------------------------------------------------#
conv_level_3_Tensor = Conv1D(filters=NUM_FILTERS,
                             kernel_size=KERNEL_SIZE,
                             strides=1,
                             padding="valid",
                             activation="relu")(maxpool_level_2_Tensor)

maxpool_level_3_Tensor = MaxPooling1D(pool_size=KERNEL_SIZE,
                                strides=1,
                                padding="valid")(conv_level_3_Tensor)
#----------------------------------------------------------#

#----------------------------------------------------------#
flatten_Tensor = Flatten()(maxpool_level_3_Tensor)

dense_Tensor = Dense(64, activation='relu')(flatten_Tensor)
output_Tensor = Dense(num_classes, activation='softmax')(dense_Tensor)


# CNN Model creation
training_model = Model(inputs=input_Tensor, outputs=output_Tensor)

print (training_model.summary())
## model description ends here
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

training_model.compile(loss='categorical_crossentropy',
                       optimizer=sgd,
                       metrics=['accuracy'])


checkpoint = ModelCheckpoint('CNN_1D_weights_{epoch:03d}_{val_acc:.4f}.hdf5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')


print ("Training is starting now")
training_model.fit(x_train, y_train,
                   batch_size=BATCH_SIZE,
                   epochs=NUM_EPOCH,
                   verbose=1,
                   callbacks=[checkpoint],
                   validation_data=(x_test, y_test))


score = training_model.evaluate(x_test, y_test,
                                batch_size=BATCH_SIZE, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
########## Model Creation ends here ############################################
#------------------------------------------------------------------------------#
'''
#------------------------------------------------------------------------------#
# ############### Test Data Prediction  ########################################
# final testing with new test data
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
test_model = load_model('CNN_1D_weights_018_0.9247.hdf5')
test_model.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['accuracy'])

# to generate a prediction on individual example
text_labels = encoder.classes_ 

new_sample_test_data = tokenizer.texts_to_matrix(test['text'])
#print (len(new_sample_test_data))
#print (new_sample_test_data.shape)

for i in range (0, len(new_sample_test_data)):

    prediction = test_model.predict(numpy.array([new_sample_test_data[i]]))
    predicted_label = text_labels[numpy.argmax(prediction)]
    print("Predicted label: " + predicted_label + "\n") 

    # for loop ends here   
######## Final Testing ends here ###############################################
#------------------------------------------------------------------------------#
'''
#------------------------------------------------------------------------------#
# It may happen: model.fit did not end the session cleanly. So, clearing the 
# session manually might be required.
# example of session object: which can raise error
# <tensorflow.python.client.session.Session object at 0x7f0896aabe48>
K.clear_session()
#------------------------------------------------------------------------------#



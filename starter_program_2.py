import os
import sys
from pandas import DataFrame
import numpy
#from classify_operation_2 import process_data
from classify_operation_2a import process_data
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
global data, test

data = 0
test = 0

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
clf_flag = 0

load_training_testing_data()

print (type(data))    


clf_flag = process_data(data, test, clf_flag)
    

if (clf_flag == 1):
    print ("Classification Complete")

else:
    print ("Sorry !!")    
    

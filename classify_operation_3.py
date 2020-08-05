""" 
    Naive Bayse --> Multinomial Naive Bayse 
"""


import os
from textblob import TextBlob
import numpy

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from sklearn.naive_bayes import MultinomialNB

import itertools
import matplotlib.pyplot as plt
from matplotlib.cbook import MatplotlibDeprecationWarning
import warnings
warnings.simplefilter('ignore', MatplotlibDeprecationWarning)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def split_into_tokens(message):
    #message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]
#------------------------------------------------------------------------------#    

#------------------------------------------------------------------------------#
def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    
    pc.update_scalarmappable()
    ax = pc.get_axes()
    
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if numpy.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)
#------------------------------------------------------------------------------#        

#------------------------------------------------------------------------------#
def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, 
    figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    
    '''
    Help taken from:
        - https://stackoverflow.com/a/16124677/395857 
        - https://stackoverflow.com/a/25074150/395857
    '''

    fig, ax = plt.subplots()    
    
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(numpy.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(numpy.arange(AUC.shape[1]) + 0.5, minor=False)

    # setting tick labels
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # removing last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def plot_classification_report(classification_report, 
    title='Classification report ', cmap='RdBu'):
    
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        #print(v)
        plotMat.append(v)

    #print ('plotMat: {0}'.format(plotMat))
    #print ('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    
    heatmap(numpy.array(plotMat), title, xlabel, ylabel, xticklabels, 
        yticklabels, figure_width, figure_height, correct_orientation, 
        cmap=cmap)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def process_data(data, test, clf_flag):
    

    mail_train, mail_test, label_train, label_test = train_test_split(
        data['text'], 
        data['class'], 
        test_size=0.2, 
    
        #random_state=42)
        random_state=0)

    print ("Num Training Mails = %s" %len(mail_train))
    print ("Num Testing Messages = %s" %len(mail_test))
    print ("Sum of Training and Testing mails = %s" %(len(mail_train) + len(mail_test)))

    
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(analyzer=split_into_lemmas)),

        # TfidfTransformer(): Transform a count matrix to a normalized tf or 
        # tf-idf representation
        ('tfidf', TfidfTransformer()),

        ('classifier', MultinomialNB())
    ])

    # pipeline parameters to automatically explore and tune
    tuned_parameters = [
        {'tfidf__use_idf': (True, False)},
        {'vectorizer__analyzer': (split_into_lemmas, split_into_tokens)},
    ]

    grid = GridSearchCV(
        pipeline,
    
        # parameters to tune via cross validation
        tuned_parameters,  
    
        refit=True,  
        n_jobs=1,  
        scoring='accuracy',
    
        # type of cross validation to use
        cv = StratifiedKFold(n_splits=10)  
    )

    my_detector = grid.fit(mail_train, label_train)

    print("\nBest parameters set found on development set: %s" %grid.best_params_)

    predictions =  my_detector.predict(mail_test)

    #print ("\n\nClassification Report:")
    clf_rep = classification_report(label_test, predictions, labels=None, 
        target_names=None, sample_weight=None)

    plot_classification_report(clf_rep)
    plt.savefig('classification_report.png', dpi=200, format='png', 
        bbox_inches='tight')
    plt.close()



    lines = clf_rep.split('\n')
    classes = []
    class_names = []

    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        class_names.append(t[0])

    #print  (class_names)   


    #print ('\n\nConfusion matrix on Testing Messages:')
    cnf_matrix = (confusion_matrix(label_test, predictions, labels=None, 
        sample_weight=None))

    numpy.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
        title='Confusion Matrix - without normalization')
    plt.savefig('CNF_Matrix_without_normalization.png', dpi=200, format='png', 
        bbox_inches='tight')
    plt.close()


    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
        title='Normalized Confusion Matrix')
    plt.savefig('Normalized_CNF_Matrix.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()

    #plt.show()

    print ("\n\nF1 Score:")
    print (f1_score(label_test, predictions, labels=None, pos_label=1, 
        average='weighted', sample_weight=None))

    # final testing
    for i in range (0, len(test)):
        test_sample = numpy.array([test.iloc[i]['text']])
        prediction_result =  my_detector.predict(test_sample)
        print ("Result: %s" %prediction_result)
    
    
    clf_flag = 1
    return clf_flag
#------------------------------------------------------------------------------#

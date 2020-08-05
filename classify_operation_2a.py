""" 
    Support Vector Machine (SVM) classifier :  SGDClassifier --> loss = hinge 
"""

import os
import sys
from textblob import TextBlob
import numpy

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import SGDClassifier
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
if (sys.platform == "win32"):
    #subprocess.run("chcp 65001", shell=True)
    import win_unicode_console
    win_unicode_console.enable()
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
def process_data(data, test, clf_flag):
    

    mail_train, mail_test, label_train, label_test = train_test_split(
        data['text'], 
        data['class'], 
        test_size=0.2, 
    
        #random_state=42)
        random_state=0)

    #print (type(mail_train))
    #print (mail_train.shape)

    print ("Num Training Mails = %s" %len(mail_train))
    print ("Num Testing Messages = %s" %len(mail_test))
    print ("Sum of Training and Testing mails = %s" %(len(mail_train) + 
        len(mail_test)))

    
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(analyzer=split_into_lemmas)),

        # TfidfTransformer(): Transform a count matrix to a normalized tf or 
        # tf-idf representation
        ('tfidf', TfidfTransformer()),

        ('classifier', SGDClassifier(
            alpha=0.0002, 
            average=False, 
            class_weight=None, 
            epsilon=0.1, 
            eta0=0.0, 
            fit_intercept=True, 
            l1_ratio=0.15, 
            learning_rate='optimal', 
            loss='hinge',  # linaer SVM 
            max_iter=1000, 
            n_jobs=1, 
            penalty='l2', 
            power_t=0.5, 
            random_state=None, 
            shuffle=True, 
            verbose=0, 
            warm_start=False))
    ])

    # pipeline parameters to automatically explore and tune
    tunning_parameters = [
        {'tfidf__use_idf': (True, False)},
        {'vectorizer__analyzer': (split_into_lemmas, split_into_tokens)},
    ]

    grid = GridSearchCV(
        pipeline,
    
        # parameters to tune via cross validation
        tunning_parameters,  
    
        refit=True,  
        n_jobs=1,  
        scoring='accuracy',
    
        # type of cross validation to use
        cv = StratifiedKFold(n_splits=10)  
    )

    my_detector = grid.fit(mail_train, label_train)

    #print("\nBest parameters set found on development set: %s" %grid.best_params_)

    predictions =  my_detector.predict(mail_test)

    print ("\n\nClassification Report:")
    clf_rep = classification_report(label_test, predictions, labels=None, 
        target_names=None, sample_weight=None)
    print (clf_rep)
    
    
    print ('\n\nConfusion matrix on Testing Messages:')
    cnf_matrix = (confusion_matrix(label_test, predictions, labels=None, 
        sample_weight=None))
    print (cnf_matrix)


    print ("\n\nF1 Score:")
    F1_Score = f1_score(label_test, predictions, labels=None, pos_label=1, 
        average='weighted', sample_weight=None)
    print (F1_Score)    
        
        
    # final testing with new test data
    for i in range (0, len(test)):
        test_sample = numpy.array([test.iloc[i]['text']])
        prediction_result =  my_detector.predict(test_sample)
        
        #print ("**************************************************************")
        #print ("Input Text:\n")
        #print (test.iloc[i]['text'])
        print ("\nOutput Result: [%s]" %prediction_result)
        #print ("**************************************************************")
    
    
    clf_flag = 1
    return clf_flag
#------------------------------------------------------------------------------#

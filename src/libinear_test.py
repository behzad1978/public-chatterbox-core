__author__ = 'behzadbehzadan'

#Code showing how to use LIBLINEAR.
#Note that first the C file liblinear.so.1 must be created in the working folder by the Makefile using make command.
#To add the LIBLINEAR library to Pycharm and using it, do the following:
    #1. Go to Pycharm Preferences --> Project Interpreter --> Python Interpreters
    #2. In the bottom pane click on Paths --> then select + --> navigate to the liblinear zip file (liblinear.tar.gz)

#To use the liblinear binary code from the shell, we need to create the liblinear.so.1 C-file by executing make in the
#Shell from the same folder that the liblinear files have been extracted to.

from liblinearutil import *

#The 'heart_scale' file is a text file with each observation saved in one line. The first column indicates the labels
#assigned to each observation. The rest of the columns are the features (signals) of the observations. Hence, a file
#with m lines and n+1 column indicates that there are m observations, each having n features together with m labels for
#each observation. Thinking about a neural network model, observations can be considered as the training/test set (m is
#equal to the epoch size), while features are equivalent to the input signals (n is equal to the number of input nodes),
#and finally labels correspond to the output nodes (if we have +1 and -1 as labels, these would correspond to 2 output
#nodes).
input_file_path = '/Users/behzadbehzadan/Chatterbox_UCL_Advance/LIBLINEAR/liblinear-1.92'
input_fle_name = '/heart_scale'

#The file is read line by line. The first column are the labels (-1, +1) which is stored in y. The training data is
#stored in x.
y, x = svm_read_problem(input_file_path+input_fle_name)

#We train the file using liblinear functions. There are some parameters:
    #-s --> defining function type (default 1) --> for classification chose from 1 to 7 --> for regression 11 to 13
    #-c --> defining cost function (default 1) --> chose from 0 and try going up to 4 (see if you see any difference)
    #-B --> defining bias (default -1) --> bias<0 means no bias is used; bias>0 means instance x converts to [x; bias]
m = train(y[:200], x[:200], '-s 1 -c 4')

#we could also use the following alternative representation to do the training:
another_way=False
if another_way:
    prob = problem(y, x)
    param = parameter('-s 1 -c 4')
    m = train(prob, param)

#p_labels: a list of predicted labels
#p_acc: a tuple including accuracy (for classification), MSE, and squared correlation coefficient (for regression).
p_label, p_acc, p_val = predict(y[200:], x[200:], m)

#we can save the trained model to reload it later.
save_model('heart_scale.model', m)

#load the model again
m = load_model('heart_scale.model')

p_label, p_acc, p_val = predict(y, x, m)

ACC, MSE, SCC = evaluations(y, p_label)

######################################################### Second Test ##################################################

#Example of dense data
y, x = [1, -1], [[1, 0, 1], [-1, 0, -1]]
#Example of Sparse data
#y, x = [1, -1], [{1: 1, 3: 1}, {1: -1, 3: -1}]
prob = problem(y, x)
param = parameter('-c 4 -B 1')
m = train(prob, param)

######################################################### Third Test ##################################################

#This test distinguishes squares from rectangles!

#+1 labels refers to squares. -1 labels refer to rectangles.
y = [1, 1, 1, 1, -1, -1, -1, -1]

#The system is not sensitive over choice of labels --> we can use +1 and 0 instead of +1 and -1.
y = [1, 1, 1, 1, 0, 0, 0, 0]

#Or we can even use +1 and -2 as labels
y = [1, 1, 1, 1, -2, -2, -2, -2]

#The width/length of the squares or rectangles are listed --> [2,2] refers to a square while [6,9] to a rectangle.
#x = [ [2,2], [4,4], [9,9], [10,10], [5,6], [3,4], [6,9], [4,1] ]
#we can make the data sparse by increasing the input dimension without affecting the final result.
x = [[2, 0, 2, 0], [4, 0, 4, 0], [9, 0, 9, 0], [10, 0, 10, 0], [5, 0, 6, 0], [3, 0, 4, 0], [6, 0, 9, 0], [4, 0, 1, 0]]

prob = problem(y, x)

param = parameter('-s 3 -c 1 -B 0')

m = train(prob, param)

#p_acc: a tuple including accuracy (for classification), MSE, and squared correlation coefficient (for regression).
p_label, p_acc, p_val = predict(y, x, m)

#we perform 3-fold cross validation
param_cv = parameter('-s 3 -c 1 -B 0 -v 3')
m_cv = train(prob, param_cv)

################################################### Read from Text File ################################################

#In the following we show how to read the above input data from a text file rather than from inside the code itself.

#we have saved the data into a text file:
    #the first column indicates the labels.
    #each row represents an input data with their features (input signals) presented in respective columns.
input_file_path = '/Users/behzadbehzadan/Chatterbox KEA (UCL Advance)/LIBLINEAR/liblinear-1.92/python'
input_fle_name = '/square_prob'

y, x = svm_read_problem(input_file_path + input_fle_name)

prob = problem(y, x)

param = parameter('-s 3 -c 1 -B 0')

m = train (prob, param)

p_label, p_acc, p_val = predict(y, x, m)

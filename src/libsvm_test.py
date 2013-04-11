__author__ = 'behzadbehzadan'
import svm
from svmutil import *

#Example of dense data
x = [[1, 1], [1,2], [2,2], [2,1], [-1,-1],[-1,-2],[-2,-2],[-2,-1],[-4,4],[-5,4],[-6,5],[-7,5]]
y = [1]*4 + [-1]*4 + [0]*4
#Example of Sparse data
#y, x = [1, -1], [{1: 1, 3: 1}, {1: -1, 3: -1}]
prob = svm_problem(y, x)
param = svm_parameter('-s 0 -t 0 -c 4 -b 1')
m = svm_train(prob, param)

#Q: How do I get the decision value(s)?
#We print out decision values for regression. For classification, we solve several binary SVMs for multi-class cases.
#You can obtain values by easily calling the subroutine svm_predict_values. Their corresponding labels can be obtained
#from svm_get_labels. Details are in README of libsvm package.

x_test = [[3,3], [4, 3], [4, 4], [4, 3], [-3, -3], [-3, -4], [-4, -3], [-4, -4], [-6, 4], [-2, 3],[-10, 10], [-7, 5]]
p_label, p_acc, p_val = svm_predict(y, x_test, m)


##############################################################

# #The width/length of the squares or rectangles are listed --> [2,2] refers to a square while [6,9] to a rectangle.
# x = [ [2,2], [4,4], [9,9], [10,10], [5,6], [3,4], [6,9], [4,1] ]
# y = [1, 1, 1, 1, -2, -2, -2, -2]
#
# prob = svm_problem(y, x)
#
# param = svm_parameter('-s 3 -t 0 -c 1 -b 0')
#
# m = svm_train(prob, param)
#
# p_label, p_acc, p_val = svm_predict(y, x, m)



print "xxx"
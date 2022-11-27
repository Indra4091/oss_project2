
#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/Indra4091/oss_project2.git

import sys
from sys import argv
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def load_dataset(dataset_path):
	#To-Do: Implement this function
	data = pd.read_csv(dataset_path, encoding = 'utf-8')
	return data

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	df = dataset_df.groupby("target").size()
	return dataset_df.shape[1] - 1, df[0], df[1]

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	x = dataset_df.drop(columns="target", axis=1)
	y = dataset_df["target"]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testset_size, random_state = 2)
	return x_train, x_test, y_train, y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train, y_train)
	acc = accuracy_score(y_test, dt_cls.predict(x_test))
	prec = precision_score(y_test, dt_cls.predict(x_test))
	recall = recall_score(y_test, dt_cls.predict(x_test))
	return acc, prec, recall


def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	rf_cls = RandomForestClassifier()
	rf_cls.fit(x_train, y_train)
	acc = accuracy_score(rf_cls.predict(x_test), y_test)
	prec = precision_score(y_test, rf_cls.predict(x_test))
	recall = recall_score(y_test, rf_cls.predict(x_test))
	return acc, prec, recall
	

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	svm_pipe = make_pipeline(StandardScaler(), SVC()) #normalizing the features
	svm_pipe.fit(x_train, y_train)
	acc = accuracy_score(y_test, svm_pipe.predict(x_test))
	prec = precision_score(y_test, svm_pipe.predict(x_test))
	recall = recall_score(y_test, svm_pipe.predict(x_test))
	return acc, prec, recall

def print_performances(acc, prec, recall):
#	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)



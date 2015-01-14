import re, math, collections, itertools, os
import nltk
from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier

# import data
data_processed_positive = os.path.join('data', 'processed_bi', 'positive.txt')
data_processed_negative = os.path.join('data', 'processed_bi', 'negative.txt')

#creates a feature selection mechanism that uses all words
def get_word_set(words):
	return dict([(word, True) for word in words])

#this function takes a feature selection mechanism and returns its performance in a variety of metrics
def rate_feature(feature_select):
	feature_positive = []
	feature_negative = []

	with open(data_processed_positive, 'r') as sentence_positive:
		for i in sentence_positive:
			word_positive = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			word_positive = [feature_select(word_positive), 'pos']
			feature_positive.append(word_positive)

	with open(data_processed_negative, 'r') as sentence_negative:
		for i in sentence_negative:
			word_negative = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			word_negative = [feature_select(word_negative), 'neg']
			feature_negative.append(word_negative)

	#split the data into training and test set just by natural data order
	split_position_positive = int(math.floor(len(feature_positive)*9/10))
	split_position_negative = int(math.floor(len(feature_negative)*9/10))
	feature_training = feature_positive[:split_position_positive] + feature_negative[:split_position_negative]
	feature_test = feature_positive[split_position_positive:] + feature_negative[split_position_negative:]

	#train using Naive Bayes Classifier
	classifier = NaiveBayesClassifier.train(feature_training)	

	#initiates reference Sets and test Sets
	set_def = collections.defaultdict(set)
	set_test = collections.defaultdict(set)	

	#puts correctly labeled sentences in set_def and the predictively labeled version in test sets
	for i, (features, label) in enumerate(feature_test):
		set_def[label].add(i)
		predicted = classifier.classify(features)
		set_test[predicted].add(i)	

	#prints metrics to show how well the feature selection did
	print 'train set: %d samples\ntest set: %d samples' % (len(feature_training), len(feature_test))
	print 'accuracy:', nltk.classify.util.accuracy(classifier, feature_test)
	print 'positive accuracy:', nltk.metrics.precision(set_def['pos'], set_test['pos'])
	print 'negative accuracy:', nltk.metrics.precision(set_def['neg'], set_test['neg'])

#tries using all words as the feature selection mechanism
rate_feature(get_word_set)
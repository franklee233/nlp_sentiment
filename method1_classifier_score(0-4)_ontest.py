import re, math, os, collections, itertools
import nltk
from nltk.classify import NaiveBayesClassifier

#io
data_processed_score0 = os.path.join('data', 'processed_score', 'score0.txt')
data_processed_score1 = os.path.join('data', 'processed_score', 'score1.txt')
data_processed_score2 = os.path.join('data', 'processed_score', 'score2.txt')
data_processed_score3 = os.path.join('data', 'processed_score', 'score3.txt')
data_processed_score4 = os.path.join('data', 'processed_score', 'score4.txt')
data_test_input = os.path.join('data', 'processed_score', 'test.txt')
data_test_output = "test_output.txt"

#get the word set
def get_word_set(words):
	return dict([(word, True) for word in words])

#classification
def rate_feature(feature_select):
	feature_score0 = []
	feature_score1 = []
	feature_score2 = []
	feature_score3 = []
	feature_score4 = []
	feature_test = []

	with open(data_processed_score0, 'r') as sentence_score0:
		for i in sentence_score0:
			word_score = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			word_score = [feature_select(word_score), '0']
			feature_score0.append(word_score)

	with open(data_processed_score1, 'r') as sentence_score1:
		for i in sentence_score1:
			word_score = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			word_score = [feature_select(word_score), '1']
			feature_score1.append(word_score)

	with open(data_processed_score2, 'r') as sentence_score2:
		for i in sentence_score2:
			word_score = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			word_score = [feature_select(word_score), '2']
			feature_score2.append(word_score)

	with open(data_processed_score3, 'r') as sentence_score3:
		for i in sentence_score3:
			word_score = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			word_score = [feature_select(word_score), '3']
			feature_score3.append(word_score)

	with open(data_processed_score4, 'r') as sentence_score4:
		for i in sentence_score4:
			word_score = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			word_score = [feature_select(word_score), '4']
			feature_score4.append(word_score)

	with open(data_test_input, 'r') as test_input:
		for i in test_input:
			word_score = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			word_score = [feature_select(word_score), '4']
			feature_test.append(word_score)


	feature_training = feature_score0 + feature_score1 + feature_score2 + feature_score3 + feature_score4

	classifier = NaiveBayesClassifier.train(feature_training)	

	set_def = collections.defaultdict(set)
	#set_test['i'] contains line no. of sentence with sentiment score i
	set_test = collections.defaultdict(set)	
	for i, (features, label) in enumerate(feature_test):
		set_def[label].add(i)
		predicted = classifier.classify(features)
		set_test[predicted].add(i)

	print 'train set: %d samples\ntest set: %d samples' % (len(feature_training), len(feature_test))

	with open(data_test_output, 'wb') as test_output:
		for i in range (0,5):
			test_output.write("\nsentiment score %d \n" % i)
			for phrase in set_test[str(i)]:
	  			test_output.write("%s , " % phrase)

#run
rate_feature(get_word_set)
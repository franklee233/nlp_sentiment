import re, math, collections, itertools, os
import nltk
from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier
from nltk.probability import FreqDist, ConditionalFreqDist

# import data
data_processed_positive = os.path.join('data', 'processed_bi', 'positive.txt')
data_processed_negative = os.path.join('data', 'processed_bi', 'negative.txt')

#this function takes a feature selection mechanism and returns its performance in a variety of metrics
def evaluate_features(feature_select):
	feature_postive = []
	feature_negative = []

	with open(data_processed_positive, 'r') as sentence_postive:
		for i in sentence_postive:
			word_positive = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			word_positive = [feature_select(word_positive), 'pos']
			feature_postive.append(word_positive)

	with open(data_processed_negative, 'r') as sentence_negative:
		for i in sentence_negative:
			word_negative = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			word_negative = [feature_select(word_negative), 'neg']
			feature_negative.append(word_negative)

	#split the data into training and test set just by natural data order
	split_position_postive = int(math.floor(len(feature_postive)*9/10))
	split_position_negative = int(math.floor(len(feature_negative)*9/10))
	feature_traning = feature_postive[:split_position_postive] + feature_negative[:split_position_negative]
	feature_test = feature_postive[split_position_postive:] + feature_negative[split_position_negative:]

	#train using Naive Bayes Classifier
	classifier = NaiveBayesClassifier.train(feature_traning)	

	#initiates reference Sets and test Sets
	set_def = collections.defaultdict(set)
	set_test = collections.defaultdict(set)	

	#puts correctly labeled sentences in set_def and the predictively labeled version in test sets
	for i, (features, label) in enumerate(feature_test):
		set_def[label].add(i)
		predicted = classifier.classify(features)
		set_test[predicted].add(i)	

	#prints metrics to show how well the feature selection did
	print 'train on %d instances, test on %d instances' % (len(feature_traning), len(feature_test))
	print 'accuracy:', nltk.classify.util.accuracy(classifier, feature_test)
	print 'pos precision:', nltk.metrics.precision(set_def['pos'], set_test['pos'])
	print 'pos recall:', nltk.metrics.recall(set_def['pos'], set_test['pos'])
	print 'neg precision:', nltk.metrics.precision(set_def['neg'], set_test['neg'])
	print 'neg recall:', nltk.metrics.recall(set_def['neg'], set_test['neg'])
	classifier.show_most_informative_features(10)

#creates a feature selection mechanism that uses all words
def make_full_dict(words):
	return dict([(word, True) for word in words])

#tries using all words as the feature selection mechanism
print 'using all words as features'
evaluate_features(make_full_dict)

#scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
def rate_word():
	#creates lists of all positive and negative words
	word_positive = []
	word_negative = []

	with open(data_processed_positive, 'r') as sentence_postive:
		for i in sentence_postive:
			posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			word_positive.append(posWord)
	with open(data_processed_negative, 'r') as sentence_negative:
		for i in sentence_negative:
			negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			word_negative.append(negWord)
			
	word_positive = list(itertools.chain(*word_positive))
	word_negative = list(itertools.chain(*word_negative))

	#build frequency distribution of all words and then frequency distributions of words within positive and negative labels
	word_fd = FreqDist()
	cond_word_fd = ConditionalFreqDist()
	for word in word_positive:
		word_fd.inc(word.lower())
		cond_word_fd['pos'].inc(word.lower())
	for word in word_negative:
		word_fd.inc(word.lower())
		cond_word_fd['neg'].inc(word.lower())

	#finds the number of positive and negative words, as well as the total number of words
	pos_word_count = cond_word_fd['pos'].N()
	neg_word_count = cond_word_fd['neg'].N()
	total_word_count = pos_word_count + neg_word_count

	#builds dictionary of word scores based on chi-squared test
	word_scores = {}
	for word, freq in word_fd.iteritems():
		pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
		neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
		word_scores[word] = pos_score + neg_score

	return word_scores

#finds word scores
word_scores = rate_word()

#finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
	best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
	best_words = set([w for w, s in best_vals])
	return best_words

#creates feature selection mechanism that only uses best words
def best_word_features(words):
	return dict([(word, True) for word in words if word in best_words])

#numbers of features to select
numbers_to_test = [10, 100, 1000, 10000, 15000]
#tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
	print 'evaluating best %d word features' % (num)
	best_words = find_best_words(word_scores, num)
	evaluate_features(best_word_features)

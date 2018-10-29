from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from math import log, sqrt 
import multiprocessing as mtp 
import nltk
import matplotlib.pyplot as plt 
import wordcloud as WordCloud
import pandas as pd 
import numpy as np 
import os 
import sys
import csv

class spamFilter():

	def __init__(self):

		os.chdir("C:/Users/julia/SpamOrHamProject/data")

		dataSet = pd.read_csv('spam.csv', encoding='latin-1')
		dataSet = dataSet.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
		dataSet = dataSet.rename(columns={'v1':'label', 'v2':'message'})

		totalMails = dataSet['message'].shape[0]
		trainIndex, testIndex = list(), list()

		for i in range(dataSet.shape[0]):
			if np.random.uniform(0, 1) < 0.75:
				trainIndex += [i]
			else:
				testIndex += [i] 

		trainData = dataSet.loc[trainIndex]
		testData = dataSet.loc[testIndex]

		trainData.reset_index(inplace = True)
		trainData.drop(['index'], axis = 1, inplace = True)

		testData.reset_index(inplace = True)
		testData.drop(['index'], axis = 1, inplace = True)

		allwordsuse = dict()
		spamuse = dict()
		hamuse = dict()

		numwordsspam = 0
		numwordsham = 0 
		numwordsall = 0

		for i in range(0, len(trainData)):
			words = word_tokenize(trainData.loc[i]['message'])
	#		words = [word for word in words if len(words) > 2]
			for i in range(0, len(words)):

				numwordsall += 1

				allwordsuse[words[i].lower()] = allwordsuse.get(words[i].lower(), 0) + 1

				if trainData.loc[i]['label'] == 'spam':
					spamuse[words[i].lower()] = spamuse.get(words[i].lower(), 0) + 1
					numwordsspam += 1

				else:
					hamuse[words[i].lower()] = spamuse.get(words[i].lower(), 0) + 1
					numwordsham += 1


		self.dataSet = dataSet 
		self.totalMails = totalMails
		self.trainIndex = trainIndex
		self.testIndex = testIndex
		self.trainData = trainData
		self.testData = testData
		self.allwordsuse = allwordsuse
		self.spamuse = spamuse
		self.hamuse = hamuse
		self.numwordsall = numwordsall
		self.numwordsham = numwordsham
		self.numwordsspam = numwordsspam
		alpha = 0.8
		self.alpha = alpha

	def visualizeSpam(self):
		spam_words = ''.join(list(dataSet[dataSet['label'] == 'spam']['message']))
		spam_wc = WordCloud(width = 512, height = 512).generate(spam_words)

		plt.figure(figsize = (10, 8), facecolor = 'k')
		plt.imshow(spam_wc)
		plt.axis('off')
		plt.tight_layout(pad = 0)
		plt.show()

	def visualizeHam(self):
		spam_words = ''.join(list(dataSet[dataSet['label'] == 'ham']['message']))
		spam_wc = WordCloud(width = 512, height = 512).generate(spam_words)

		plt.figure(figsize = (10, 8), facecolor = 'k')
		plt.imshow(spam_wc)
		plt.axis('off')
		plt.tight_layout(pad = 0)
		plt.show()

	def processMessage(self, message, lower_case = True, stem = True, stop_words = True, gram = 2):
		if lower_case:
			message = message.lower()
		
		words = word_tokenize(message)
		words = [w for w in words if len(w) > 2]

		if gram > 1:
			w = [] 
			for i in range(len(words) - gram + 1):
				w += [''.join(words[i:i + gram])]
			return w 

		if stop_words:
			sw = stopwords.words('english')
			words = [word for word in words if word not in sw]

		if stem:
			stemmer = PorterStemmer()
			words = [stemmer.stem(word) for word in words]

		return words

	def classify(self, message):

		mt = word_tokenize(message)

		probSpam = 1

		for i in range(0, len(mt)):

			try:
				tempVal = len(self.trainData) / self.allwordsuse.get(mt[i], 0)


				idfW = log(tempVal)
			except:
				idfW = 0

			tempNumerator = self.spamuse.get(mt[i], 0) * idfW + self.alpha 

			tempIdfxVal = self.numwordsall / self.numwordsspam

			idfX = log(tempIdfxVal)

			tempDivisor = self.numwordsall * idfX + self.alpha * self.numwordsspam

			print(tempNumerator)
			print(tempDivisor)
			print(self.allwordsuse.get(mt[i], 0))
			print(mt[i])

			pWSpam = tempNumerator / tempDivisor

			probSpam = probSpam * pWSpam

		print(probSpam)
		if probSpam > .5 :
			return 'spam'
		else:
			return 'ham'

	def evaluate(self):

		numCorrect = 0
		numIncorrect = 0
		numTotal = len(self.testData)

		for i in range(0, len(self.testData)):

			tmpStr = self.classify(self.testData.loc[i]['message'])

			print(type(tmpStr))
			print(tmpStr)

			print(self.testData.loc[i]['label'])

			tempVal = self.testData.loc[i]['label']

			if tmpStr == tempVal:

				numCorrect = numCorrect + 1 

			else:

				numIncorrect = numIncorrect + 1
		print(numCorrect)
		print(numTotal)
		print(numCorrect / numTotal)

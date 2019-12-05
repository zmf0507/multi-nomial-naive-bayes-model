from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import pickle
import os
import math


def removeStopWords(words):
	filteredWords = []
	stop_words=stopwords.words('english')	
	for word in words:
		if word not in stop_words:
			filteredWords.append(word)

	return filteredWords

def loadFromPickle(pickleFile):
	file = open(pickleFile,'rb')
	pickleData = pickle.load(file)
	file.close()
	return pickleData	

classCountWords = loadFromPickle("class_word_count.pickle")


def setClassProbability(pathList):
	# groupPath='20_newsgroup'
	docCount=0
	totalCount=0
	count={}
	for path in pathList:
		for r, d, f in os.walk(path):
			for file in f:
				docCount+=1
		count[getClassName(path)]=docCount
		totalCount+=docCount
		docCount=0
	
	for path in pathList:
		className=getClassName(path)
		count[className]=float(count[className])/float(totalCount)
	return count	

def getClassName(path):
	path = path[::-1]
	path = path.split('/')[0]
	path = path[::-1]
	return path

def getAllDataSet():
	filesList=[]
	path="20_newsgroups"
	# for r, d, folder in os.walk(path):
	# 		# filesList.append(os.path.join(r, folder))
	# 		print(folder)
	filesList =  [x[0] for x in os.walk(path)]

	return filesList[1:]


def classifyDoc(taskNumber, k, vocabulary, testData, className):
	wordProbabilty = 1
	prob = 0
	trainedModel = loadFromPickle(str(k)+"-model_task_"+str(taskNumber)+".pickle")
	for word in testData:
		# print(trainedModel[getClassName(className)][word])
		if (word in trainedModel[getClassName(className)]):
			wordProbabilty=trainedModel[getClassName(className)][word]
			prob+=math.log(wordProbabilty,10)
			# print("invoc", trainedModel[getClassName(className)][word])
			# print(wordProbabilty)
		else:
			wordProbabilty=setOovProbabilty(word, len(vocabulary), className, k)
			prob+=math.log(wordProbabilty,10)	
		# if(wordProbabilty==0.0):
		# 	print(word)
	return prob

def setOovProbabilty(word, vocSize , className, k):
	prob = float(k)/float(classCountWords[getClassName(className)]+(vocSize+1)*k)
	return prob

# def getClassWordsCount(trainedModel, className):
# 	count=0
# 	for key in trainedModel

task1 = [
'20_newsgroups/rec.motorcycles',
'20_newsgroups/rec.sport.baseball',
]

task2 = getAllDataSet()

tasks = [task1, task2]
kSmoothValues = [1,5,10,100]

print("Enter the path of the file")
path=input()
file = open(path, 'r')
# print(file.read()) 
testWords = file.read()
# testWords = testWords.split()
list=word_tokenize(testWords)
tokenizer=RegexpTokenizer(r'([A-Za-z0-9]+)')
testWords=tokenizer.tokenize(testWords)
testWords=removeStopWords(testWords)

count = 1
for k in kSmoothValues:
	count = 1
	print("FOR K = ", k)
	for task in tasks:
		maxProb = -1000000000000000
		maxClass = ""
		classProbabilty = setClassProbability(task)
		print("Task-",count)
		vocabulary = loadFromPickle("vocabulary_task_" + str(count) +".pickle")
		for className in task:
				docProb = classifyDoc(count, k, vocabulary, testWords, className)
				# print(docProb)
				docProb+=math.log(classProbabilty[getClassName(className)],10)
				print(className, ":", docProb)
				# print(docProb)
				if(docProb > maxProb):
					maxProb=docProb
					maxClass=getClassName(className)

		print("\nMax Probabilty Log:", maxProb,10, "className : ", maxClass)				
		count+=1			


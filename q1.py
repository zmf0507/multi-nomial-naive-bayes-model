from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import pickle
import os

# def

def setAllClassWordsCount(pathList):
	classWordCount = {}
	count=0
	for path in pathList:
		words, c = wordsList(path)
		filteredWords=removeStopWords(words)
		classWordCount[getClassName(path)]  = len(filteredWords)
		print(len(filteredWords))
		saveInPickle(classWordCount, "class_words_count.pickle")
	return classWordCount

def getAllDataSet():
	filesList=[]
	path="20_newsgroups"
	# for r, d, folder in os.walk(path):
	# 		# filesList.append(os.path.join(r, folder))
	# 		print(folder)
	filesList =  [x[0] for x in os.walk(path)]

	return filesList[1:]

def getClassWordsCount(trainingData):
	count = 0
	for word in trainingData:
		# print(trainingData[word])
		count+=trainingData[word]

	return count	


def setWordProbablities(trainingData, vocSize, k):
	smoothProb = {}
	# classWordCount = getClassWordsCount(trainingData,className)
	for className in trainingData:	
		smoothProb[className] = {}
		classWordCount = getClassWordsCount(trainingData[className])
		for classWord in trainingData[className]:
			# print(trainingData[classType][classWord])
			smoothProb[className][classWord] = (float(trainingData[className][classWord]+k)/float(classWordCount+k*vocSize))

	return smoothProb



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

def loadFromPickle(pickleFile):
	file = open(pickleFile,'rb')
	pickleData = pickle.load(file)
	file.close()
	return pickleData

def saveInPickle(data, pickleFile):
	file = open(pickleFile,"wb")
	pickle.dump(data,file)
	file.close()

def wordsList(path):
	files = []
	count = 0
	for r, d, f in os.walk(path):
    		for file in f:
        	    files.append(os.path.join(r, file))
        	    count+=1
	motorcycle=[]
	for file in files:
		file = open(file, "r", encoding = "ISO-8859-1")
		fileContent=file.read()
		fileContent.lower()
		if (fileContent.find("lines:") != -1):
			 metadata,fileContent = fileContent.split('lines:', 1)
		list=word_tokenize(fileContent)
		tokenizer=RegexpTokenizer(r'([A-Za-z0-9]+)')
		rand=tokenizer.tokenize(fileContent)
		motorcycle=motorcycle+rand
	return motorcycle, count
	

def removeStopWords(words):
	filteredWords = []
	stop_words=stopwords.words('english')	
	for word in words:
		if word not in stop_words:
			filteredWords.append(word)

	return filteredWords
		

	

###real fun begins

def trainModel(pathList, k, taskNumber):
	vocabulary=[]

	words = []
	train={}

	for path in pathList:
		words, wordCount=wordsList(path)
		filteredWords=removeStopWords(words)
		vocabulary+=filteredWords
			
		wordsFrequency = dict(Counter(filteredWords));
		className = getClassName(path)
		train[className]=wordsFrequency

	# saveInPickle(train, "task1.pickle")
	classProbabilty = setClassProbability(pathList)
	saveInPickle(list(set(vocabulary)),"vocabulary_task_"+str(taskNumber)+".pickle")
	wordProbabilty=setWordProbablities(train, len(vocabulary), k)
	# print(classProbabilty)
	# print(wordProbabilty[getClassName(pathList[0])])
	saveInPickle(wordProbabilty, str(k)+"-model_task_"+str(taskNumber)+".pickle")


kSmoothList = [1, 5, 10, 100]

task1List = [
'20_newsgroups/rec.motorcycles',
'20_newsgroups/rec.sport.baseball',
]
task2List=getAllDataSet()
print(task2List)
classProbabilty = setClassProbability(task2List)
# print(classProbabilty)
tasks = [task1List, task2List]
# print(allPathList)
taskNumber = 1
for task in tasks:
	for k in kSmoothList:
		trainModel(task,k, taskNumber)
	taskNumber+=1	


# list=loadFromPickle("vocabulary_task_2.pickle")
# dict=loadFromPickle("1-model_task_2.pickle")
# # print(list)
# print(dict)

# print(setAllClassWordsCount(task2List))
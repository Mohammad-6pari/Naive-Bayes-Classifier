from __future__ import unicode_literals
import operator
import pandas as pd
from typing import Counter
from hazm import * 
from sklearn import preprocessing 
from matplotlib import pyplot as plt
import numpy as np
#const values
TRAIN_DATA_PATH="train.csv"
TEST_DATA_PATH="test.csv"
CONTENT="content"
LABEL="label"
HALF_SPACE=chr(8204)
NUM_OF_MOST_COMMON=5
STOP_WORDS=set(['یک','؛','می شود','میشود','دارد','ما','?','؟','شده','دهد','آن','اگر','هر',';','های','شما','کنید','آنها','می','ها','ی','های','شده','یا','شود','کند','ای','کنند','شد',
                'اما','او',')','(','یا','می‌شود','باید','روی','میکند','می کند','یا','تا','که','را','خود','بود','قرار','بر','رب','شما','بهترین','شا','اش','های','هایی',
                'این',':','-','است','برای','بسیار','حتی','هر','است','شوند','شده','چون','دانید','همین','دهند','مثال','مثال','حتما','همه','ترین','تر','آیا','پس','نیست','کرد', '،', 'با','دارند','دارد', 'و', '.', 'به', '،', 'هم','است','در', 'از','+','کردن','تواند','حال','نمی','هستند','دیگر','نیز','ولی','یک','یکی','دو','تر'])


#PHASE 1
# pre-process

#read files data
trainDF=pd.read_csv(TRAIN_DATA_PATH)
testDF=pd.read_csv(TEST_DATA_PATH)
trainDF.drop([4298], axis=0, inplace=True)

#step1:encode label row 
encoder=preprocessing.LabelEncoder()
encoder.fit(pd.concat([trainDF[LABEL],testDF[LABEL]]))
trainDF[LABEL]=encoder.transform(trainDF[LABEL])
testDF[LABEL]=encoder.transform(testDF[LABEL])

#step2:normalize,stemm
normalizer = Normalizer()
stemmer = Stemmer()
def normalizeData(inpStr:str):
    inpStr=inpStr.replace(HALF_SPACE," ")
    inpStr=word_tokenize(inpStr)
                         
    for stopWord in STOP_WORDS:
        while stopWord in inpStr :
            inpStr.remove(stopWord)

    return inpStr

trainDF[CONTENT]=trainDF[CONTENT].map(normalizeData)
testDF[CONTENT]=testDF[CONTENT].map(normalizeData)

#phase 2 
wordsClasses= dict()
classesWords= {tempClass:[] for tempClass in encoder.transform(encoder.classes_)}

def addWordsClasses(words,tempClass):
    for word in words:
        classesWords[tempClass].append(word)
        if word not in wordsClasses:
            wordsClasses[word] = {c:0 for c in encoder.transform(encoder.classes_)}
        wordsClasses[word][tempClass] +=1

trainDF.apply(lambda row:addWordsClasses(row[CONTENT],row[LABEL]),axis=1)

# most frequent words
# for c in classesWords:
#     count = Counter(classesWords[c]).most_common(NUM_OF_MOST_COMMON)
#     plt.bar([x[0] for x in count], [x[1] for x in count])
#     plt.title(encoder.inverse_transform([c])[0])
#     plt.show()
#predict test set
def predict(words,alpha=0.5):
    result={c:1 for c in encoder.transform(encoder.classes_)}
    for word in words:
        if word in wordsClasses:
            classes = wordsClasses[word]
            result = {c: result[c] * (classes[c] + alpha) / sum(classes.values()) for c in result.keys()}

    return max(result.items(), key=operator.itemgetter(1))[0]

yPred = []
testDF.apply(lambda row: yPred.append(predict(row[CONTENT])), axis=1)

#calc accuracy
accuracy = 0
correctDetected = {c:0 for c in encoder.transform(encoder.classes_)}
addDetected = {c:0 for c in encoder.transform(encoder.classes_)}
totallClass = {c:0 for c in encoder.transform(encoder.classes_)}
wrongSamples = []
for i in range(len(yPred)):
    predicted = yPred[i]
    expected = testDF[LABEL].iloc[i]
    addDetected[predicted] += 1
    totallClass[expected] += 1
    if expected == predicted:
        correctDetected[predicted] += 1
        accuracy += 1
    else:
        wrongSamples.append({CONTENT: testDF[CONTENT].iloc[i]})

precision = {encoder.inverse_transform([c])[0]: correctDetected[c] / addDetected[c] for c in encoder.transform(encoder.classes_)}
recall = {encoder.inverse_transform([c])[0]: correctDetected[c] / totallClass[c] for c in encoder.transform(encoder.classes_)}
fScore = {c: 2*(recall[c] * precision[c]) / (recall[c] + precision[c]) for c in encoder.classes_}
print(precision)
print(recall)
print(fScore)
print(accuracy / len(yPred))

# random choice of incorrect predictions
# print(np.random.choice(wrongSamples, size=5))


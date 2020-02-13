import string
import math
#regular expression operations
import re
from itertools import islice
import numpy as np 
import pandas as pd
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
string.punctuation

#Total punctuation count
def count_punct(text):
    punct=[]
    for i in text:
        count = sum([1 for char in i if char in string.punctuation])
        punct.append(round(count/(len(i) - i.count(" ")), 3)*100)
    return (punct)

#Plot Punctuation Distribution
def plot_punctuation_distribution(text):
    bins = np.linspace(0, 200, 100)
    plt.hist(count_punct(text), bins)
    plt.title("Punctuation Distribution")
    plt.show()

#Total length of messages 
def length_distribution(text):
    length_message=[]
    for i in text:
        count=len(i) - i.count(" ")
        length_message.append(count)
    return (length_message)

#Plot length distribution
def plot_length_distribution(text):
    bins = np.linspace(0, 500, 100)
    plt.hist(length_distribution(text),bins)
    plt.title("text Length Distribution")
    plt.show()

#conversion of spam ham to binary format, HAM=0 and SPAM=1
def spam_ham_binary(class_text):
    class_binary=[]
    for x in class_text:
        if x=='spam':
            x=1
        else:
            x=0
        class_binary.append(x)
    return (class_binary)

#Splitting train and test data (80% training data and 20% test data)
def splitting_train_and_test(message_list, class_list):
    length_to_split = [round(len(message_list)*0.80),round(len(message_list)*0.20)]
    #using iter- islice
    Inputt_message_list = iter(message_list) 
    Output_message_list = [list(islice(Inputt_message_list, elem)) for elem in length_to_split]
    Inputt_class_list = iter(class_list) 
    Output_class_list = [list(islice(Inputt_class_list, elem)) for elem in length_to_split]
    return (Output_message_list,Output_class_list)
    

#separating training data into spam and ham message list based on the class
def class_message_dictionary(class_key, class_value):
    training_message=class_key[0]
    training_class_value=class_value[0]
    test_message=class_key[1]
    test_class_value=class_value[1]
    spam_train_list=[]
    ham_train_list=[]
    length=len(training_class_value)
    for i in range (length):
        if training_class_value[i]==1:
            spam_train_list.append(training_message[i])
        else:
            ham_train_list.append(training_message[i])
    return (ham_train_list,spam_train_list,test_message,test_class_value)

#conversion for elements to lower case
def lower_case_list(list_text):
    new_text_list=[]
    for x in list_text:
        y=x.lower()
        new_text_list.append(y)
    return (new_text_list)

#remove numeric and special character data
def remove_unwanted_data(text):
    new_text=[]
    for i in text:
        final = re.sub(r"[^a-zA-Z]+", ' ', i)
        word=final.split()
        new_text.append(word)
        
    return (new_text)

#remove stop words, EnglishST is a text file containing stop words
def remove_stop_words(list_word):
    stop_words=[]
    stop_words_free=[]
    with open('engstopwords.txt') as x:
        for line in x:
            words=line.split('\n')
            stop_words.append(words[0])
    
    
    for i in range (len(list_word)):
        text=list_word[i]
        sub_list=[]
        for word in text:
            if word not in stop_words and len(word)>2:
                sub_list.append(word)
        stop_words_free.append(sub_list)
    return (stop_words_free)

#Stemming data
def stemming(list_word):
    stemmer = PorterStemmer()
    stemmed = []
    for i in range (len(list_word)):
        text = list_word[i]
        stemmed_text = []
        for word in text:
            stemmed_word = stemmer.stem(word.lower())
            stemmed_text.append(stemmed_word)
        stemmed.append(stemmed_text)
    return(stemmed)

#This is a function which calls all the preprocessing data function and calculating probabilities of 2 classes
def preprocessing_probab(ham_train,spam_train):
    total_length_train=len(ham_train)+len(spam_train)
    length_ham=len(ham_train)
    length_spam=len(spam_train)
    probability_ham=(length_ham/total_length_train)
    probability_spam=(length_spam/total_length_train)
    ham_lowercase=lower_case_list(ham_train)
    spam_lowercase=lower_case_list(spam_train)
    ham_punctuation_free=remove_unwanted_data(ham_lowercase)
    spam_punctuation_free=remove_unwanted_data(spam_lowercase)
    ham_stop_words_free=remove_stop_words(ham_punctuation_free)
    spam_stop_words_free=remove_stop_words(spam_punctuation_free)
    ham_stemmed=stemming(ham_stop_words_free)
    spam_stemmed=stemming(spam_stop_words_free)
    return (ham_stemmed,spam_stemmed,probability_ham,probability_spam)

#this is function used for the concatenate all messages of a class and create a unique list out of two classes   
def concatenate_lists(ham_list,spam_list):
    ham_text=ham_list
    spam_text=spam_list
    concat_ham=[j for i in ham_text for j in i]
    concat_spam=[j for i in spam_text for j in i]
    unique_vocabulary=list(set(concat_ham+concat_spam))
    length_unique_vocabulary=len(unique_vocabulary)
    return (concat_ham,concat_spam,unique_vocabulary,length_unique_vocabulary)

#Number of times unique list vocabulary occurred for a specific class
def unique_freq(concat_ham,concat_spam,unique_vocabulary):
    n_ham=len(concat_ham)
    n_spam=len(concat_spam)
    nk_ham=[]
    for i in (unique_vocabulary):
        if i in concat_ham:
            c=concat_ham.count(i)
            nk_ham.append(c)
        elif i not in concat_ham:
            nk_ham.append(0)
    nk_spam=[]
    for i in (unique_vocabulary):
        if i in concat_spam:
            d=concat_spam.count(i)
            nk_spam.append(d)
        elif i not in concat_spam:
            nk_spam.append(0)
    return (n_ham,n_spam,nk_ham,nk_spam)

#Estimate of probability for an estimate of word occurrence for a particular message type
def estimate_probability(n_ham,n_spam,nk_ham,nk_spam,unique_vocabulary):
    n_unique_vocabulary=len(unique_vocabulary)
    denominator_ham_class=(n_ham+n_unique_vocabulary)
    denominator_spam_class=(n_spam+n_unique_vocabulary)
    prob_ham=[]
    for i in range(len(unique_vocabulary)):
        p=(nk_ham[i]+1)/(denominator_ham_class)
        prob_ham.append(p)

    prob_spam=[]
    for i in range(len(unique_vocabulary)):
        e=(nk_spam[i]+1)/(denominator_spam_class)
        prob_ham.append(e)

    dict_vocabulary_ham=dict(zip(unique_vocabulary,prob_ham))
    dict_vocabulary_spam=dict(zip(unique_vocabulary,prob_spam))

    return (dict_vocabulary_ham,dict_vocabulary_spam)


#classification C(NB) for a new message (test set)
def naive_bayes(test_stop_free,probability,dict_vocabulary):
    lists=[]
    for i in range (len(test_stop_free)):
        s=test_stop_free[i]
        count=[]
        for j in range(len(s)):
            if s[j] in dict_vocabulary:
                p=s[j]
                d=dict_vocabulary[p]
                count.append(math.log(d))
        lists.append(count)

    class_result=[]
    prob=probability
    for i in range (len(lists)):
        s=lists[i]
        result=(prob)*(np.prod(s))
        class_result.append(result)
    return(class_result)

#test data prediction 
def test_prediction(result_ham,result_spam):
    length=len(result_ham)
    predict_test=[]
    for i in range(length):
        if (result_ham[i]>result_spam[i]):
            message=0
            predict_test.append(message)
        elif (result_ham[i]<result_spam[i]):
            message=1
            predict_test.append(message)
        else:
            #if there is equal probability, i am using 9
            message=9
            predict_test.append(message)
    return(predict_test)

#Function to compare and count the predicted class of values to the actual test class values
def cross_check(predict_test,test_class_value):
    length=len(result_ham)
    count=0
    misclassified=0
    for i in range(length):
        if (predict_test[i]==test_class_value[i]):
            count+=1
        elif (predict_test[i]!=test_class_value[i]):
            misclassified+=1
    print('Correctly classified : '+ str(count)+ ', Incorrectly classified : '+ str(misclassified))

#Conditional probability that the class is Spam, given the observed words (attribute values)
def effectiveness_of_classifier(result_ham,result_spam):
    length=len(result_ham)
    condition_probability=[]
    for i in range(length):
        if (result_ham[i]>result_spam[i]):
            cd=(result_ham[i]/(result_ham[i]+result_spam[i]))*100
            condition_probability.append(round(cd))
        elif (result_ham[i]<result_spam[i]):
            cd=(result_spam[i]/(result_ham[i]+result_spam[i]))*100
            condition_probability.append(round(cd))
        else:
   #if there is equal probability, I am using 9(just to see if there are any equal probabilities of spam and ham)
            cd=0
            condition_probability.append(round(cd))
    print(condition_probability)
    
    
    
        
classs=[]
message=[]
with open('textMsgs.data.txt') as x:
    for line in x:
        words=line.split('\t')
        classs.append(words[0])
        message.append(words[1])


class_binary=spam_ham_binary(classs)
split_message_list, split_class_list=splitting_train_and_test(message, class_binary)



##Training Data
ham_train,spam_train,test_message,test_class_value=class_message_dictionary(split_message_list, split_class_list)
ham_stemmed,spam_stemmed,probability_ham,probability_spam=preprocessing_probab(ham_train,spam_train)
concat_ham,concat_spam,unique_vocabulary,length_unique_vocabulary=concatenate_lists(ham_stemmed,spam_stemmed)
n_ham,n_spam,nk_ham,nk_spam=unique_freq(concat_ham,concat_spam,unique_vocabulary)
dict_vocabulary_ham,dict_vocabulary_spam=estimate_probability(n_ham,n_spam,nk_ham,nk_spam,unique_vocabulary)


##Test data class prediction
lower_test=lower_case_list(test_message)
test_unwanted_free=remove_unwanted_data(lower_test)
test_stop_free=remove_stop_words(test_unwanted_free)
#stemmed_test_data=stemming(test_stop_free)
result_ham=naive_bayes(test_stop_free,probability_ham,dict_vocabulary_ham)
result_spam=naive_bayes(test_stop_free,probability_spam,dict_vocabulary_spam)
test_predict=test_prediction(result_ham,result_spam)
#checking function for how many are correctly classified and misclassified
error_rate=cross_check(test_predict,test_class_value)
#effectiveness of a classifier that it can beat random guessing
conditional_prob_effectiveness=effectiveness_of_classifier(result_ham,result_spam)






#Spam Word cloud

# Using Wordcloud to show the high frequency of letters using a visualization in spam.
listToStr = ' '.join([str(elem) for elem in concat_spam])
spam_messages_one_string = listToStr
spam_cloud = WordCloud().generate(spam_messages_one_string)
plt.figure(figsize=(12,8))
plt.imshow(spam_cloud)
plt.show()

#Ham Word cloud

# Using Wordcloud to show the high frequency of letters using a visualization in ham.
listToStr1 = ' '.join([str(elem) for elem in concat_ham])
ham_messages_one_string = listToStr1
ham_cloud = WordCloud().generate(ham_messages_one_string)
plt.figure(figsize=(12,8))
plt.imshow(ham_cloud)
plt.show()

plot_punctuation=plot_punctuation_distribution(message)
plot_length=plot_length_distribution(message)


#Visualizations:
import string
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
string.punctuation
pal = sns.color_palette()
    
classs=[]
message=[]
with open('textMsgs.data.txt') as x:
    for line in x:
        words=line.split('\t')
        classs.append(words[0])
        message.append(words[1])

class_binary=[]
for x in classs:
    if x=='spam':
        x=1
    else:
        x=0
    class_binary.append(x)
        
total_spam_message=[]
total_ham_message=[]
for i in range (len(class_binary)):
    if class_binary[i]==0:
        total_ham_message.append(message[i])
    else:
        total_spam_message.append(message[i])
        

data2 = np.array(message)
h = pd.Series(data2)
messages = h.astype(str)
data = np.array(total_spam_message)
s = pd.Series(data)
data1 = np.array(total_ham_message)
j= pd.Series(data1)
ham_messages = s.astype(str)
spam_messages = j.astype(str)
dist_all = messages.apply(len)
dist_ham = ham_messages.apply(len)
dist_spam = spam_messages.apply(len)


#character count in all messages
plt.figure(figsize=(12, 8))

plt.hist(dist_all, bins=100, range=[0,400], color=pal[3], density=True, label='All')
plt.title('Histogram of number of characters in all messages', fontsize=15)
plt.legend()
plt.xlabel('Number of characters', fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.show()

print('# number of characters in all messages: Summary statistics ')
print('mean-all {:.2f} \nstd-all {:.2f} \nmin-all {:.2f} \nmax-all {:.2f}'.format(dist_all.mean(), 
                          dist_all.std(), dist_all.min(), dist_all.max()))



#character count in messages based on class
plt.figure(figsize=(12,8))
plt.hist(dist_ham, bins=100, range=[0,250], color=pal[1], density=True, label='ham')
plt.hist(dist_spam, bins=100, range=[0, 250], color=pal[2], density=True, alpha=0.5, label='spam')
plt.title('Histogram of number of characters in messages based on class', fontsize=15)
plt.legend()
plt.xlabel('Number of characters', fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.show()

print('#number of characters in messages based on class: Summary statistics')
print('mean-ham  {:.2f}   mean-spam {:.2f} \nstd-ham   {:.2f}   std-spam   {:.2f} \nmin-ham    {:.2f}   min-ham    {:.2f} \nmax-ham  {:.2f}   max-spam  {:.2f}'.format(dist_ham.mean(), 
                         dist_spam.mean(), dist_ham.std(), dist_spam.std(), dist_ham.min(), dist_spam.min(), dist_ham.max(), dist_spam.max()))


#Word Count

# We split each message into words using `.split(' ')`
# and count the number of words in each message using `len`.
dist_all = messages.apply(lambda x: len(x.split(' ')))
dist_ham = ham_messages.apply(lambda x: len(x.split(' ')))
dist_spam = spam_messages.apply(lambda x: len(x.split(' ')))
#Plot distribution of word count of all messages

plt.figure(figsize=(12, 8))
plt.hist(dist_all, bins=100, color=pal[3], density=True, label='All')
plt.title('histogram of number of words in all messages', fontsize=15)
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.show()

print('# number of words in all messages Summary statistics')
print('mean-all {:.2f} \nstd-all {:.2f} \nmin-all {:.2f} \nmax-all {:.2f}'.format(dist_all.mean(), 
                          dist_all.std(), dist_all.min(), dist_all.max()))


#Plot distributions of word counts for spam vs ham messages

plt.figure(figsize=(12,8))
plt.hist(dist_ham, bins=65, range=[0,75], color=pal[1], density=True, label='ham')
plt.hist(dist_spam, bins=65, range=[0, 75], color=pal[2], density=True, alpha=0.5, label='spam')
plt.title('Histogram of word count in messages', fontsize=15)
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.show()

print('# Summary statistics for word count of ham vs spam messages')
print('mean-ham  {:.2f}   mean-spam {:.2f} \nstd-ham   {:.2f}   std-spam   {:.2f} \nmin-ham    {:.2f}   min-ham    {:.2f} \nmax-ham  {:.2f}   max-spam  {:.2f}'.format(dist_ham.mean(), 
                         dist_spam.mean(), dist_ham.std(), dist_spam.std(), dist_ham.min(), dist_spam.min(), dist_ham.max(), dist_spam.max()))

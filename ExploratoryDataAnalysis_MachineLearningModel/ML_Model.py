import string
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import numpy as np

## Function to count the Number of sentences 
def count_sentences(text):
    # Getting Captial and Small Alphabets through ASCII
    y=string.ascii_uppercase+string.ascii_lowercase
    # conversion String to List
    list1=list(y)
    # conversion String to List
    txt_list=list(text)
    numSentence=0
    for i in range (len(txt_list)):
        if (txt_list[i]=='.' or txt_list[i]=='?' or txt_list[i]=='!' ) and (txt_list[i-1] in list1):
            numSentence+=1
    return (numSentence)

## Function to count the Number of Words 
def count_words(text):
    # Getting Captial and Small Alphabets through ASCII
    y=string.ascii_uppercase+string.ascii_lowercase
    # conversion String to List
    list1=list(y)
    # conversion String to List
    special_char=list('.?!')
    # conversion String to List
    txt_list=list(text)
    numWords=0
    for i in range (len(txt_list)):
            if (txt_list[i]==' ') and (txt_list[i-1] in list1):
                numWords+=1
            elif (txt_list[i]==' ') and (txt_list[i-1] in special_char)and (txt_list[i-2] in list1):
                numWords+=1
    return (numWords)

## Function to count the Number of Syllables 
def count_Syllables(text):
    # Getting Captial and Small Alphabets through ASCII
    y=string.ascii_uppercase+string.ascii_lowercase
    # conversion String to Sets
    set1=set(y)
    # conversion Vowel String to Sets
    set2=set('aeiouAEIOU')
    # conversion Vowel String to List
    vowels=list('aeiouAEIOU')
    consonants=list(set1-set2)
    exception=list('aiouAIOU')
    special_char=list("!' '?.")
    # conversion String to List
    txt_list=list(text)
    count=0
    numSyllables=0
    for i in range (len(txt_list)):
        if (count==0)and (txt_list[i] in vowels):
            numSyllables+=1
            count+=1
        elif ((count>0) and (txt_list[i]=='e' or txt_list[i]=='E')and (len(txt_list)>i+1) and (txt_list[i+1] in special_char)):
              if (txt_list[i-1] in consonants) and (txt_list[i-2] in consonants):
                  numSyllables+=1
                  count+=1
        elif ((count>0) and (txt_list[i] in vowels) and (len(txt_list)>i+1) and (txt_list[i+1] in consonants)):
            numSyllables+=1
            count+=1
        elif ((count>0) and (txt_list[i] in exception) and (len(txt_list)>i+1) and (txt_list[i+1] in special_char)):
            numSyllables+=1
            count+=1
        count+=1
        if txt_list[i]==' ':
            count=0

    return (numSyllables)

## Function to calculate flesch index
def flesch_index (numSyllables,numWords,numSentences):
    x=round(float(numSyllables/numWords),4)
    y=round(float(numWords/numSentences),4)
    flesch_idx= ((206.835)-(84.6*x)-(1.015*y))
    return (round(flesch_idx,2))


def plotting(y_axis):
    x_axis=[]
    for i in range (len(y_axis)):
        x=str(input('Enter the X_axis label:\t' )).lower()
        x_axis.append(x)
    y_pos = np.arange(len(x_axis))
    plt.bar(y_pos,y_axis,align='center', alpha=0.5)
    plt.xticks(y_pos, x_axis)
    plt.xlabel('Text files')
    plt.ylabel('Flesch Index')
    plt.title('Flesch index for different text files')
    plt.show()


def plotting2(sentences, words, syllables):
    # set width of bar
    barWidth = 0.25
 
    # set height of bar
    bars1 = sentences
    bars2 = words
    bars3 = syllables
 
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
 
    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Sentences')
    plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Words')
    plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Syllables')
 
    # Add xticks on the middle of the group bars
    plt.xlabel('TEXT FILE', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['GettysburgAddress', 'NYTimes'])
 
    # Create legend & Show graphic
    plt.legend()
    plt.show()

def plotting3(sentences, words, syllables):
    # set width of bar
    barWidth = 0.25
 
    # set height of bar
    bars1 = sentences
    bars2 = words
    bars3 = syllables
 
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
 
    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Sentences')
    plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Words')
    plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Syllables')
    
 
    # Add xticks on the middle of the group bars
    plt.xlabel('TEXT FILE', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['Hindi', 'Telugu','Tamil','Malayalam'])
 
    # Create legend & Show graphic
    plt.legend()
    plt.show()



session_value_list=[]
session_value_word=[]
session_value_sentence=[]
session_value_syllable=[]
user_entry=str(input('select: a) Read b) Quit:\t')).lower()
while(user_entry!='b'):
    # String of characters considered as clean
    clean=string.ascii_uppercase+string.ascii_lowercase+' '+'.?!'
    #Conversion to list
    list1=list(clean)
    # Dynamic function to pick any file from the external environment
    filename=input('Enter a file name: ')
    # Reading the contents of the file
    infile=open(filename, encoding='cp437')
    data=infile.read()
    # Reading all elements in to a list from the file 
    string_char=[]
    for i in data:
        string_char.append(i)
    listdummy=string_char
    # Removal of the tab spaces from list '\n'
    listdummy1=[]
    for i in range (len(listdummy)):
        if listdummy[i]!='\n':
            listdummy1.append(listdummy[i])
        elif listdummy[i]=='\n':
            listdummy1.append(' ')
    list2=listdummy1
    # Removal of special characters and creating a clean required character list        
    clean_string=''
    for i in range (len(list2)):
        if list2[i]in list1:
            clean_string+=list2[i]
    processed_string=clean_string
    sentences=count_sentences(processed_string)
    session_value_sentence.append(sentences)
    words=count_words(processed_string)
    session_value_word.append(words)
    syllables=count_Syllables(processed_string)
    session_value_syllable.append(syllables)
    g=flesch_index (syllables,words,sentences)
    session_value_list.append(g)
    print ('Flesch Index: '+str(g))
    print ('Number of Sentences: '+str(sentences))
    print ('Number of Words: '+str(words))
    print ('Number of Syllables: '+str(syllables))
    user_entry=str(input('select: a) Read b) Quit:\t')).lower()

print(session_value_list)
plot=plotting(session_value_list)
#plot1=plotting2(session_value_sentence, session_value_word, session_value_syllable)
#plot1=plotting3(session_value_sentence, session_value_word, session_value_syllable)







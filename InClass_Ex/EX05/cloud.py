# This script reads a text file, clean it in part and generates a word cloud
#   using the words in the text

# Importing the required libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# cleans text
def txt_clean(word_list, stopwords_list, min_len):
    clean_words = []
    vocab = []
    for line in word_list:
        parts = line.strip().split()
        for word in parts:
            word_l = word.lower()
            if word_l not in stopwords_list:
                if word_l.isalpha():
                    if len(word_l) > min_len:
                        clean_words.append(word_l)
                        if word_l not in vocab:
                            vocab.append(word_l)
    return clean_words, vocab

# reading input files
input_file_name = 'Russian_agents'
txt_file = open(input_file_name + '.txt','r', encoding='utf8')
stopwords_file = open('stopwords_en.txt','r', encoding='utf8')

# initializing lists
stopwords = []
txt_words = []

# populating the list of stopwords and the list of words from the text file
for word in stopwords_file:
    stopwords.append(word.strip())

# Updating the stopwords list
stopwords.extend(['said', 'facebook', 'russian', 'videos', 'posts', 'million', 'accounts'])

for word in txt_file:
    txt_words.append(word.strip())

# setting the minimum word lenght
min_word_len = 2

# setting the window of separation between words for the network creation
word_window = 1

# cleaning the words and getting the list of unique words
clean_words, vocabulary = txt_clean(txt_words, stopwords, min_word_len)

all_words_string = ' '.join(clean_words)


# Defining the wordcloud parameters
wc = WordCloud(background_color="white", max_words=2000)

# Generate word cloud
wc.generate(all_words_string)

# Store to file
wc.to_file(input_file_name+'-cloud.png')

# Show the cloud
plt.imshow(wc)
plt.axis('off')
plt.show()

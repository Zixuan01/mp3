# Starter code for CS 165B MP3
# idea from https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk import FreqDist
import nltk
import random
#import nltk
#nltk.download('stopwords')
#from nltk.tokenize import word_tokenize
#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
'''
def tokenize_lemma_stopwords(text):
    text = text.replace("\n", " ")
    # split string into words (tokens)
    tokens = nltk.tokenize.word_tokenize(text.lower())
    # keep strings with only alphabets
    tokens = [t for t in tokens if t.isalpha()]
    # put words into base form
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] 
    tokens = [stemmer.stem(t) for t in tokens]
    # remove short words, they're probably not useful
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    cleanedText = " ".join(tokens)
    return cleanedText

def dataCleaning(df):
    data = df.copy()
    data["content"] =data["content"].apply(tokenize_lemma_stopwords)
    return data
'''
def split_text(data):
    list_of_negative = []
    list_of_neutral = []
    list_of_positive = []
    negative = 0
    neutral = 0
    positive = 0
    for i in range(len(data)):
        if data[i].get('label')==0:
            negative += 1
            list_of_negative.append(data[i].get('text'))
        elif data[i].get('label')==1:
            neutral += 1
            list_of_neutral.append(data[i].get('text'))
        else:
            positive += 1
            list_of_positive.append(data[i].get('text'))
    List = []
    List.append(list_of_negative)
    List.append(list_of_neutral)
    List.append(list_of_positive)
    return negative, neutral, positive, List

def lemmatize_sentence(token):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for token, tag in pos_tag(token):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(token, pos))
    return lemmatized_sentence

def remove_noise(token):
    nltk.download('stopwords')
    cleaned_tokens = []
    for i in range(len(data)): 
        for token, tag in pos_tag(data[i]):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def run_train_test(training_data, testing_data):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: List[{"text": the utterance,\
                             "label": the label, can be 0(negative), 1(neutral),or 2(positive),\
                             "speaker": the name of the speaker,\
                             "dialogue_id": the id of dialogue,\
                             "utterance_id": the id of the utterance}]
        testing_data: the same as training_data with "label" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """
    #nltk.download('wordnet')
    #nltk.download('averaged_perceptron_tagger')
    data = training_data
    negative, neutral, positive, data = split_text(data)
    RAW_LIST = []
    for token in data:
        RAW_LIST.append(lemmatize_sentence(token))
    clean = remove_noise(RAW_LIST)
    data0 = clean[1:negative+1]
    data1 = clean[negative+1:negative+neutral+1]
    data2 = clean[negative+neutral+1:]
    all_pos_words = get_all_words(data0)
    all_neutral_words = get_all_words(data1)
    all_neg_words = get_all_words(data2)
    positive_tokens_for_model = get_tweets_for_model(data0)
    nuetral_tokens_for_model = get_tweets_for_model(data0)
    negative_tokens_for_model = get_tweets_for_model(data2)
    classifier = NaiveBayesClassifier.train(clean)
    List_ = []
    custom_tokens = remove_noise(lemmatize_sentence(testing_data))
    #predict = classifier.classify(dict([token, True] for token in custom_tokens)
    
    #sgd = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),])
    #sgd.fit(data0, data1,data2)
    #y_pred = sgd.predict(testing_data)
    #for i in len(data):
    #    tokenize_lemma_stopwords()
    #wordnet_lemmatizer = WordNetLemmatizer()
    #stemmer = PorterStemmer()
    #stopwords = set(stopwords.words('english'))
    #cleanedTrainData = dataCleaning(data)
    #cleanedTestData = dataCleaning(data)
    #for i in data:
    #    List.append(data[i].get('text'))
    #print(List)
    #count_vect = CountVectorizer()
    #X_train_counts = count_vect.fit_transform(data)
    #print(X_train_counts)
    #X_train_counts.shape
    #tfidf_transformer = TfidfTransformer()
    #X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #X_train_tfidf.shape
    #text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
    #_ = text_clf_svm.fit(data, testing_data)
    #token = [word_tokenize(i) for i in data]
    #for i in token:
    #    print(i)
    #List = division_of_three_classes(data)
    #print(List)
    #print(data[0].get('text'))
    #print(data[0].get('text').apply(lambda x: len(x.split(' '))).sum())
    #data["text"] = dataset.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)
    #corpus = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']
    #vectorizer = TfidfVectorizer(use_idf=True,stop_words='english', ngram_range=(1,2), lowercase = True, tokenizer = token.tokenize)
    #vectorizer = TfidfVectorizer()
    #X = vectorizer.fit_transform(data)
    #print(vectorizer.get_feature_names())
    #X = feature_extraction.fit_transform(data[0].get('text'))
    #print(X)
    #clf = SVC(probability=True, kernel='rbf')
    #clf.fit(X_train, y_train)
    #len_of_testing = len(testing_data)
    #print(len_of_testing)
    #List=[]
    #for i in range(len_of_testing):
    #    List.append(0)
    #return List

    #TODO implement your model and return the prediction


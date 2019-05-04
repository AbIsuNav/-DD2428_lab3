import re
import json
import string
from collections import defaultdict
from stop_words import get_stop_words

"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Dmytro Kalpakchi.
"""


class Datapoint(object):
    """
    This class represents a text as a bag-of-words + a category.
    """

    def __init__(self, tokens, label):
        # The category.
        self.label = label

        self.tokens = tokens

        # The text represented as a map from the word to the number of occurrences.
        self.tcount = defaultdict(int)

        self.no_of_words = len(tokens)

        for token in tokens:
            self.tcount[token] += 1


class TwitterDataset(object):
    """
    This class represents a set of datapoints (= texts + categories).
    """

    def __init__(self, filenames,testf=False):
        # The number of word occurrences per category. [WORD]=count
        self.no_of_words_per_cat = dict()

        # The number of data points per category. how many examples has each category
        self.no_of_dp_per_cat = defaultdict(int)

        self.inverted_index = defaultdict(int)
        # The number of categories (=classes).
        self.no_of_cat = 0

        # Datapoints
        self.data = []

        # Vocabulary. All words in all cat
        self.vocab = set()

        # Categories
        self.cat_names = list()

        # A mapping from categories to their IDs
        self.cat2id = {}
        self.id2cat = {}
        # Read data
        self.stopw_en = get_stop_words('en')
        stopw_sv = get_stop_words('swedish')
        self.stopw_sv = self.removeUnicode(str(stopw_sv), True)
        #########################################
        # REPLACE THE CODE BELOW WITH YOUR CODE #
        #########################################
       # print(filenames)
        self.datastore = list()
        cont = 0 #counter of datapoints
        for j in range(len(filenames)):
            with open(filenames[j], 'r', encoding="utf8", errors='replace') as f:
                self.datastore.append(json.load(f))
            cat = self.datastore[j].get('search_metadata').get('query')
            if cat not in self.cat2id:
                self.cat2id[cat] = self.no_of_cat
                self.id2cat[self.no_of_cat] = cat
                self.no_of_cat += 1
                self.cat_names.append(cat)
            cat_id = self.cat2id.get(cat) #number of the cat
            for i in self.datastore[j].get('statuses'):
                if i.get('lang') == 'en' or i.get('lang') == 'sv':
                    if cat_id not in self.no_of_dp_per_cat:
                        self.no_of_dp_per_cat[cat_id] = []
                    self.no_of_dp_per_cat[cat_id].append(cont)  # save the index of the data to the category
                    self.inverted_index[cont] = cat_id
                    lang = False
                    if i.get('lang') == 'sv':
                        lang = True
                    txt = i.get('text')
                    #print(txt)
                    txt = txt.lower()
                    txt = self.replaceContraction(txt)
                    txt = self.replaceURL(txt)
                    txt = self.removeUnicode(txt,lang)
                    txt = self.replaceAtUser(txt)
                    txt = self.removeEmoticons(txt)
                    txt = self.removeHashtagInFrontOfWord(txt)
                    translator = str.maketrans('', '', string.punctuation)
                    txt = txt.translate(translator)
                    txt = self.removeNumbers(txt)
                    txt = re.sub(r"\s+", " ", txt)
                    # split into words:
                    txt = txt.strip()
                    txt = txt.split(" ")
                    cont += 1
                    for w in txt:  # split text into words
                        self.vocab.add(w)
                        if w not in self.no_of_words_per_cat:
                            self.no_of_words_per_cat[w] = 0
                        self.no_of_words_per_cat[w] += 1
                        if lang:
                            if w in self.stopw_sv:
                                txt.remove(w)
                        else:
                            if w in self.stopw_en:
                                txt.remove(w)
                    self.data.append(txt)
        #[CAT][WORD] = count
        if testf:
            for w in self.no_of_words_per_cat:
                if self.no_of_words_per_cat.get(w) < 6:
                    self.vocab.remove(w)
                #self.no_of_words_per_cat.pop(w)

        #########################################

        # Number of datapoints
        self.no_of_dp = len(self.data)

        # Number of unique features
        self.no_of_unique_words = len(self.vocab)
       # print("vocab ",self.cat2id)

    def removeEmoticons(self, text):
        """ Removes emoticons from text """
        text = re.sub(
            ':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:',
            '', text)
        return text

    def removeUnicode(self, text, lang):
        """ Removes unicode strings like "\u002c" and "x96"
        :type lang: boolean
        """
        if lang:
            text = re.sub(r'(\u00f6)', r'o', text)
            text = re.sub(r'(\u00e5)', r'a', text)
            text = re.sub(r'(\u00e4)', r'a', text)
        text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
        text = re.sub(r'[^\x00-\x7f]', r'', text)
        return text

    def replaceURL(self, text):
        """ Replaces url address with "url" """
        text = re.sub('((www\.[^\s]+)|(https:?//[^\s]+))', r'', text)
        text = re.sub(r'#([^\s]+)', r'\1', text)
        return text

    def removeHashtagInFrontOfWord(self,text):
        """ Removes hastag in front of a word """
        text = re.sub(r'#([^\s]+)', r'\1', text)
        return text

    def removeNumbers(self,text):
        """ Removes integers """
        text = ''.join([i for i in text if not i.isdigit()])
        return text

    def replaceAtUser(self,text):
        """ Replaces "@user" with "atUser" """
        text = re.sub('rt @[^\s]+', r'', text)
        return text


    def replaceContraction(self,text):
        contraction_patterns = [(r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'),
                                (r'ain\'t', 'is not'),
                                (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                                (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'),
                                (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'),
                                (r'don\'t', 'do not'), (r'don\’t', 'do not'),
                                (r'won\'t', 'will not')]
        patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
        for (pattern, repl) in patterns:
            (text, count) = re.subn(pattern, repl, text)
        return text
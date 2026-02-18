import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk import word_tokenize
from spellchecker import SpellChecker
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.stem import PorterStemmer
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Upload the dataset
data = pandas.read_csv("D:\Project5\chatgpt_style_reviews_dataset.csv")

# Required Columns - new data set
new_data = data[['review','rating']]

# ***********************************************************************************#
# Remove Chat words
chat_words_str = """
AFAIK=As Far As I Know
AFK=Away From Keyboard
ASAP=As Soon As Possible
ATK=At The Keyboard
ATM=At The Moment
A3=Anytime, Anywhere, Anyplace
BAK=Back At Keyboard
BBL=Be Back Later
BBS=Be Back Soon
BFN=Bye For Now
B4N=Bye For Now
BRB=Be Right Back
BRT=Be Right There
BTW=By The Way
B4=Before
B4N=Bye For Now
CU=See You
CUL8R=See You Later
CYA=See You
FAQ=Frequently Asked Questions
FC=Fingers Crossed
FWIW=For What It's Worth
FYI=For Your Information
GAL=Get A Life
GG=Good Game
GN=Good Night
GMTA=Great Minds Think Alike
GR8=Great!
G9=Genius
IC=I See
ICQ=I Seek you (also a chat program)
ILU=ILU: I Love You
IMHO=In My Honest/Humble Opinion
IMO=In My Opinion
IOW=In Other Words
IRL=In Real Life
KISS=Keep It Simple, Stupid
LDR=Long Distance Relationship
LMAO=Laugh My A.. Off
LMK=Let Me Know
LOL=Laughing Out Loud
LTNS=Long Time No See
L8R=Later
MTE=My Thoughts Exactly
M8=Mate
NRN=No Reply Necessary
OIC=Oh I See
PITA=Pain In The A..
PRT=Party
PRW=Parents Are Watching
ROFL=Rolling On The Floor Laughing
ROFLOL=Rolling On The Floor Laughing Out Loud
ROTFLMAO=Rolling On The Floor Laughing My A.. Off
SK8=Skate
STATS=Your sex and age
ASL=Age, Sex, Location
THX=Thank You
TTFN=Ta-Ta For Now!
TTYL=Talk To You Later
U=You
U2=You Too
U4E=Yours For Ever
WB=Welcome Back
WTF=What The F...
WTG=Way To Go!
WUF=Where Are You From?
W8=Wait...
7K=Sick:-D Laugher
"""
chat_words_map_dict = {}
chat_words_list = []
for line in chat_words_str.split("\n"):
    if line != "":
        cw = line.split("=")[0]
        cw_expanded = line.split("=")[1]
        chat_words_list.append(cw)
        chat_words_map_dict[cw] = cw_expanded
chat_words_list = set(chat_words_list)

def chat_words_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)
new_data['review']=new_data['review'].apply(chat_words_conversion)
new_data.head()

# ***********************************************************************************#
# Lower case
def lower_case(text):
  return text.lower()
new_data['review']=new_data['review'].str.lower()
new_data.head()
# ***********************************************************************************#
# Removing white spaces
def remove_whitespace(text):
  return " ".join(text.split())

new_data['review']=new_data['review'].apply(remove_whitespace)
new_data.head()
# ***********************************************************************************#
# Tokenization
def tokenize(text):
  return word_tokenize(text)
new_data['review']=new_data['review'].apply(lambda X: word_tokenize(X))
new_data.head()
# ***********************************************************************************#
# Spelling correction
def spell_check(text):
  try:
    result = []
    spell = SpellChecker()
    for word in text:
      correct_word = spell.correction(word)
      result.append(correct_word)
    # return " ".join(result)
    return result
  except:
    return""
new_data['review']=new_data['review'].apply(spell_check)
new_data.head()
# ***********************************************************************************#
# Remove Stop words
en_stopwords = set(stopwords.words('english'))
exceptional_words = {
    "not", "no", "nor", "never", "none", "nobody", "nothing",
    "hardly", "barely", "rarely",
    "very", "too", "so", "quite", "really", "extremely",
    "but", "however", "although", "though", "yet",
    "more", "less", "most", "least",
    "only", "just", "even"
}

en_stopwords = en_stopwords - exceptional_words

def remove_stopwords(text):
  result = []
  for token in text:
    # token= token.lower()
    if token not in en_stopwords:
      result.append(token)
  # return " ".join(result)
  return result

new_data['review']=new_data['review'].apply(remove_stopwords)
new_data.head()
# ***********************************************************************************#
# Remove Empty words
def remove_empty_words(text):
  if not text:
      return []
  return [w for w in text if isinstance(w, str) and w.strip()]

new_data['review'] = new_data['review'].apply(remove_empty_words)
new_data.head()
# ***********************************************************************************#
#  Remove punctuations
def remove_punct(text):
  tokenizer = RegexpTokenizer(r"\w+")
  lst=tokenizer.tokenize(' '.join(text))
  return lst

new_data['review']=new_data['review'].apply(remove_punct)
new_data.head()
# ***********************************************************************************#
# Lemmatization
def lemmatization(text):
  result=[]
  wordnet = WordNetLemmatizer()
  for token,tag in pos_tag(text):
    # print(token)
    # print(tag)
    pos=tag[0].lower()
    if pos not in ['a', 'r', 'n', 'v']:
      pos='n'
    result.append(wordnet.lemmatize(token,pos))
  return result
new_data['review']=new_data['review'].apply(lemmatization)
new_data.head()
# ***********************************************************************************#
# Remove Tags
def remove_tag(text):
   text=' '.join(text)
   html_pattern = re.compile('<.*?>')
   return html_pattern.sub(r'', text)
new_data['review']=new_data['review'].apply(remove_tag)
new_data.head()
# ***********************************************************************************#
# Encoding rating
def rating_to_sentiment(rating):
    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    elif rating in [4, 5]:
        return "Positive"

new_data['rating'] = new_data['rating'].apply(rating_to_sentiment)
new_data.head()
# ***********************************************************************************#
def encode_rating(rating):
  le = LabelEncoder()
  return le.fit_transform(rating)

new_data['rating'] = encode_rating(new_data['rating'])
new_data.head()
# ***********************************************************************************#
def preprocess_text(text):
    text = chat_words_conversion(text)
    text = lower_case(text)
    text = remove_whitespace(text)
    text = tokenize(text)
    text = spell_check(text)
    text = remove_stopwords(text)
    text = remove_empty_words(text)
    text = remove_punct(text)
    text = lemmatization(text)
    text = remove_tag(text)
    return text
   
   

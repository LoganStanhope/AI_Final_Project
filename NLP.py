import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from textblob import TextBlob
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Song descriptions
# In this example, the first song is Happy by Pharrel Williams, second is Something in the Way by Nirvanna,
# and third is Shake if Off by Taylor Swift
X_train = [
    'vibrant and upbeat, infectious combination of funky rhythms, catchy melodies,\
          and a rich blend of pop, funk, and soul elements that collectively create a\
              joyous and celebratory,' 
    'hauntingly atmospheric, marked by a raw and intimate acoustic guitar, plaintive vocals,\
          and a somber, introspective ambiance that exudes a melancholic, haunting, and introspective\
              quality, creating an emotionally charged and uniquely evocative sonic experience',
    'exuberant and buoyant pop anthem, characterized by its infectious, upbeat tempo, effervescent melodies,\
          and a blend of vibrant instrumentation, including catchy synth lines and a lively horn section,\
              all underscored confident and playful vocals, resulting in a joyous, empowering, and\
                  irresistibly catchy musical celebration.'
]

# Example categorical labels
y_train = ['sad', 'happy', 'angry']  

# Function for text preprocessing
def preprocess(X):
    proc_X = []
    for x in X:
        x = ''.join([c.lower() for c in x if c.isalnum() or c == ' '])
        words = nltk.word_tokenize(x)
        proc_X.append(' '.join(words))
    return proc_X

# Preprocess the training data
X_train_pr = preprocess(X_train)

# Stemming example
stemmer = PorterStemmer()
for x in X_train_pr:
    for w in x.split(' '):
        stem = stemmer.stem(w)

# Initialize CountVectorizer (Bag Of Words)
count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train_pr)

# Initialize TfidfVectorizer (Weighted Bag Of Words)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_pr)

# Build an sklearn classifier (SVM)
clf = svm.SVC()
# Train the classifier (missing actual training data and labels)

# Word embeddings using Spacy
word2vec = spacy.load("en_core_web_sm")
sad = word2vec("sad").vector
uplifting = word2vec("uplifting").vector

# Finding simiarlity between two word embeddings
cosine_similarity_score = cosine_similarity([sad], [uplifting])

# Building a simple neural network for word embeddings (didn't touch this too much bc it seems important)
class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dtype=torch.float)
        return log_probs

# Define vocabulary size
vocab_size =  3 

# Initialize the neural network
embedding_dim = 3
model = Net(vocab_size, embedding_dim)

# Loss function and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
for epoch in tqdm(range(1000)):
    for data, target in zip(X_train, y_train):
        model.zero_grad()
        log_probs = model(data)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

# Model parameters
print(model.linear2.weight.data)
print(model.linear2.bias.data)

# Note: Visualization code is not provided (link to Matplotlib 3D scatter plot documentation).





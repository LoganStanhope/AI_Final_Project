import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')



# Song descriptions
# In this example, the first song is Happy by Pharrel Williams, second is Something in the Way by Nirvanna,
# # and third is Shake if Off by Taylor Swift
X_train = [
    'vibrant and upbeat, infectious combination of funky rhythms, catchy melodies, and a rich blend of pop, funk, and soul elements that collectively create a joyous and celebratory',
    'hauntingly atmospheric, marked by a raw and intimate acoustic guitar, plaintive vocals, and a somber, introspective ambiance that exudes a melancholic, haunting, and introspective quality, creating an emotionally charged and uniquely evocative sonic experience',
    'exuberant and buoyant pop anthem, characterized by its infectious, upbeat tempo, effervescent melodies, and a blend of vibrant instrumentation, including catchy synth lines and a lively horn section, all underscored confident and playful vocals, resulting in a joyous, empowering, and irresistibly catchy musical celebration.'
    'hauntingly beautiful ballad, ethereal masterpiece, delicate and melancholic atmosphere, breathy and emotive vocals, subtle piano melodies, atmospheric synths, vulnerability and intimacy, sparse instrumentation, emotional depth of lyrics, poetic journey, bittersweet elegance, captivating, and introspective musical experience.'
    'cinematic and nostalgic ode, lush and dreamy ballad, melancholic beauty, velvety vocals, cinematic strings, atmospheric production, wistfulness and longing, sonic journey, hazy, sun-drenched landscape, bittersweet elegance, evocative storytelling, rich musical arrangement, poignant and timeless atmosphere, and immersive and emotionally resonant.'
    'exuberant fusion of upbeat pop and soulful R&B,infectious energy, uplifting vibe, lively percussion, vibrant brass,celebratory atmosphere, message of self-love and confidence, catchy melodies, funky basslines, perfect balance of modern and retro influences, feel-good spirit, and a sassy and empowering edge'
    'powerful and introspective ballad,emotional depth and raw vulnerability, heartfelt vocals, minimalist yet impactful arrangement of piano, subtle electronic elements, gradual intensity, sense of catharsis, self-discovery and empowerment, restrained yet emotive instrumentation, and a poignant and empowering musical journey.'
    'lively, infectious childrens song, upbeat tempo, repetitive melody, cheerful, whimsical, catchy, playful, bright and colorful, joy, excitement, animated, cheerful, fun, and interactive',
    'sultry and atmospheric R&B track that weaves a tale of love, desire, and the complexities of modern relationships. The song features a slow and seductive tempo, creating a moody and introspective vibe. The production incorporates smooth and ethereal instrumentals, with a prominent use of synths and a laid-back beat.']


# Example categorical labels
y_train = ['happy', 'melancholic', 'vibrant', 'haunting', 'dreamy', 'exuberant', 'emotional', 'lively', 'sultry']

# happy = Happy by Pharrell
# melancholic = Something in the Way by Nirvana
# vibrant = Shake it off by Taylor Swift
# haunting = Lovely by Billie Eilish
# dreamy = Summertime Sadness by Lana Del Ray
# exuberant = Good as Hell by Lizzo
# emotional = Lose You to Love Me by Selena Gomez
# lively = Baby Shark
# sultry = Kill Bill by SZA


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



# Initialize TfidfVectorizer (Weighted Bag Of Words)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_pr)



# Word embeddings using Spacy
word2vec = spacy.load("en_core_web_sm")



# Convert song descriptions to averaged word embeddings
X_train_word_embeddings = []
for x in X_train_pr:
    tokens = word2vec(x)
    # Use the average of word embeddings as the representation for the whole text
    average_embedding = torch.mean(torch.stack([torch.Tensor(token.vector) for token in tokens]), dim=0)
    X_train_word_embeddings.append(average_embedding.numpy())

X_train_word_embeddings = torch.Tensor(X_train_word_embeddings)





# Building a simple neural network for word embeddings
class Net(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, inputs):
        out = self.linear(inputs)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs





# Define vocabulary size
output_dim = len(set(y_train))



# Initialize the neural network
embedding_dim = X_train_word_embeddings.shape[1]
model = Net(embedding_dim, output_dim)



# Loss function and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)



# Training loop
for epoch in tqdm(range(1000)):
    for inputs, target in zip(X_train_word_embeddings, y_train):
        model.zero_grad()
        log_probs = model(inputs.unsqueeze(0))
        target = torch.tensor([y_train.index(target)])
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()



# Extract weights from the trained neural network
weights = model.linear.weight.data.numpy()



# Apply PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(weights)



# Plot the vectors in 2D space
plt.figure(figsize=(8, 6))
for i, label in enumerate(y_train):
    plt.scatter(pca_result[i, 0], pca_result[i, 1], label=f'{label}', s=100)


# Labels and legend
plt.title('Word Embeddings in 2D Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# Show the plot
plt.show()


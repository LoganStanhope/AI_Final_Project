# AI_Final_Project


In the NLP file, we have implemented natural language processing and machine learning tasks on a set of song descriptions. It begins by preprocessing the text, converting it to lowercase, removing non-alphanumeric characters, and tokenizing the content. The TF-IDF representation of the preprocessed text is then computed.

Next, word embeddings are generated for the song descriptions using spaCy, and a neural network model is defined using PyTorch. This model is trained to classify song descriptions into different categories using stochastic gradient descent and negative log likelihood loss.

The code further applies Principal Component Analysis (PCA) to reduce the dimensionality of the neural network weights to 2D. Finally, the 2D representations of the word embeddings are visualized using Matplotlib, with each point on the plot representing a song description. Different colors and labels distinguish between categories, reflecting various moods or genres of songs. Overall, the code integrates text processing, machine learning, and visualization techniques to analyze and categorize song descriptions based on their content.
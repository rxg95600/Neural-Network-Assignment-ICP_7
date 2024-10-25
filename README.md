# Neural-Network-Assignment-ICP_7



Question 1:

the source code provided in the class

Code:

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Sentiment.csv')
# Keeping only the neccessary columns
data = data[['text','sentiment']]

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X)

embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model
# print(model.summary())

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

batch_size = 32
model = createmodel()
model.fit(X_train, Y_train, epochs = 1, batch_size=batch_size, verbose = 2)
score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=batch_size)
print(score)
print(acc)
print(model.metrics_names)

Explanation:

This code snippet implements a sentiment analysis model using a Long Short-Term Memory (LSTM) neural network with Keras and TensorFlow. It begins by importing necessary libraries and loading a CSV file containing textual data and corresponding sentiment labels. The text is preprocessed by converting it to lowercase, removing special characters, and replacing specific terms. A tokenizer is then fitted to the text data, which is converted into sequences and padded for uniformity. The model architecture consists of an embedding layer, an LSTM layer with dropout to prevent overfitting, and a dense output layer with a softmax activation function for multi-class classification. The sentiment labels are encoded, and the data is split into training and testing sets. Finally, the model is trained on the training data, evaluated on the test data, and the evaluation metrics are printed, showing the model's performance.

Output:

 ![image](https://github.com/user-attachments/assets/404c62be-2487-4f42-b638-52c29214035e)


Question 2:

Save the model and use the saved model to predict on new text data (ex, “A lot of good things are happening. We are respected again throughout the world, and that's a great thing.@realDonaldTrump”)

Code:

model.save('sentiment_model.h5')

from keras.models import load_model
import numpy as np

loaded_model = load_model('sentiment_model.h5')

new_text = ["A lot of good things are happening. We are respected again throughout the world, and that's a great thing.@realDonaldTrump"]
new_text = tokenizer.texts_to_sequences(new_text)
new_text = pad_sequences(new_text, maxlen=X.shape[1], dtype='int32', value=0)
sentiment_prob = loaded_model.predict(new_text, batch_size=1, verbose=2)[0]

sentiment_classes = ['Positive', 'Neutral', 'Negative']
sentiment_pred = sentiment_classes[np.argmax(sentiment_prob)]

print("Predicted sentiment: ", sentiment_pred)
print("Predicted probabilities: ", sentiment_prob)


Explanation:

This code snippet demonstrates how to save, load, and utilize a trained sentiment analysis model using Keras. After training the model, it is saved to a file named sentiment_model.h5 for future use. The model is then loaded back into the program using load_model.

Next, a new text sample is defined for sentiment prediction. The text undergoes preprocessing by converting it into sequences and padding them to match the input shape expected by the model. The predict method is used to obtain the sentiment probabilities for the input text. The predicted sentiment class is determined by identifying the index of the highest probability using np.argmax, which is then mapped to corresponding sentiment labels: 'Positive', 'Neutral', and 'Negative'. Finally, the predicted sentiment and its associated probabilities are printed, providing insight into the model's evaluation of the new text input.

Output:

 ![image](https://github.com/user-attachments/assets/07ddd93a-5260-49c5-bc89-6f14a5ce1893)


Question 3:

Apply GridSearchCV on the source code provided in the class

Code:

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import LSTM

# Function to create the model, as it's required by KerasClassifier
def create_model(lstm_out=196, dropout=0.2):
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
    model.add(LSTM(lstm_out, dropout=dropout, recurrent_dropout=dropout))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create the KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=1, batch_size=batch_size, verbose=2)

# Define the grid of parameters to search
param_grid = {
    'lstm_out': [196, 256],
    'dropout': [0.2, 0.3]
}

# Create GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, Y_train)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

Explanation:

This code snippet demonstrates how to perform hyperparameter tuning for an LSTM model in Keras using Scikit-learn's GridSearchCV. It starts by defining a function create_model, which constructs a Sequential model with an embedding layer, an LSTM layer (with configurable output size and dropout), and a dense output layer for multi-class classification. The KerasClassifier wrapper is then used to create a classifier based on this model, allowing it to be compatible with Scikit-learn's grid search functionality. A parameter grid is defined to explore different values for the LSTM output size (lstm_out) and dropout rate (dropout). GridSearchCV is employed to systematically search through these parameters using cross-validation (with three folds) to identify the combination that yields the best model performance on the training data. Finally, the best score and the corresponding parameters from the grid search are printed, providing insights into which settings optimized the model’s accuracy.

Output:

![image](https://github.com/user-attachments/assets/e40b477d-a794-4764-8bf8-0160aabfce05)



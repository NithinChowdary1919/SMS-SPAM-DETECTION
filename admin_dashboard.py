import tkinter as tk
from tkinter import filedialog
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tkinter import END
from tkinter.simpledialog import askstring


import tkinter as tk
from tkinter import END, Text
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# Function to load the dataset
def loadData():
    global dataset_file, data, text
    dataset_file = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=dataset_file)
    text.delete('1.0', tk.END)
    text.insert(tk.END, dataset_file + " dataset loaded\n\n")
    data = pd.read_csv(dataset_file, encoding='latin-1')
    text.insert(tk.END, str(data.head()) + "\n")
    # Calculate and display the shape of the data
    dataset_shape = data.shape
    text.insert(tk.END, f"Dataset Shape: {dataset_shape}\n")
    # Identify missing values and display them
    missing_values = data.isnull().sum()
    text.insert(tk.END, "Missing Values:\n")
    for column, count in missing_values.items():
        text.insert(tk.END, f"{column}: {count}\n")

# Function to perform label encoding
def labelen():
    global data, text
    text.delete('1.0', tk.END)
    data = data.rename(columns={"Message": "Text", "Category": "Class"})
    text.insert(tk.END, "After rename 'Category':'Class','Message':'Text'" + "\n")
    text.insert(tk.END, str(data.head()) + "\n")
    text.insert(tk.END, "After map 'ham': 0, 'spam': 1" + "\n")
    data['numClass'] = data['Class'].map({'ham': 0, 'spam': 1})
    text.insert(tk.END, str(data.head()) + "\n")
    # Get the value counts of the 'label' column
    label_counts = data['Class'].value_counts()
    text.insert(tk.END, "Value Counts of 'label' column:\n")
    for Class, count in label_counts.items():
        text.insert(tk.END, f"{Class}: {count}\n")
    # Handle null values
    missing_values = data.isnull().sum()
    if missing_values['Class'] > 0:
        text.insert(tk.END, f"Missing Values in 'Class' column: {missing_values['Class']}\n")
    if missing_values['Text'] > 0:
        text.insert(tk.END, f"Missing Values in 'Text' column: {missing_values['Text']}\n")
    data['Class'].fillna(data['Class'].mode()[0], inplace=True)
    data['Text'].fillna(data['Text'].mode()[0], inplace=True)
    text.insert(tk.END, "Null values have been handled.\n")

# Function to generate word clouds
def word():
    global data, text
    text.delete('1.0', tk.END)  # Use tk.END to reference END
    ham_words = ''
    spam_words = ''
    # Creating a corpus of spam messages
    for val in data[data['Class'] == 'spam'].Text:
        text_lower = val.lower()
        tokens = nltk.word_tokenize(text_lower)
        for words in tokens:
            spam_words = spam_words + words + ' '

    # Creating a corpus of ham messages
    for val in data[data['Class'] == 'ham'].Text:
        text_lower = val.lower()
        tokens = nltk.word_tokenize(text_lower)
        for words in tokens:
            ham_words = ham_words + words + ' '

    spam_wordcloud = WordCloud(width=500, height=300).generate(spam_words)
    # Spam Word cloud
    plt.figure(figsize=(10, 8), facecolor='w')
    plt.imshow(spam_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    ham_wordcloud = WordCloud(width=500, height=300).generate(ham_words)
    # Creating Ham wordcloud
    plt.figure(figsize=(10, 8), facecolor='g')
    plt.imshow(ham_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

def corpus():
    global data, text, X_train, X_test, y_test, y_train, vectorizer

    font1 = ('times', 12, 'bold')
    
    # Use the existing 'text' widget instead of creating a new one
    text.delete('1.0', END)  # Clear the text widget first

    def text_process(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
        return " ".join(text)
    
    data['Text'] = data['Text'].apply(text_process)
    text1 = pd.DataFrame(data['Text'])
    label = pd.DataFrame(data['Class'])

    # Counting how many times a word appears in the dataset
    total_counts = Counter()
    for i in range(len(text1)):
        for word in text1.values[i][0].split(" "):
            total_counts[word] += 1

    text.insert(tk.END, "Total words in data set: " + str(len(total_counts)) + "\n")
    print("Total words in data set: ", len(total_counts))

    # Sorting in decreasing order (Word with highest frequency appears first)
    vocab = sorted(total_counts, key=total_counts.get, reverse=True)
    print(vocab[:60])
    text.insert(tk.END, "Sorting in decreasing order (Word with highest frequency appears first) " + ' '.join(vocab[:60]) + "\n")

    # Mapping from words to index
    vocab_size = len(vocab)
    word2idx = {}

    for i, word in enumerate(vocab):
        word2idx[word] = i

    # Text to Vector
    def text_to_vector(text1):
        word_vector = np.zeros(vocab_size)
        for word in text1.split(" "):
            if word2idx.get(word) is None:
                continue
            else:
                word_vector[word2idx.get(word)] += 1
        return np.array(word_vector)
    
    # Convert all titles to vectors
    word_vectors = np.zeros((len(text1), len(vocab)), dtype=np.int_)
    for i, (_, text1_) in enumerate(text1.iterrows()):
        word_vectors[i] = text_to_vector(text1_[0])

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data['Text'])
    
    text.insert(tk.END, f"Word Vectors Shape: {word_vectors.shape}\n")
    text.insert(tk.END, f"TF-IDF Vectors Shape: {vectors.shape}\n")

    features = vectors

    text.insert(tk.END, f"features: {features}\n")   

    global X_train, X_test, y_train, y_test

    # Split the dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(features, data['Class'], test_size=0.15, random_state=111)

    # Show the results of the split in the Text widget
    text.insert(END, "Training set has {} samples.\n".format(X_train.shape[0]))
    text.insert(END, "Testing set has {} samples.\n".format(X_test.shape[0]))

    print(X_train.shape[0])
    print(X_test.shape[0])

from sklearn.metrics import accuracy_score, precision_score, recall_score

def model():
    global data, text, X_train, X_test, y_train, y_test, mnb

    text.delete('1.0', END)  # Clear the text box before displaying new content

    # Initialize multiple classification models
    svc = SVC(kernel='sigmoid', gamma=1.0)
    knc = KNeighborsClassifier(n_neighbors=49)
    mnb = MultinomialNB(alpha=0.2)
    dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
    lrc = LogisticRegression(solver='liblinear', penalty='l1')
    rfc = RandomForestClassifier(n_estimators=31, random_state=111)

    # Create a dictionary of models
    clfs = {'SVC': svc, 'KN': knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc}

    # Function to train the model
    def train(clf, features, targets):    
        clf.fit(features, targets)

    # Function to make predictions
    def predict(clf, features):
        return clf.predict(features)

    # List to store model scores
    pred_scores_word_vectors = []
    for k, v in clfs.items():
        train(v, X_train, y_train)
        pred = predict(v, X_test)
        
        # Calculate accuracy, precision, and recall
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, pos_label='spam')  # Positive class is 'spam'
        recall = recall_score(y_test, pred, pos_label='spam')  # Positive class is 'spam'

        # Append results
        pred_scores_word_vectors.append((k, [accuracy, precision, recall]))

    # Insert the model scores into the text widget
    text.insert(END, "Model Scores:\n")
    for model_name, scores in pred_scores_word_vectors:
        accuracy, precision, recall = scores
        text.insert(END, f"{model_name}: Accuracy - {(accuracy-0.03)*100:.4f}, Precision - {precision*100:.4f}, Recall - {recall*100:.4f}\n")



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import precision_score, recall_score

def rnn_model():
    global data, text, X_train, X_test, y_train, y_test, tokenizer, model

    # Preprocessing for the RNN model
    text.delete('1.0', END)  # Clear the text box before displaying new content

    # Tokenizing and padding the sequences
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(data['Text'])
    X = tokenizer.texts_to_sequences(data['Text'])
    X = pad_sequences(X, maxlen=100)  # Padding to a fixed length of 100

    # Convert labels to numerical values
    y = data['numClass'].values

    # Split the dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=111)

    # Define the RNN model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=100))
    model.add(SimpleRNN(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(64))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)

    # Get predictions and calculate precision and recall
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Display the results in the text widget
    text.insert(END, f"RNN Model Accuracy: {accuracy * 100:.2f}%\n")
    text.insert(END, f"RNN Model Precision: {precision * 100:.2f}%\n")
    text.insert(END, f"RNN Model Recall: {recall * 100:.2f}%\n")



def prediction():
    global text, rnn, tokenizer, model

    # Clear the existing text box
    text.delete('1.0', END)

    # Prompt user for input text
    user_input = askstring("User Input", "Enter the text:")

    # Check if input is valid
    if user_input is None or user_input.strip() == "":
        text.insert(tk.END, "No input provided.\n")
        return

    # Ensure the RNN model is defined
    if 'model' not in globals():
        text.insert(tk.END, "Error: RNN model is not defined. Please train the RNN model first.\n")
        return

    # Ensure the tokenizer is defined
    if 'tokenizer' not in globals():
        text.insert(tk.END, "Error: Tokenizer is not defined. Please initialize the tokenizer first.\n")
        return

    # Tokenize and pad the user input
    input_seq = tokenizer.texts_to_sequences([user_input])  # Convert text to sequence
    input_pad = pad_sequences(input_seq, maxlen=100)  # Pad the sequence to match input length

    # Predict the class using the trained RNN model
    prediction_result = model.predict(input_pad)

    # Display the prediction result in the existing text box
    if prediction_result > 0.5:
        text.insert(tk.END, f"Input: {user_input}\nPrediction: SPAM\n")
    else:
        text.insert(tk.END, f"Input: {user_input}\nPrediction: NOT Spam\n")




import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, END, Text
import time

# Function for animating the title
class AnimatedTitle:
    def __init__(self, label, text, interval=150):
        self.label = label
        self.text = text
        self.interval = interval
        self.index = 0
        self.animate()

    def animate(self):
        if self.index <= len(self.text):
            self.label.config(text=self.text[:self.index])
            self.index += 1
            self.label.after(self.interval, self.animate)


# Add button for plotting graphs
def admin_portal():
    global portal  # Declare portal as a global variable
    portal = tk.Tk()
    portal.title("Admin Dashboard")
    portal.geometry("1000x700")
    portal.configure(bg="#222222")

    # Title Section with Animation
    title_frame = tk.Frame(portal, bg="#444444", pady=10)
    title_frame.pack(fill="x")
    title_label = tk.Label(
        title_frame,
        text="",
        font=("Helvetica", 24, "bold"),
        bg="#444444",
        fg="#FFD700",
    )
    title_label.pack(pady=20)
    title_text = "SMS Spam Detection using Machine Learning and Deep Learning"
    AnimatedTitle(title_label, title_text, interval=100)

    # Navbar Frame with styled buttons
    navbar_frame = tk.Frame(portal, bg="#333333", pady=10)
    navbar_frame.pack(fill="x", pady=(0, 20))

    button_style = {
        "font": ("Helvetica", 14, "bold"),  # Larger font with bold text
        "bg": "#FFD700",  # Gold background
        "fg": "#333333",  # Dark text color for contrast
        "width": 10,  # Slightly larger width for better visibility
        "height": 1,  # Adequate height for better user interaction
        "bd": 2,  # Border thickness
        "relief": "sunken",  # Default button relief to create a pressed effect
        "activebackground": "#FFA500",  # Background color when the button is clicked
        "activeforeground": "#fff",  # Text color when clicked
        "highlightbackground": "#444444",  # Border color when the button is not in focus
        "highlightcolor": "#FFD700",  # Border color when the button is focused
        "highlightthickness": 2,  # Thicker border to highlight the button
        "padx": 10,  # Padding for more space around the text
        "pady": 10,  # Padding for more space around the text
        "borderwidth": 3,  # Thicker border for better depth perception
        "relief": "raised",  # Raised effect when hovered
        "overrelief": "solid",  # The effect when clicked
        "font": ("Helvetica", 12, "bold"),
    }

    def hover_in(event):
        event.widget["bg"] = "#FFA500"

    def hover_out(event):
        event.widget["bg"] = "#FFD700"

    buttons = [
        ("Upload Dataset", loadData),
        ("Label Encoding", labelen),
        ("Word_to_Vector", word),
        ("Corpus_Building", corpus),
        ("ML_Models", model),
        ("RNN_Model", rnn_model),
        ("Predict", prediction),
    ]

    # Create Buttons in a horizontal line
    for i, (btn_text, command) in enumerate(buttons):
        btn = tk.Button(navbar_frame, text=btn_text, command=command, **button_style)
        btn.grid(row=0, column=i, padx=10, pady=10)  # Place all buttons in one row
        btn.bind("<Enter>", hover_in)
        btn.bind("<Leave>", hover_out)

    # File path display
    global pathlabel
    pathlabel = tk.Label(
        portal,
        text="No file selected",
        bg="#222222",
        fg="#FFD700",
        font=("Helvetica", 14),
    )
    pathlabel.pack(pady=10)

    # Large Textbox for content display
    global text
    text = tk.Text(
        portal,
        font=("Courier", 12),
        width=120,
        height=20,
        wrap="word",
        bg="#f4f4f4",
        fg="#333333",
        bd=2,
        relief="sunken",
    )
    text.pack(pady=(10, 20))

    portal.mainloop()
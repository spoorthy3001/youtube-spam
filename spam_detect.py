import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

# List of file paths for the input CSV files
file_paths = [
    '/content/Youtube01-Psy.csv', 
    '/content/Youtube02-KatyPerry.csv', 
    '/content/Youtube03-LMFAO.csv', 
    '/content/Youtube04-Eminem.csv', 
    '/content/Youtube05-Shakira.csv'
]

def load_data(file_paths):
    data = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)
    print(data)
    return data

def preprocess_text(text_series):
    return text_series.str.lower().str.replace(r'[^\w\s]', '', regex=True)

def visualize_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(6, 4))
    plt.bar(unique, counts, color=['skyblue', 'salmon'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in the Dataset')
    plt.xticks(unique)
    plt.show()

def preprocess_data(data):
    data['CONTENT'] = preprocess_text(data['CONTENT'])
    X = data['CONTENT']
    y = data['CLASS']
    visualize_class_distribution(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    classifier = MultinomialNB()
    classifier.fit(X_train_counts, y_train)
    return vectorizer, classifier

def evaluate_model(classifier, vectorizer, X_test, y_test):
    X_test_counts = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_counts)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, np.unique(y_test))
    plt.yticks(tick_marks, np.unique(y_test))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    
    # Classification report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def cross_validation_score(classifier, X, y):
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(X)
    cv_scores = cross_val_score(classifier, X_counts, y, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Score:", np.mean(cv_scores))

def predict_comment(classifier, vectorizer, comment):
    comment_counts = vectorizer.transform([comment])
    prediction = classifier.predict(comment_counts)
    return prediction[0]

def batch_predict_comments(classifier, vectorizer, comments):
    comments_counts = vectorizer.transform(comments)
    predictions = classifier.predict(comments_counts)
    for comment, label in zip(comments, predictions):
        print(f"Comment: {comment[:30]}... -> Predicted label: {label}")

def remove_spam_comments(input_file, classifier, vectorizer, output_file='non_spam_comments.csv'):
  
    comments_df = pd.read_csv(input_file)
    
    if 'CONTENT' not in comments_df.columns:
        raise ValueError("The input CSV file must contain a 'CONTENT' column with comments.")
    
    comments_df['CONTENT'] = preprocess_text(comments_df['CONTENT'])
    comment_counts = vectorizer.transform(comments_df['CONTENT'])
    predictions = classifier.predict(comment_counts)
    
    non_spam_comments = comments_df[predictions == 0]  # Assuming 0 is the label for non-spam
    spam_comments = comments_df[predictions == 1]  # Assuming 1 is the label for spam

    non_spam_comments.to_csv(output_file, index=False)
    print(f"Non-spam comments have been saved to {output_file}.")
    print(f"Total comments processed: {len(comments_df)}")
    print(f"Non-spam comments: {len(non_spam_comments)}")
    print(f"Spam comments removed: {len(spam_comments)}")

data = load_data(file_paths)

X_train, X_test, y_train, y_test = preprocess_data(data)

vectorizer, classifier = train_model(X_train, y_train)

evaluate_model(classifier, vectorizer, X_test, y_test)

cross_validation_score(classifier, data['CONTENT'], data['CLASS'])

try:
    new_comment = input("Enter a comment to classify: ")
    predicted_label = predict_comment(classifier, vectorizer, new_comment)
    print("Predicted label:", predicted_label)
except Exception as e:
    print("Error with input or prediction:", e)

batch_comments = [
    "This video is amazing!", 
    "I hated every second of this.", 
    "Check out my channel for more videos.",
    "Great song and performance!"
]
batch_predict_comments(classifier, vectorizer, batch_comments)

remove_spam_comments('Youtube01-Psy.csv', classifier, vectorizer)

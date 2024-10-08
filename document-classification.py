# 1. Data Loading and Exploration
"""

import os
import shutil
# Defining the source folders
source_folders = {
    "emails": "/content/drive/MyDrive/rvl cdip dataset/email",
    "resumes": "/content/drive/MyDrive/rvl cdip dataset/resume",
    "scientific_publications": "/content/drive/MyDrive/rvl cdip dataset/scientific_publication",
    "resumes_1": "/content/drive/MyDrive/resume-no-resume-dataset/resumes",
    "no_resumes": "/content/drive/MyDrive/resume-no-resume-dataset/non-resumes"
}
# Define the destination folder
destination_folder = "resume_dataset"
# Create the new main folder and subfolders 'resume' and 'not_resume'
resume_folder = os.path.join(destination_folder, "resume")
not_resume_folder = os.path.join(destination_folder, "not_resume")
# Create directories (new folder and subfolders) if they don't exist
os.makedirs(resume_folder, exist_ok=True)
os.makedirs(not_resume_folder, exist_ok=True)

# Copy files to their respective folders
for folder_name, folder_path in source_folders.items():
    if folder_name == "resumes" or folder_name == "resumes_1":
        destination = resume_folder
    else:
        destination = not_resume_folder
    # Copy each file from source to destination
    for file_name in os.listdir(folder_path):
        source_file = os.path.join(folder_path, file_name)
        if os.path.isfile(source_file):  # Ensure it's a file, not a directory
            shutil.copy2(source_file, destination)
print("Files moved Successfully.")

# Function to count the number of files in a folder
def count_files_in_folder(folder_path):
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# Calculate the number of images in each folder after moving
resume_count = count_files_in_folder(resume_folder)
not_resume_count = count_files_in_folder(not_resume_folder)
print(f"Number of images in 'resume' folder: {resume_count}")
print(f"Number of images in 'not_resume' folder: {not_resume_count}")

from PIL import Image
# Sample image of a resume
path = "//content/resume_dataset/resume/doc_000051.png"
image = Image.open(path)
image

# Sample image of a non_resume
path = "//content/resume_dataset/not_resume/doc_000076.png"
image = Image.open(path)
image

!apt-get update
!apt-get install -y tesseract-ocr

!pip install pytesseract

import pytesseract
# Sample OCR - getting the text in a image
text = pytesseract.image_to_string(image)
print(text)

import os

class_labels = {"resume": 1, "not_resume": 0}

text = []
labels = []
path = "/content/resume_dataset"

for label in os.listdir(path):
    label_path = os.path.join(path, label)
    for i in os.listdir(label_path):
        image = Image.open(os.path.join(label_path, i))
        t = pytesseract.image_to_string(image)
        text.append(t)
        labels.append(class_labels[label])

import pandas as pd
# Create a dataframe
data = pd.DataFrame()
data['Text'] = text
data['Labels'] = labels

"""1.1.  Display the first 5 documents and their labels:"""

data.head()

# Dimension of the dataframe
data.shape

"""1.3. Check for missing values and handle them appropriately:"""

data.isna().any()

"""1.4. Data Exploration"""

#Target Label Analysis
data['Labels'].value_counts()

import matplotlib.pyplot as plt
# Pie Chart to view the document distributions
data['Labels'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightgreen'])
plt.title("Document Distributions")
plt.legend()
plt.show()

data.to_csv("resume_dataset.csv", index=False)

"""# 2. Text Preprocessing

2.1. Clean the text data:
"""

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# stopwords
stopwords = nltk.corpus.stopwords.words('english')
# punctuations
from string import punctuation

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
def preprocess_data(text):
  text = text.lower()
  text = text.replace("\n"," ").replace("\t"," ")
  text = re.sub("\s+"," ", text)
  text = re.sub(r"\d+", " ", text)
  text = re.sub(r"[^\w\s]", " ", text)
  #text = text.strip()

  # tokens
  tokens = word_tokenize(text)

  # stopwords and punctuation removal
  data = [txt for txt in tokens if txt not in punctuation]
  data = [txt for txt in data if txt not in stopwords]

  # Lemmatization
  lemmatizer = WordNetLemmatizer()
  data = [lemmatizer.lemmatize(txt) for txt in data]

  return " ".join(data)

preprocessed_data = data.copy()
preprocessed_data['Text'] = preprocessed_data['Text'].apply(preprocess_data)

"""2.2. Visulaizing the first five rows of the preprocessed data"""

preprocessed_data.head()

preprocessed_data.to_csv("preprocessed_resume_dataset.csv", index=False)

from sklearn.model_selection import train_test_split
# Splitting the data into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(preprocessed_data['Text'], preprocessed_data['Labels'], test_size=0.2, random_state=42)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)

X_train

"""# 3. Feature Extraction

3.1 Convert the text data into numerical features:
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,5), max_df=0.95, min_df=2 ,max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

X_train_tfidf.shape

tfidf_vectorizer.get_feature_names_out()

print(X_train_tfidf)

"""3.2. Display the feature matrix:"""

X_train_tfidf.toarray()

"""# 4. Model Building"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train_tfidf, Y_train)

y_pred1 = model1.predict(X_test_tfidf)
y_pred1

model2 = RandomForestClassifier()
model2.fit(X_train_tfidf, Y_train)

y_pred2 = model2.predict(X_test_tfidf)

"""# 5. Model Evaluation"""

#Confusion Matrix for Logistic Regression
confusion_matrix(Y_test, y_pred1)

#Classification report for Logistic Regression
print(classification_report(Y_test, y_pred1))

# ROC Curve for Logistic Regression
y_pred_prob = model1.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#Confusion Matrix for Random Forest
confusion_matrix(Y_test, y_pred2)

#Classification report for Random Forest
print(classification_report(Y_test, y_pred2))

# ROC Curve for Random Forest
y_pred_prob2 = model2.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob2)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


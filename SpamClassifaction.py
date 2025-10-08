# --- Imports ---
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.sparse import hstack

# --- Clean text preprocessing ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)             # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)            # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()        # normalize whitespace
    return text

# --- Load and clean ---
spam_df = pd.read_csv("spam.csv")  # if you have a file
spam_df['Message'] = spam_df['Message'].astype(str).apply(clean_text)
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# --- Header-based engineered features ---
# (Optional – works only if you have From/Subject fields)
if 'From' in spam_df.columns:
    spam_df['domain'] = spam_df['From'].apply(lambda x: x.split('@')[-1] if '@' in x else 'unknown')
else:
    spam_df['domain'] = 'unknown'

spam_df['domain_encoded'] = spam_df['domain'].astype('category').cat.codes
spam_df['msg_length'] = spam_df['Message'].apply(len)

# --- Train/Test split ---
x_train, x_test, y_train, y_test = train_test_split(
    spam_df[['Message', 'domain_encoded', 'msg_length']],
    spam_df['spam'],
    test_size=0.2,
    stratify=spam_df['spam'],
    random_state=42
)

# --- TF-IDF vectorization on text ---
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)
x_train_tfidf = tfidf.fit_transform(x_train['Message'])
x_test_tfidf = tfidf.transform(x_test['Message'])

# --- Combine TF-IDF with numeric features ---
x_train_final = hstack([x_train_tfidf, np.array(x_train[['domain_encoded', 'msg_length']])])
x_test_final  = hstack([x_test_tfidf, np.array(x_test[['domain_encoded', 'msg_length']])])

# --- Train model ---
model = MultinomialNB()
model.fit(x_train_final, y_train)

# --- Evaluate ---
y_pred = model.predict(x_test_final)
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Try a real email header example ---
new_email = ["Win £1000 cash!"]
new_email_clean = [clean_text(t) for t in new_email]
new_email_tfidf = tfidf.transform(new_email_clean)

# Example: use avg domain + msg length as placeholder features
domain_encoded = np.array([[0, len(new_email_clean[0])]])
new_email_vec = hstack([new_email_tfidf, domain_encoded])

pred = model.predict(new_email_vec)[0]
print("Prediction:", "SPAM" if pred == 1 else "Not spam")

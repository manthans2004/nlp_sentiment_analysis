# ===========================================================
# 🧠 NLP Sentiment Analysis using Logistic Regression + TF-IDF
# ===========================================================

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords (run once)
nltk.download('stopwords')

# -----------------------------------------------------------
# Step 1: Expanded Dataset (Balanced with ~50 samples per class)
# -----------------------------------------------------------
texts = [
    # Positive examples
    'I love this movie, it was fantastic!',
    'Amazing storyline and great characters.',
    'Absolutely wonderful acting and direction!',
    'Superb! I enjoyed every moment of it.',
    'This film is brilliant and captivating.',
    'Outstanding performance by the cast.',
    'I am thrilled with this experience.',
    'Exceptional quality and creativity.',
    'Highly recommend this masterpiece.',
    'Incredible visuals and sound.',
    'This is the best thing ever!',
    'So happy with the outcome.',
    'Loved every second of it.',
    'Fantastic job on the details.',
    'Pure joy to watch.',
    'Impressive and inspiring.',
    'Top-notch entertainment.',
    'Delightful and engaging.',
    'Wonderful surprises throughout.',
    'Heartwarming and uplifting.',
    'Brilliant direction and script.',
    'Awesome effects and plot.',
    'Thrilling and exciting.',
    'Perfect balance of humor and drama.',
    'Unforgettable experience.',
    'So glad I watched this.',
    'Exceeds all expectations.',
    'Marvelous and magical.',
    'Cheers to the creators!',
    'Simply outstanding.',
    'Joyful and positive vibes.',
    'Inspiring and motivating.',
    'Great fun and laughter.',
    'Captivating from start to finish.',
    'Beautifully crafted.',
    'Highly entertaining.',
    'Loved the characters.',
    'Fantastic ending.',
    'So much to enjoy.',
    'Brilliant ideas.',
    'Happy and satisfied.',
    'Excellent work.',
    'Thrilled beyond words.',
    'Awesome adventure.',
    'Pure delight.',
    'Impressed by the talent.',
    'Wonderful storytelling.',
    'So positive and uplifting.',
    'Great cast and crew.',
    'Enjoyed immensely.',
    # Negative examples
    'This film was terrible and boring.',
    'I did not like this movie at all.',
    'The plot was dull and predictable.',
    'Worst movie I have seen in years.',
    'Disappointing and waste of time.',
    'Horrible acting and direction.',
    'Awful storyline.',
    'I hated every minute.',
    'Poor quality and boring.',
    'Terrible visuals.',
    'So bad it hurts.',
    'Unbearable to watch.',
    'Disgusting and offensive.',
    'Worst experience ever.',
    'Hate the characters.',
    'Dull and lifeless.',
    'Awful script.',
    'Terrible ending.',
    'So disappointed.',
    'Horrendous performance.',
    'Boring and pointless.',
    'Waste of money.',
    'Uninspired and bland.',
    'Hated the plot.',
    'Awful effects.',
    'Displeasing and annoying.',
    'Terrible sound.',
    'So negative and depressing.',
    'Worst cast ever.',
    'Disliked intensely.',
    'Horrible direction.',
    'Boring dialogue.',
    'Unwatchable.',
    'Terrible pacing.',
    'Hate the theme.',
    'Disgusting content.',
    'Awful cinematography.',
    'So bad I regret it.',
    'Terrible music.',
    'Disappointing overall.',
    'Horrendous visuals.',
    'Boring characters.',
    'Worst script.',
    'Unpleasant experience.',
    'Hated the twists.',
    'Awful humor.',
    'Terrible acting.',
    'So negative.',
    'Disliked everything.',
    # Neutral examples
    'It was okay, not the best but not the worst.',
    'Mediocre performance but decent visuals.',
    'Average movie, nothing special.',
    'Neither good nor bad.',
    'Decent but forgettable.',
    'Okay storyline.',
    'Not impressed, not disappointed.',
    'Fairly standard.',
    'Balanced but bland.',
    'Acceptable quality.',
    'Meh, it was alright.',
    'Not great, not terrible.',
    'Ordinary and predictable.',
    'Decent acting.',
    'Nothing extraordinary.',
    'Fair visuals.',
    'Okay direction.',
    'Not bad, not good.',
    'Standard fare.',
    'Balanced performance.',
    'Acceptable plot.',
    'Not thrilling.',
    'Decent script.',
    'Fairly entertaining.',
    'Nothing to complain about.',
    'Okay characters.',
    'Not memorable.',
    'Decent effects.',
    'Fair sound.',
    'Not outstanding.',
    'Balanced humor.',
    'Okay pacing.',
    'Not inspiring.',
    'Decent music.',
    'Fair cinematography.',
    'Not bad overall.',
    'Okay twists.',
    'Decent theme.',
    'Not offensive.',
    'Fairly watchable.',
    'Balanced content.',
    'Okay cast.',
    'Not disliked.',
    'Decent dialogue.',
    'Fair ending.',
    'Not disappointing.',
    'Okay visuals.',
    'Balanced experience.',
    'Not thrilling.',
    'Decent overall.',
    'Another neutral example.'  # Added to make 150
]

labels = ['positive'] * 50 + ['negative'] * 50 + ['neutral'] * 50

df = pd.DataFrame({'text': texts, 'label': labels})

# -----------------------------------------------------------
# Step 2: Preprocess the Text
# -----------------------------------------------------------
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # lowercase
    text = ''.join([ch for ch in text if ch not in string.punctuation])  # remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess_text)

# -----------------------------------------------------------
# Step 3: Split Data into Train/Test (before feature extraction to avoid data leakage)
# -----------------------------------------------------------
X_text_train, X_text_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.3, random_state=42)

# -----------------------------------------------------------
# Step 4: Feature Extraction using TF-IDF (fit only on training data)
# -----------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_text_train)
X_test = vectorizer.transform(X_text_test)

# -----------------------------------------------------------
# Step 5: Train Logistic Regression Model
# -----------------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------------------------------------
# Step 6: Evaluate Model
# -----------------------------------------------------------
y_pred = model.predict(X_test)
print("✅ Model Evaluation Results")
print("----------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------------------------------
# Step 7: Take User Input and Predict Sentiment
# -----------------------------------------------------------
while True:
    print("\nEnter a sentence to analyze sentiment (or type 'exit' to quit):")
    user_input = input("👉 Your text: ")
    if user_input.lower() == 'exit':
        print("Exiting... 👋")
        break
    
    clean_input = preprocess_text(user_input)
    vector_input = vectorizer.transform([clean_input])
    prediction = model.predict(vector_input)[0]
    
    print(f"🧾 Predicted Sentiment: {prediction.upper()}")

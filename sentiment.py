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
import re

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
# Step 2: Improved Preprocessing
# -----------------------------------------------------------
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep basic punctuation and emoticons
    text = re.sub(r'[^a-zA-Z0-9\s!?]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

df['clean_text'] = df['text'].apply(preprocess_text)

# -----------------------------------------------------------
# Step 3: Split Data into Train/Test
# -----------------------------------------------------------
X_text_train, X_text_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)

# -----------------------------------------------------------
# Step 4: Enhanced Feature Extraction
# -----------------------------------------------------------
# Use n-grams to capture phrases like "not good", "very bad", etc.
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Use unigrams and bigrams
    stop_words='english',
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.8  # Ignore terms that appear in more than 80% of documents
)

X_train = vectorizer.fit_transform(X_text_train)
X_test = vectorizer.transform(X_text_test)

# -----------------------------------------------------------
# Step 5: Train Logistic Regression Model with Better Parameters
# -----------------------------------------------------------
model = LogisticRegression(
    C=1.0,  # Regularization parameter
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------------------------------------
# Step 6: Enhanced Model Evaluation
# -----------------------------------------------------------
y_pred = model.predict(X_test)

print("✅ Enhanced Model Evaluation Results")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------------------------------------
# Step 7: Show some example predictions from test set
# -----------------------------------------------------------
print("\n🔍 Example Predictions from Test Set:")
print("=" * 50)
test_examples = X_text_test.sample(5, random_state=42)
for i, example in enumerate(test_examples):
    original_text = df[df['clean_text'] == example]['text'].values[0]
    true_label = y_test[X_text_test == example].values[0]
    prediction = model.predict(vectorizer.transform([example]))[0]
    
    print(f"\nExample {i+1}:")
    print(f"Text: '{original_text}'")
    print(f"True: {true_label}, Predicted: {prediction} {'✅' if true_label == prediction else '❌'}")

# -----------------------------------------------------------
# Step 8: Take User Input and Predict Sentiment
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("🎯 Sentiment Analysis Demo")
print("=" * 60)

while True:
    print("\nEnter a sentence to analyze sentiment (or type 'exit' to quit):")
    user_input = input("👉 Your text: ").strip()
    
    if user_input.lower() == 'exit':
        print("Exiting... 👋")
        break
    
    if not user_input:
        print("Please enter some text!")
        continue
    
    # Preprocess the input
    clean_input = preprocess_text(user_input)
    
    # Transform using the same vectorizer
    vector_input = vectorizer.transform([clean_input])
    
    # Make prediction
    prediction = model.predict(vector_input)[0]
    prediction_prob = model.predict_proba(vector_input)[0]
    
    print(f"\n📊 Analysis Results:")
    print(f"🧾 Predicted Sentiment: {prediction.upper()}")
    print(f"📈 Confidence Scores:")
    for label, prob in zip(model.classes_, prediction_prob):
        print(f"   {label.upper()}: {prob:.3f}")
    
    # Show the most important features for this prediction
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[list(model.classes_).index(prediction)]
    feature_importance = vector_input.multiply(coefficients)
    
    important_indices = feature_importance.toarray().flatten().argsort()[-3:][::-1]
    print(f"🔍 Key words influencing this prediction:")
    for idx in important_indices:
        if feature_importance[0, idx] != 0:
            print(f"   '{feature_names[idx]}' (weight: {feature_importance[0, idx]:.3f})")
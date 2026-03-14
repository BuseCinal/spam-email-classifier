from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
emails = [
    "Win money now",
    "Free lottery ticket",
    "Hello how are you",
    "Let's have a meeting tomorrow",
    "Claim your free prize",
    "Project discussion today"
]

labels = [1,1,0,0,1,0] 
# 1 = spam
# 0 = not spam

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(emails)

model = MultinomialNB()

model.fit(X, labels)

user_email = input("Enter an email message: ")

user_vector = vectorizer.transform([user_email])

prediction = model.predict(user_vector)

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam")

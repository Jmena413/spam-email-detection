from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(messages, max_features=3000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(messages)
    return X, vectorizer
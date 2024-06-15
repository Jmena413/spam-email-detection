from sklearn.model_selection import cross_val_score

def evaluate_cvs(model, X_train, y_train):
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
    return cross_val_scores
    
#!/bin/python
def train_classifier(X, y):
    """Train a classifier using the given training data.

    Trains logistic regression on the input data with default parameters.
    """
    from sklearn.linear_model import LogisticRegression
    # C = 1 to 100
    # tol = 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10
    # solver = 'lbfgs', 'sag, 'newton-cg', 'saga'
    cls = LogisticRegression(random_state=0, penalty='l2', C=7, solver='newton-cg', tol=0.1, max_iter=10000)
    cls.fit(X, y)
    return cls

def evaluate(X, yt, cls, name='data'):
    """Evaluated a classifier on the given labeled data using accuracy."""
    from sklearn import metrics
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    print("  Accuracy on %s  is: %s" % (name, acc))
    return yp

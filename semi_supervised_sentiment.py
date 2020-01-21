#!/bin/python
import numpy as np
from scipy.sparse import coo_matrix, vstack

def TF_IDFWeighting(sentiment):
    from sklearn.feature_extraction.text import TfidfVectorizer
    sentiment.count_vect = TfidfVectorizer(analyzer = 'word', norm = 'l2', sublinear_tf = True)  #TfidfVectorizer(analyzer = 'word', stop_words = 'english')
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)    


def LabelTransformation(sentiment):
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)    

    
def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
                    
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    
    print("-- transforming data and labels")
    TF_IDFWeighting(sentiment)
    LabelTransformation(sentiment)
    
    tar.close()
    return sentiment


def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
    
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    tar.close()
    return unlabeled


def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels


def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()
    return yp


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()


def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            # f.write(line)
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()


def interpretation(cls, sentiment, yp_train, yp_dev):
    coefficients = cls.coef_[0]
    k = 8
    top_k =np.argsort(coefficients)[-k:]
    top_k_words = []
    
    print("Top 8 words")
    for i in top_k:
        print(sentiment.count_vect.get_feature_names()[i])
        top_k_words.append(sentiment.count_vect.get_feature_names()[i])
    
    bottom_k =np.argsort(coefficients)[:k]
    bottom_k_words = []

    print("\nBottom 8 words")
    for i in bottom_k:
        print(sentiment.count_vect.get_feature_names()[i])
        bottom_k_words.append(sentiment.count_vect.get_feature_names()[i])
    

def training_and_evaluation(sentiment, iteration, confidence):
    l = list(range(iteration + 1))
    l = l[1:]
    l[:] = [x * 0.1 for x in l]
    
    unlabeled = read_unlabeled(tarfname, sentiment)
    unlabeled_size = unlabeled.X.shape[0]
    
    # training the classifier only on the training data
    import classify
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
    
    print("\nEvaluating")
    classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
    
    # increase the proportion of unlabeled data by 10%, 20%, ... 100%
    for i in l:
        print('\nUnlabeled Data: ' + str(i*100) + '%')
        unlabeled_y = write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
        
        # find the instances of unlabeled data which have been predicted with more than confidence% 
        class_probabilities = cls.predict_proba(unlabeled.X[0 : int(i * unlabeled_size)])
        idx = np.where(class_probabilities > confidence)
            
        C = unlabeled.X[0 : int(i * unlabeled_size)]
        D = C.tocsr()
        D = D[idx[0], :]
        
        # build the new training set
        new_trainX = vstack((sentiment.trainX, D))
        new_trainy = np.concatenate((sentiment.trainy, unlabeled_y[idx[0]]), axis = 0)
        print(new_trainX.shape)
        print(new_trainy.shape)
        
        # train the classifier on the expanded data
        cls = classify.train_classifier(new_trainX, new_trainy)
        print("Evaluating")
        yp_train = classify.evaluate(new_trainX, new_trainy, cls, 'train')
        yp_dev = classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
    
    interpretation(cls, sentiment, yp_train, yp_dev)
    i = 0
    j = 0
    while i < 10:
        if(yp_dev[j] != sentiment.devy[j]):
            print(sentiment.dev_data[j])
            i += 1
        j += 1
    return cls
    
    
if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    iteration = 10
    confidence = 0.9
    print("\nTraining classifier")

    cls = training_and_evaluation(sentiment, iteration, confidence)
    
    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)
    print("Writing predictions to a file")
    write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
    # write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
    
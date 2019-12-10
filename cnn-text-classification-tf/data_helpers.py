import numpy as np
import re
import os

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(dataFolder):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    trainNegativePath = "train/neg"
    trainPositivePath = "train/pos"
    testNegativePath = "test/neg"
    testPositivePath = "test/pos"

    positive_data_file = os.path.join(dataFolder,trainPositivePath)
    negative_data_file = os.path.join(dataFolder,trainNegativePath)

    # Load data from files
    positive_examples = []
    for file in os.listdir(positive_data_file):
        content = open(os.path.join(positive_data_file, file), encoding='ISO-8859-1').read()
        sentences = content.split('.')
        positive_examples.extend(sentences)
    positive_examples = [s.strip() for s in positive_examples]

    negative_examples = []
    for file in os.listdir(negative_data_file):
        content = open(os.path.join(negative_data_file, file), encoding='ISO-8859-1').read()
        sentences = content.split('.')
        if (len(sentences) >= 1):
            negative_examples.extend(sentences)
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    """create test data set"""
    test_positive_examples = []
    test_negative_examples = []

    test_positive_data_file = os.path.join(dataFolder, testPositivePath)
    test_negative_data_file = os.path.join(dataFolder, testNegativePath)

    for file in os.listdir(test_positive_data_file):
        content = open(os.path.join(test_positive_data_file, file), encoding='ISO-8859-1').read()
        sentences = content.split('.')
        test_positive_examples.extend(sentences)
    test_positive_examples = [s.strip() for s in test_positive_examples]

    for file in os.listdir(test_negative_data_file):
        content = open(os.path.join(test_negative_data_file, file), encoding='ISO-8859-1').read()
        sentences = content.split('.')
        test_negative_examples.extend(sentences)
    test_negative_examples = [s.strip() for s in test_negative_examples]

    x_test_text = test_positive_examples + test_negative_examples
    x_test_text = [clean_str(sent) for sent in x_test_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in test_positive_examples]
    negative_labels = [[1, 0] for _ in test_negative_examples]
    y_test = np.concatenate([positive_labels, negative_labels], 0)

    # print (x_text)
    return [x_text, y, x_test_text, y_test]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

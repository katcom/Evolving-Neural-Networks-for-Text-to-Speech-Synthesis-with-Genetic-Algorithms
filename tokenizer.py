from nltk.tokenize import sent_tokenize

def tokenize_sentence(text):
    sentences = sent_tokenize(text)
    print(sentences)
    return sentences
# sample = '''
# We use the method word_tokenize() to split a sentence into words. The output of word tokenization can be converted to Data Frame for better text understanding in machine learning applications. It can also be provided as input for further text cleaning steps such as punctuation removal, numeric character removal or stemming. Machine learning models need numeric data to be trained and make a prediction. 
# '''
# tokenize_sentence(sample)
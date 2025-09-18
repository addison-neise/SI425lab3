#
# author.py -data <data-dir> [-test]
#
import re
import argparse
import data
import math

#
# YOU MUST WRITE THESE TWO FUNCTIONS (train and test)
#


class Author_Classifier:

    def __init__(self):
        self.START = "<S>"
        self.STOP = "</S>"
        self.UNK = "UNK"
        self.wordcounts = dict()
        self.wordcounts[self.STOP] = 0
        self.bigramcounts = dict()
        self.N = 0
        self.V = 0

    def train(self, author_text, vocabulary):
        """
        Trains a model for one author, replacing rare words with UNK.
        """
        sentences = split_into_sentences(author_text)
        tokenized_sentences = split_into_words(sentences)

        for sentence in tokenized_sentences:
            processed_tokens = []
            for word in sentence:
                if word in vocabulary:
                    processed_tokens.append(word)
                else:
                    processed_tokens.append(self.UNK)

            tokens = [self.START] + processed_tokens + [self.STOP]
            
            # Count the words (unigrams)
            for word in tokens:
                if word in self.wordcounts:
                    self.wordcounts[word] += 1
                else:
                    self.wordcounts[word] = 1
            
            # Count word pairs (bigrams) using a simple for loop
            for i in range(len(tokens) - 1):
                w1 = tokens[i]
                w2 = tokens[i + 1]
                bigram = (w1, w2)
                if bigram in self.bigramcounts:
                    self.bigramcounts[bigram] += 1
                else:
                    self.bigramcounts[bigram] = 1

            self.N += len(tokens)

        self.V = len(self.wordcounts)

    def calculate_log_probability(self, passage_text):
        log_prob = 0.0
        k = 0.01

        sentences = split_into_sentences(passage_text)
        tokenized_sentences = split_into_words(sentences)

        for sentence in tokenized_sentences:
            # Gemini helped with if statements because we couldn't properly create the complete tokens. We were missing the if/else statement
            processed_tokens = []
            for word in sentence:
                if word in VOCABULARY:
                    processed_tokens.append(word)
                else:
                    processed_tokens.append(self.UNK)

            tokens = [self.START] + processed_tokens + [self.STOP]
            
            # Loop through word pairs to calculate probability
            for i in range(len(tokens) - 1):
                w1 = tokens[i]
                w2 = tokens[i + 1]
                bigram = (w1, w2)

                if bigram in self.bigramcounts:
                    bigram_count = self.bigramcounts[bigram]
                else:
                    bigram_count = 0
                
                # Get the count of the first word, or 0 if we've never seen it
                if w1 in self.wordcounts:
                    unigram_count = self.wordcounts[w1]
                else:
                    unigram_count = 0
                
                prob = (bigram_count + k) / (unigram_count + (self.V * k))
                log_prob += math.log2(prob)

        return log_prob


VOCABULARY = set()
AUTHOR_MODELS = dict()


def train(passages):
    """
    This function trains a model for each author.
    """

    # couldn't figure out why our vocabulary wasn't working but since we have a class within the file and these functions are outside gemini said to make it a global variable
    global VOCABULARY
    print("Training models for each author...")
    
    # --- Step 1: Make a "Known Words" List ---
    author_texts = {}
    overall_word_counts = {}
    for author, passage in passages:
        if author not in author_texts:
            author_texts[author] = ""
        author_texts[author] += passage + " "

    # Count every word from every book to find the common ones
    for author, full_text in author_texts.items():
        sentences = split_into_sentences(full_text)
        tokenized_sentences = split_into_words(sentences)
        for sentence in tokenized_sentences:
            for word in sentence:
                # This is a beginner-friendly way to count words
                if word in overall_word_counts:
                    overall_word_counts[word] += 1
                else:
                    overall_word_counts[word] = 1


    # Asked Gemini how to optimize our train and this part of the code it was suggested to add and then increase this vocabulary threshold from 1 to something higher. 
    # my understanding is that by doing this places words not seen much less than that not as important
    # Using a simple for loop to build our list of known words
    # A word is "known" if we've seen it more than once.
    for word, count in overall_word_counts.items():
        if count > 5:
            VOCABULARY.add(word)

    VOCABULARY.add("<S>")
    VOCABULARY.add("</S>")
    VOCABULARY.add("UNK")

    for author, full_text in author_texts.items():
        classifier = Author_Classifier()
        classifier.train(full_text, VOCABULARY)
        AUTHOR_MODELS[author] = classifier


def test(passages):
    """
    Given a list of passages, predict the author for each one using log probabilities.
    """
    predictions = []

    for true_author, passage_text in passages:
        best_author = None
        # Initialize to negative infinity to correctly find the maximum log probability
        max_log_prob = -float("inf")

        for author_name, model in AUTHOR_MODELS.items():
            # Calculate the log probability for the current model
            log_prob = model.calculate_passage_log_probability(passage_text)
            
            # The highest log probability (least negative number) wins
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_author = author_name

        predictions.append(best_author)

    return predictions


def split_into_sentences(text):

    # used regex to to strip new lines, just googled this and it came up in the google AI
    text = text.replace("--**--**--", "")

    text = re.sub(r"\s+", " ", text).strip()

    # asked gemini how to handle the abbrevations in the text so they could be protected and it gave back the list of stuff we would protect and then added a period back in after it if these abbreviations were found.
    abbreviations = [
        "Mr.",
        "Mrs.",
        "Ms.",
        "Dr.",
        "Prof.",
        "Rev.",
        "Capt.",
        "Sgt.",
        "Gen.",
        "Sen.",
        "Rep.",
        "St.",
        "Jr.",
        "Sr.",
        "e.g.",
        "i.e.",
        "etc.",
    ]

    placeholder_text = text
    for abb in abbreviations:
        placeholder_text = placeholder_text.replace(abb, abb[:-1] + "<PERIOD>")

    sentences = re.split(r"(?<=[.!?])\s+", placeholder_text)
    # used this stack overflow on how to use regex to split sentences with punctuation (https://stackoverflow.com/questions/72872978/split-text-into-sentences-wit0hout-nltk)
    # gemini helped to put together the abbreviated words
    final_sentences = []
    for sentence in sentences:
        # Change the placeholder back to a period and strip any extra whitespace.
        restored_sentence = sentence.replace("<PERIOD>", ".").strip()
        # Ensure we don't add empty strings to our list.
        if restored_sentence:
            final_sentences.append(restored_sentence)

    return final_sentences


def split_into_words(sentences):
    # keep and split underscores, commas, periods, semi-colons
    # https://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    token_sentence = []
    for sentence in sentences:
        token = re.findall(r"[\w']+|[.,!?;]", sentence)
        token_sentence.append(token)
    
    return token_sentence


# DO NOT CHANGE ANYTHING BELOW THIS LINE.
#
if __name__ == "__main__":

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="author.py")
    parser.add_argument(
        "-data",
        action="store",
        dest="data",
        type=str,
        default="data",
        help="Directory containing the books",
    )
    parser.add_argument(
        "-test",
        action="store",
        type=bool,
        default=False,
        help="Use the test set not dev",
    )
    args = parser.parse_args()

    passages = data.Passages(args.data)

    # TRAIN
    train(passages.get_training())

    # TEST
    if args.test:
        testset = passages.get_test()
    else:
        testset = passages.get_development()
    predicted_labels = test(testset)

    # EVALUATE
    accuracy = data.evaluate(predicted_labels, testset)
    print("Final Accuracy: %.2f%%\n\n" % (accuracy))

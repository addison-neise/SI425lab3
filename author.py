#
# author.py -data <data-dir> [-test]
#
import re
import argparse
import data


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

    def train(self, passages):
        """
        Given a list of passages and their known authors, train your learning model.
        passages: a List of passage pairs (author,text)
        Returns: void
        """
        # used gemini to find how to access the for author, passage in passages.

        for author, passage in passages:
            passage_words = split_into_sentences(passage)

            passage_words.append(self.STOP)
            passage_words = [self.START] + passage_words

            for w in passage_words:
                if not w in self.wordcounts:
                    self.wordcounts[w] = 1
                else:
                    self.wordcounts[w] += 1

            for i in range(len(passage_words)):
                w1 = passage_words[i - 1]
                w2 = passage_words[i]
                if not (w1, w2) in self.bigramcounts:
                    self.bigramcounts[(w1, w2)] = 1
                else:
                    self.bigramcounts[(w1, w2)] += 1
            self.N += len(passage_words)

        self.V += len(passage_words)

    def get_word_probability(self, sentence, index):
        if index == 0:
            w1 = "<s>"
        else:
            w1 = sentence[index - 1]

        if index == len(sentence):
            w2 = self.STOP
        else:
            w2 = sentence[index]

        bigram = (w1, w2)

        unigram = self.wordcounts.get(w1, 0)
        bigram_count = self.bigram_counts.get(bigram, 0)

        k = 0.0101

        return (bigram_count + k) / (unigram + (self.V * k))


author_models = list()
author_passages = dict()


# calling this function for the testing from Chambers
def train(passages):
    """
    Orchestrates the training process by creating a separate model for each author.
    """
    print("Training models for each author...")
    # Step 1: Group all passages by their author
    author_texts = {}
    for author, passage in passages:
        if author not in author_texts:
            author_texts[author] = ""
        author_texts[author] += passage + " "

    # Step 2: Train a separate classifier for each author
    for author, full_text in author_texts.items():
        classifier = Author_Classifier()
        classifier.train(full_text)
        author_models[author] = classifier
        print(f"  - Model for '{author}' trained.")


def test(passages):
    """
    Given a list of passages, predict the author for each one.
    passages: a List of passage pairs (author,text)
    Returns: a list of author names, the author predictions for each given passage.
    """

    return []


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

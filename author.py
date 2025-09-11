#
# author.py -data <data-dir> [-test]
#
import re
import argparse
import data

 
#
# YOU MUST WRITE THESE TWO FUNCTIONS (train and test)
#

def train(passages):
    '''
    Given a list of passages and their known authors, train your learning model.
    passages: a List of passage pairs (author,text) 
    Returns: void
    '''
    pass


def test(passages):
    '''
    Given a list of passages, predict the author for each one.
    passages: a List of passage pairs (author,text)
    Returns: a list of author names, the author predictions for each given passage.
    '''
    return []


def split_into_sentences(text):

# used regex to to strip new lines, just googled this and it came up in the google AI
    text = text.replace("--**--**--", "")

    text = re.sub(r'\s+', ' ', text).strip()

    # asked gemini how to handle the abbrevations in the text so they could be protected and it gave back the list of stuff we would protect and then added a period back in after it if these abbreviations were found.
    abbreviations = [
        'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Rev.', 'Capt.', 'Sgt.', 'Gen.', 
        'Sen.', 'Rep.', 'St.', 'Jr.', 'Sr.', 'e.g.', 'i.e.', 'etc.'
    ]
    
    placeholder_text = text
    for abb in abbreviations:
        placeholder_text = placeholder_text.replace(abb, abb[:-1] + '<PERIOD>')

    sentences = re.split(r'(?<=[.!?])\s+', placeholder_text)
# used this stack overflow on how to use regex to split sentences with punctuation (https://stackoverflow.com/questions/72872978/split-text-into-sentences-wit0hout-nltk)
# gemini helped to put together the abbreviated words
    final_sentences = []
    for sentence in sentences:
        # Change the placeholder back to a period and strip any extra whitespace.
        restored_sentence = sentence.replace('<PERIOD>', '.').strip()
        # Ensure we don't add empty strings to our list.
        if restored_sentence:
            final_sentences.append(restored_sentence)
            
    return final_sentences

def split_into_words(sentences):
    # keep and split underscores, commas, periods, semi-colons
    # https://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    for sentence in sentences:
        for word in sentence:
            re.findall(r"[\w']+|[.,!?;]", sentence)
#
# DO NOT CHANGE ANYTHING BELOW THIS LINE.
#
if __name__ == '__main__':

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="author.py")
    parser.add_argument('-data', action="store", dest="data", type=str, default='data', help='Directory containing the books')
    parser.add_argument('-test', action="store", type=bool, default=False, help='Use the test set not dev')
    args = parser.parse_args()
    
    passages = data.Passages(args.data)

    # TRAIN
    train(passages.get_training())

    # TEST
    if args.test:  testset = passages.get_test()
    else:          testset = passages.get_development()
    predicted_labels = test(testset)
    
    # EVALUATE
    accuracy = data.evaluate(predicted_labels, testset)
    print('Final Accuracy: %.2f%%\n\n' % (accuracy))

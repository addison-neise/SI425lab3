#
# author.py -data <data-dir> [-test]
#

import argparse
import data


#
# YOU MUST WRITE THESE TWO FUNCTIONS (train and test)
#

class Author_Classifier():

    def __init__(self):
        self.START = '<S>'
        self.STOP = '</S>'
        self.UNK = 'UNK'
        self.wordcounts = dict()
        self.wordcounts[self.STOP] = 0
        self.bigramcounts = dict()
        self.N = 0
        self.V = 0

    def train(self, passages):
        '''
        Given a list of passages and their known authors, train your learning model.
        passages: a List of passage pairs (author,text) 
        Returns: void
        '''
        # used gemini to find how to access the for author, passage in passages.

        for author, passage in passages:
            passage = passage.copy()
            passage.append(self.STOP)
            passage = ['<s>'] + passage

            for w in passage:
                if not w in self.wordcounts:
                    self.wordcounts[w] = 1
                else:
                    self.wordcounts[w] += 1

            for i in range(len(passage)):
                w1 = passage[i - 1]
                w2 = passage[i]
                if not (w1, w2) in self.bigramcounts:
                    self.bigramcounts[(w1, w2)] = 1
                else:
                    self.bigramcounts[(w1, w2)] += 1
            self.N += len(passage)
        pass

#calling this function for the testing from Chambers
def train(passages):
    x = Author_Classifier()
    x.train(passages)


def test(passages):
    '''
    Given a list of passages, predict the author for each one.
    passages: a List of passage pairs (author,text)
    Returns: a list of author names, the author predictions for each given passage.
    '''
    return []




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

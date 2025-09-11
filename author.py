#
# author.py -data <data-dir> [-test]
#

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

This is an implementation of a recursive neural tensor net (RNTN), as described in:
========================================================

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)

Included:
--------
-rntn.py, the main program
-rntn\_dictionary.py, dictionary class for phrases
-phrase_tree.py, classes for sentences, phrases, and individual tokens

-binary_tree.py, not actually part of the program, but contains a postorder traversal method for binary trees
                 that could be used to sort individual tree nodes to allow bottom-up or top-down traversal
		 of a tree without recursion

toydata, a very small dataset (9 hand-made sentences), for debugging or demo purposes.
         The format of this is the same as that of the Stanford Sentiment Treebank, downloadable from Stanford.

checkpoints, for storing parameter checkpoint files

To run the program (note: Python 3 only!):
------------------------------------
For a long list of options:
python3 rntn.py --help
(Note: the docopt options processor is unforgiving, and unfortunately not very informative
if the options are not a perfect match for what it expects. My apologies, I didn't find out until I was committed to using it.

Training (first time):
python3 rntn.py train --data-from=toydata --learning-rate=0.001 --wvlr=0.001 --lambda=1.0 --report-interval=100 --validate-interval=100 --checkpoint-interval=100 --checkpoint-base=rntn_test --checkpoint-dir=checkpoints --log-name=rntn-test.log --word-vector-size=10 --cost-threshold=1.0 --check-training=True --batch-size=3

Training (continuing with saved parameters):
python3 rntn.py train --data-from=toydata --learning-rate=0.001 --wvlr=0.001 --lambda=1.0 --report-interval=100 --validate-interval=100 --checkpoint-interval=100 --checkpoint-base=rntn_test --checkpoint-dir=checkpoints --log-name=rntn-test.log --word-vector-size=10 --cost-threshold=1.0 --check-training=True --batch-size=3 --params-from=checkpoints/rntn_test_20160331_145700_700.pickle



Gradient checking:
python3 rntn_procedural.py check-grad --epsilon=1e-8
Vocabulary size from toydata is 35 items.
Cost = 8.047024223401255, epsilon=1e-08
Differences:
   Value                  Numeric               Analytical                    Delta                    Ratio
       V:  -0.00331965577788650990  -0.00333856046199798584   0.00001890468411147594   0.99433747608763134451
       W:  -0.00059401383623480797  -0.00060097797540947795   0.00000696413917466998   0.98841200863986145020
      Ws:  -0.00042632559780031443  -0.00042553490493446589  -0.00000079069286584854   1.00185811081198972161
  W_bias:  -0.00013749001664109528  -0.00015362887643277645   0.00001613885979168117   0.89494973135161626221
 Ws_bias:   0.00009308110020356253   0.00009307888103649020   0.00000000221916707233   1.00002384153318213400
       L:   0.00038571836194023490   0.00038571836194023490   0.00000000000000000000   1.00000000000000000000



To check prediction accuracy:
python3 rntn.py accuracy validation --params-from=checkpoints/rntn_test_20160331_145718_800.pickle




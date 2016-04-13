This is an implementation of a recursive neural tensor net (RNTN), as described in:
----------------------------------------------------------------------

* __Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank__, Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts, Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)

Included:
--------
- rntn.py, the main program
- rntn\_dictionary.py, dictionary class for phrases
- phrase_tree.py, classes for sentences, phrases, and individual tokens

- binary_tree.py, not actually part of the program, but contains a postorder traversal method for binary trees that could be used to sort individual tree nodes to allow bottom-up or top-down traversal of a tree without recursion

- toydata, a very small dataset (9 hand-made sentences), for debugging or demo purposes.
         The format of this is the same as that of the Stanford Sentiment Treebank, downloadable from Stanford.

- checkpoints, a directory for storing parameter checkpoint files

To run the program:
------------------------------------
###For a long list of options:

    python rntn.py --help
(Note: the docopt options processor is unforgiving, and unfortunately not very informative
if the options are not a perfect match for what it expects. My apologies, I didn't find out until I was committed to using it.

###Training (first time):

    python rntn.py train --data-from=toydata --learning-rate=0.001 --wvlr=0.001 --lambda=1.0 --report-interval=100 --validate-interval=100 --checkpoint-interval=100 --checkpoint-base=rntn_test --checkpoint-dir=checkpoints --log-name=rntn-test.log --word-vector-size=10 --cost-threshold=1.0 --check-training=True --batch-size=3

###Training (continuing with saved parameters):

    python rntn.py train --data-from=toydata --learning-rate=0.001 --wvlr=0.001 --lambda=1.0 --report-interval=100 --validate-interval=100 --checkpoint-interval=100 --checkpoint-base=rntn_test --checkpoint-dir=checkpoints --log-name=rntn-test.log --word-vector-size=10 --cost-threshold=1.0 --check-training=True --batch-size=3 --params-from=checkpoints/rntn_test_20160331_145700_700.pickle



###Gradient checking:

    python rntn.py check-grad --data-from=toydata --epsilon=1e-4
    Starting at Wed Apr 13 09:16:39 2016
    Vocabulary size from toydata is 35 items.
    Cost = 2.60940033377, epsilon=0.0001
    Differences:
       Value                  Numeric               Analytical                    Delta                    Ratio
           V:   0.00011167273623868823   0.00011167274614379296  -0.00000000000990510473   0.99999991130239851422
           W:  -0.00005486056397785433  -0.00005486045947551031  -0.00000000010450234401   1.00000190487547890861
          Ws:  -0.00001717166742309928  -0.00001717167076831377   0.00000000000334521449   0.99999980518992404033
      W_bias:  -0.00002682896047190297  -0.00002682896813902627   0.00000000000766712330   0.99999971422220712558
     Ws_bias:  -0.00000427663326263428  -0.00000429158334580571   0.00000001495008317143   0.99651641784237066091
           L:  -0.00007661323182941551  -0.00007661323182941551   0.00000000000000000000   1.00000000000000000000
    Largest difference was 0.000046003162424, at index 4211



###To check prediction accuracy:

    python rntn.py accuracy validation --params-from=checkpoints/rntn_test_20160331_145718_800.pickle




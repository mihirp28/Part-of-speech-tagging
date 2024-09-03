# Part-of-speech-tagging
## Problem
A basic problem in Natural Language Processing is part-of-speech tagging, in which the goal is to mark every word in a sentence with its part of speech (noun, verb, adjective, etc.). This is valuable for improving the performance of NLP systems for tasks such as machine translation.
Sometimes this is easy: a sentence like "Blueberries are blue" clearly consists of a noun, verb, and adjective, since each of these words has only one possible part of speech (e.g., "blueberries" is a noun and can’t be a verb).
But in general, one has to look at all the words in a sentence to figure out the part of speech of any individual word. For example, consider the —- grammatically correct! —- sentence: “Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.” To figure out what it means, we can parse its parts of speech:

                       Buffalo      buffalo      Buffalo      buffalo     buffalo     buffalo    Buffalo    buffalo.
                       Adjective      Noun      Adjective      Noun        Verb        Verb      Adjective    Noun

(In other words: the buffalo living in Buffalo, NY that are buffaloed (intimidated) by buffalo living in
Buffalo, NY buffalo (intimidate) buffalo living in Buffalo, NY.) That’s an extreme example, obviously. Here’s a more mundane sentence:

                  Her    position    covers    a    number    of    daily    tasks    common    to    any    social    director.
                  DET       NOUN       VERB    DET    NOUN    ADP    ADJ      NOUN      ADJ     ADP   DET      ADJ       NOUN

where DET stands for a determiner, ADP is an adposition, ADJ is an adjective, and ADV is an adverb.(1) Many of these words can be different parts of speech: “position” and “covers” can both be nouns or verbs, for example. The only way to resolve the ambiguity is to look at the surrounding words. Labeling parts of speech thus involves an understanding of the intended meaning of the words in the sentence, as well as the relationships between the words.

Fortunately, statistical models work amazingly well for NLP problems. Consider the Bayes net shown in Figure 2. This Bayes net has random variables S = {S1, . . . , SN } and W = {W1, . . . , WN }. The W ’s represent observed words in a sentence. The S ’s represent part of speech tags, so Si ∈ { VERB, NOUN, ...}. The arrows between W and S nodes model the relationship between a given observed word and the possible parts of speech it can take on, P (Wi | Si). (For example, these distributions can model the fact that the word “dog” is a fairly common noun but a very rare verb.) The arrows between S nodes model the probability that a word of one part of speech follows a word of another part of speech, P (Si+1 | Si). (For example, these arrows can model the fact that verbs are very likely to follow nouns, but are unlikely to follow adjectives.)

Data. To help you with this assignment, we’ve prepared a large corpus of labeled training and testing data.
Each line consists of a sentence, and each word is followed by one of 12 part-of-speech tags: ADJ (adjective), ADV (adverb), ADP (adposition), CONJ (conjunction), DET (determiner), NOUN, NUM (number), PRON (pronoun), PRT (particle), VERB, X (foreign word), and . (punctuation mark). (2)
![image](https://github.com/user-attachments/assets/e31c00e5-fe6a-4bcb-b851-2631e9af8580)

## Goal
Goal in this part is to implement part-of-speech tagging in Python, using Bayes networks.
1. To get started, consider the simplified Bayes net in Figure 1. To perform part-of-speech tagging, we’ll want to estimate the most-probable tag s<sup>∗</sup><sub>i</sub> for each word W<sub>i</sub>,
   
  s<sub>i</sub><sup>*</sup> = arg max<sub>s<sub>i</sub></sub> P(S<sub>i</sub> = s<sub>i</sub> | W)

  Implement part-of-speech tagging using this simple model.

2. Now consider Figure 2, a richer Bayes net that incorporates dependencies between words. Implement Viterbi to find the maximum a posteriori (MAP) labeling for the sentence,
   
  (s*<sub>i</sub>......s*<sub>n</sub>) = arg max<sub>s<sub>i</sub></sub>,.......<sub>s<sub>n</sub></sub> P(S<sub>i</sub> = s<sub>i</sub> | W)

Your program should take as input a training filename and a testing filename. The program should use the training corpus to estimate parameters, and then display the output of Steps 1-2 on each sentence in the testing file. For the result generated by both approaches (Simple and HMM), as well as for the ground truth result, your program should output the logarithm of the joint probability P(S, W) for each solution it finds under each of the models in Figure 1 and 2. It should also display a running evaluation showing the
percentage of words and whole sentences that have been labeled correctly so far. **For example:**

            python3 ./label.py training_file testing_file
            Learning model...
            Loading test data...
            Testing classifiers...
                                    Simple   HMM     Magnus   ab     integro      seclorum   nascitur   ordo .
              0. Ground truth       -48.52   -64.33   noun   verb     adv           conj       noun     noun .
              1. Simplified         -47.29   -66.74   noun   noun     noun           adv       verb     noun .
              2. HMM                 -47.48   -63.83   noun   verb     adj           conj      noun     verb .
            
            ==> So far scored 1 sentences with 17 words.
                                              
                                      Words correct:   Sentences correct:
              0. Ground truth             100.00%         100.00 %
              1. Simplified                 2.85 %         0.00 %
              2. HMM                        71.43 %         0.00 %

We’ve already implemented some skeleton code to get you started, in three files: label.py, which is the main program, pos_scorer.py, which has the scoring code, and pos_solver.py, which will contain the actual part-of-speech estimation code. You should only modify the latter of these files. The current version of pos_solver.py we’ve supplied is very simple, as you’ll see. In your report, please make sure to include your results (accuracies) for each technique on the test file we’ve supplied, bc.test. Your code should finish
within about 10 minutes.

(1)
If you didn’t know the term “adposition”, neither did we. The adpositions in English are prepositions; in many languages,
there are postpositions too. But you won’t need to understand the linguistic theory between these parts of speech to complete
the assignment; if you’re curious, check out the "Part of Speech" Wikipedia article for some background.
(2) This dataset is based on the Brown corpus. Modern part-of-speech taggers often use a much larger set of tags - often over
100 tags, depending on the language of interest - that carry finer-grained information like the tense and mood of verbs,
whether nouns are singular or plural, etc. In this assignment we’ve simplified the set of tags to the 12 described here; the
simple tag set is due to Petrov, Das and McDonald, and is discussed in detail in their 2012 LREC paper if you’re interested.

![image](https://github.com/user-attachments/assets/2b8a211d-0aff-4808-a90f-d8ef851cf373)

   Figure 3: Our goal is to extract text from a noisy scanned image of a document.


## Solution
1. **`posterior(self, model, sentence, label)`**:
   - Calculates posterior probability of a given model, sentence, and label.
   - For the "Simple" model, it calculates sum of logarithms of emission and initial probabilities.
   - For the "HMM" model, it calculates sum of logarithms of emission, initial, and transition probabilities.
   - Returns calculated posterior probability.

2. **`emi_probab(self, word, val)`**:
   - Calculates emission probability of a word given a part-of-speech label.
   - Checks if probability is explicitly defined and returns it if present.
   - If not defined, calculates the probability based on counts and caches result for future use.
   - Returns calculated emission probability.

3. **`intl_probab(self, val)`**:
   - Calculates initial probability of a part-of-speech label.
   - Returns probability of label occurring at beginning of sentence.

4. **`train(self, train_file)`**:
   - Trains HMM model using provided training file.
   - Updates counts for global initial counts, emission counts, transition counts, and second-order transition counts.
   - Initializes lists and dictionaries for storing unique part-of-speech labels.
   - No return value.

5. **`simplified(self, wrd)`**:
   - Performs simplified part-of-speech tagging for each word in input sentence.
   - Determines part-of-speech label with maximum probability for each word using emission and initial probabilities.
   - Returns list of predicted part-of-speech labels for each word.

6. **`get_tran_probab(self, val1, val2)`**:
   - Calculates transition probability between two part-of-speech labels.
   - Checks if probability is explicitly defined and returns it if present.
   - If not defined, calculates the probability based on counts.
   - Returns calculated transition probability.

7. **`hmm_viterbi(self, wrd)`**:
   - Implements Viterbi algorithm for Hidden Markov Models.
   - Initializes Viterbi table and backpointer table.
   - Iterates through the words, updating tables to find most likely sequence of part-of-speech labels.
   - Returns the sequence of predicted part-of-speech labels.

8. **`solve(self, model, sentence)`**:
   - Solves the part-of-speech tagging problem based on the specified model.
   - Calls either `simplified` or `hmm_viterbi` based on the provided model.
   - Returns a list of predicted part-of-speech labels for each word in the input sentence.
   
 **Simplified Model**
In the Simplified model, we examined how parts-of-speech are connected to words.

Formulation
P(S/W)=max P(W/S),P(S)/P(W)

We aimed to maximize the posterior probability P(S|W) for each word in the phrase, representing the probability of a part-of-speech given the word. This involved multiplying the prior probability (P(S)) by the likelihood of the word given that part-of-speech (P(W|S)). The overall accuracy achieved with this approach was 91.51% for individual words and 36.35% for entire phrases.


**Hidden Markov Model using Viterbi Algorithm**
Hidden Markov Model with Viterbi Algorithm takes into account the relationship between consecutive words and the connection between words and the entire sentence. In this approach, we keep a record of the highest probabilities as we determine the probabilities for each word in a phrase. We utilize the maximum probability from the previous word, which is precalculated. To simplify the calculations, we created a transition matrix.

The overall accuracy of this method was 95.05% for individual words and 54.45% for entire phrases.


![image](https://github.com/user-attachments/assets/5e78c4c8-2f4b-46de-a635-485f484ff909)


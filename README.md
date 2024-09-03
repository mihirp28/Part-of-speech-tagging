# Part-of-speech-tagging
## Problem


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


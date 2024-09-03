###################################
# CS B551 Fall 2023, Assignment #3
#
# Your names and user ids:

import math

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        self.unq_parts = []      
        self.tran_prob = {}       
        self.verTble = [{}]        
        self.tran_cnt = {}       
        self.tran2_cnt = {}     
        self.emi_prob = {}
        self.emi_cnt = {}
        self.gbl_intl_cnt = {}
        

     #function to calculate posterior probability of a given model,sentence and label 
    def posterior(self, model, sentence, label):
        wrd = list(sentence)
        val = list(label)

        if model == "Simple":
            #Calculating posterior probability for Simple model
            p = sum(
                math.log10(self.emi_probab(wrd[i], val[i])) +
                math.log10(self.gbl_intl_cnt[val[i]] / sum(self.gbl_intl_cnt.values()))
                for i in range(len(wrd))
            )
            return p

        elif model == "HMM":
            #calculating posterior probability for HMM model
            intl_prob = math.log(self.gbl_intl_cnt[val[0]] / sum(self.gbl_intl_cnt.values()), 10)
            emi_sum = sum(math.log(self.emi_probab(wrd[i], val[i]), 10) for i in range(len(val)))
            tran_sum = sum(
                math.log(self.get_tran_probab(val[i - 1], val[i]), 10)
                for i in range(1, len(val))
            )
            return intl_prob + emi_sum + tran_sum

        else:
            print("Unknown")

    #function to calculate emission probability
    def emi_probab(self, word, val):
        #checking if emission probability is explicitly defined
        if word in self.emi_prob and val in self.emi_prob[word]:
            return self.emi_prob[word][val]

        #calculating emission probability based on counts
        if word in self.emi_cnt and val in self.emi_cnt[word]:
            emi_cnt = self.emi_cnt[word][val]
            total_tran_cnt = 0

            #calculating total count
            for pos in self.tran_cnt[val]:
                total_tran_cnt= total_tran_cnt+self.tran_cnt[val][pos]

            calculated_probability = emi_cnt / total_tran_cnt

            #cache calculated probability for future use
            if word not in self.emi_prob:
                self.emi_prob[word] = {}
            self.emi_prob[word][val] = calculated_probability
            return calculated_probability

        #default value if no probability is defined
        return 0.00000001
    
    #function to calculate initial probability
    def intl_probab(self, val):
        return self.gbl_intl_cnt.get(val, 0) / sum(self.gbl_intl_cnt.values()) or 0.00000001

    #do the training!
    def train(self, train_file):
        prev_lbl = None
        prev_prev_lbl = None

        i=0
        while i<len(train_file):
            wrd,pos=train_file[i]

            #updating global_initial_count
            self.gbl_intl_cnt[pos[0]]=self.gbl_intl_cnt.get(pos[0], 0) + 1

            j=0
            while j<len(wrd):
                word,crrnt_pos=wrd[j],pos[j]

                #updating emission_count_global
                self.emi_cnt.setdefault(word, {}).setdefault(crrnt_pos, 0)
                self.emi_cnt[word][crrnt_pos] += 1

                #updating tran_cnt
                if prev_lbl is not None:
                    self.tran_cnt.setdefault(prev_lbl, {}).setdefault(crrnt_pos, 0)
                    self.tran_cnt[prev_lbl][crrnt_pos] += 1

                #updating transition2_global
                if (prev_prev_lbl and prev_lbl) is not None:
                    self.tran2_cnt \
                        .setdefault(prev_prev_lbl, {}) \
                        .setdefault(prev_lbl, {}) \
                        .setdefault(crrnt_pos, 0)
                    self.tran2_cnt[prev_prev_lbl][prev_lbl][crrnt_pos] += 1

                prev_lbl=crrnt_pos

                j=j+1

            prev_prev_lbl=None
            i=i+1

        self.unq_parts=list(self.tran_cnt.keys())

    #function for simplified part-of-speech tagging
    def simplified(self, wrd):
        #initializing an empty list to store predicted labels for each word
        lbl_lst=['']*len(wrd)
        
        #iterating through each word in input sentence
        j=0
        while j<len(wrd):
            #retrieving current word from sentence
            word=wrd[j]


            #find part of speech label with maximum probability for current word
            #the label is determined by maximizing product of emission and initial probabilities
            #using lambda function and max function with key argument
            max_lbl = max(self.unq_parts, key=lambda tag: self.emi_probab(word, tag) * self.intl_probab(tag))
            
            #assigning predicted label to corresponding position in lbl_lst list
            lbl_lst[j]=max_lbl
            j=j+1

        return lbl_lst

    #function to calculate transition probability between two part-of-speech labels
    def get_tran_probab(self, val1, val2):
        if val1 in self.tran_prob and val2 in self.tran_prob[val1]:
            return self.tran_prob[val1][val2]

        #check if both part of speech labels have counts in training data
        if val1 in self.tran_cnt and val2 in self.tran_cnt[val1] and val2 in self.tran_cnt:
            total_tran_cnt = 0

            #calculating total count
            for pos in self.tran_cnt[val1]:
                total_tran_cnt += self.tran_cnt[val1][pos]

            #calculating transition probability from val1 to val2 based on counts
            val = self.tran_cnt[val1][val2] / total_tran_cnt
            return val

        return 0.0000001


    def hmm_viterbi(self, wrd):
        #initializing viterbi table and backpointer table
        self.verTble=[{}]
        vtrbi_trk={}

        #initialization for first word
        i=0
        while i<len(self.unq_parts):
            current_pos=self.unq_parts[i]
            #calculating initial probability for each part-of-speech label
            probab=self.intl_probab(current_pos)*self.emi_probab(wrd[0], current_pos)
            self.verTble[0][current_pos]=probab
            vtrbi_trk[current_pos]=[current_pos]
            i=i+1


        #iteration for remaining word
        i = 1
        while i<len(wrd):
            #adding new row for current word to Viterbi table
            self.verTble.append({})
            current_path={}

            #iterating over current possible pos positions
            for crrnt_pos in self.unq_parts:
                #find maximum value and corresponding state in previous row
                max_value, state = max(
                    (self.verTble[i - 1][pre_pos]*self.get_tran_probab(pre_pos, crrnt_pos)*
                    self.emi_probab(wrd[i], crrnt_pos), pre_pos)
                    for pre_pos in self.unq_parts
                )
                #updating Viterbi table with maximum value
                self.verTble[i][crrnt_pos]=max_value
                #updating the backpointer table with path leading to maximum value
                current_path[crrnt_pos]=vtrbi_trk[state] + [crrnt_pos]
                    
            #updatE backpointer table for current word
            vtrbi_trk=current_path
            i=i+1

        # Find the best state for the last word
        best_state=max(self.unq_parts, key=lambda pos: self.verTble[-1][pos])

        return vtrbi_trk[best_state]

  
  # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    def solve(self, model, sentence):
        wrd=list(sentence)
        if model=="Simple":
            return self.simplified(wrd)
        elif model=="HMM":
            return self.hmm_viterbi(wrd)
        else:
            print("Unknown algo!")

# Normalisation for BertTweet
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re
import pandas as pd

tweetTokenizer = TweetTokenizer()

# Same function that was used to preprocess the tweets which were used for
# pretraining the RoBERTa model we use: https://huggingface.co/vinai/bertweet-base
def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

# Same function that was used to preprocess the tweets which were used for
# pretraining the RoBERTa model we use: https://huggingface.co/vinai/bertweet-base
def normalizeTweet(tweet):
    tokens = tweetTokenizer.tokenize(tweet.replace("’", "'").replace("…", "...").replace("--", " ").replace("-"," "))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = normTweet.replace("cannot ", "can not ").replace("n't ", " n't ").replace("n 't ", " n't ").replace("ca n't", "can't").replace("ai n't", "ain't")
    normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll ", " 'll ").replace("'d ", " 'd ").replace("'ve ", " 've ")
    normTweet = normTweet.replace(" p . m .", "  p.m.") .replace(" p . m ", " p.m ").replace(" a . m .", " a.m.").replace(" a . m ", " a.m ")

    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)

    return " ".join(normTweet.split())


def bio_tagging(sentence, causes, effects):
    """
    Each token gets associated to one of the following labels:
    B-C : Begin cause
    I-C : Inside cause
    B-E : Begin effect
    I-C : Inside effect
    O   : Outside
    """

    tokens = normalizeTweet(sentence).split(" ")
    #print(tokens)
    bio = ["O"] * len(tokens)


    if not causes and not effects: # if neither cause nor effect (should not occur)
        raise ValueError(F"Neither causes nor effects exist in: \n{sentence}")
    if causes == ["nan"] and effects == ["nan"]: # no causes and no effects (often in sentences of a tweet without causality)
        return bio
    if [cause for cause in causes if cause in sentence] and not [effect for effect in effects if effect in sentence]: # if only causes and no effects return only "0"
        return bio
    if not [cause for cause in causes if cause in sentence] and [effect for effect in effects if effect in sentence]: # if only effects and no causes return only "0"
        return bio
    if causes == ["nan"] or effects == ["nan"]:
        raise ValueError(F"WEIRD, causes or effect is nan. Either both or non shoud be nan. Sentence: \n{sentence}")


    ########### SPECIAL CASES (often: cause or effect word occur several times in sentence) ############
    if sentence.startswith("my dads diabetic but i'm really bad at working out if he's having a diabetic hypo"):
        effect_index = tokens.index("hypo")
        bio[effect_index] = "B-E"
        bio[effect_index-1] = "B-C"
        return bio

    if sentence.startswith("USER I'm diabetic I'm diabetic H Y P O G L U C E M Y"):
        cause_indices = [i for i, x in enumerate(tokens) if x == "diabetic"]
        for cause_index in cause_indices:
            bio[cause_index] = "B-C"
        effect_start_index = tokens.index("H")
        bio[effect_start_index] = "B-E"
        for i in range(1,11):
            bio[effect_start_index + i] = "I-E"
        return bio

    if sentence.startswith("‘ I lost my leg to diabetes ' HTTPURL | Pakistan's soaring"):
        cause_indices = [i for i, x in enumerate(tokens) if x == "diabetes"]
        for cause_index in cause_indices:
            bio[cause_index] = "B-C"
        effect_start_index = tokens.index("lost")
        bio[effect_start_index] = "B-E"
        bio[effect_start_index+1] = "I-E"
        bio[effect_start_index+2] = "I-E"
        effect_start_index = tokens.index("amputations")
        return bio

    if sentence.startswith("USER USER I had three kids and a major weight loss due to diabetes"):
        cause_index = tokens.index("diabetes") # index searches first occurrence of token
        bio[cause_index] = "B-C"
        effect_index = tokens.index("weight")
        bio[effect_index] = "B-E"
        bio[effect_index+1] = "I-E"
        return bio

    if sentence.startswith("It just seems odd that he would go into a diabetic coma"):
        cause_index = tokens.index("diabetic") # index returns first occurrences
        effect_index = tokens.index("coma")
        bio[cause_index] = "B-C"
        bio[effect_index] = "B-E"
        return bio

    if sentence.startswith("Fat diabetic Aboriginals die Fat diabetic "):
        cause_indices = [i for i, x in enumerate(tokens) if x == "diabetic"]
        for cause_index in cause_indices:
            bio[cause_index] = "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "die"]
        for effect_index in effect_indices:
            bio[effect_index] = "B-E"
        return bio

    if sentence.startswith("USER that people think only fat people have medical"):
        cause_indices = [i for i, x in enumerate(tokens) if x == "fat"]
        bio[cause_indices[-1]] = "B-C"
        cause_index = tokens.index("eat")
        bio[cause_index] = "B-C"
        bio[cause_index+1] = "I-C"
        effect_index = tokens.index("diabetic")
        bio[effect_index] = "B-E"
        return bio

    if sentence.startswith("Using an insulin with a curve you're not familiar with is dangerous business but I guess potentially dying from a hypo is better than dying from DKA"):
        cause_index = tokens.index("hypo")
        bio[cause_index] = "B-C"
        cause_index = tokens.index("DKA")
        bio[cause_index] = "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "dying"]
        for effect_index in effect_indices:
            bio[effect_index] = "B-E"
        return bio

    if sentence.startswith("USER As a Type 1 Diabetic I need my insulin chilled in a fridge at all cost and since my electricity out in my neighborhood my insulin will become bad from the heat I need assistance for my power IMMEDIATELY"):
        cause_index = tokens.index("Type")
        bio[cause_index] = "B-C"
        bio[cause_index+1] = "I-C"
        bio[cause_index+2] = "I-C"
        effect_index = tokens.index("insulin") # only first occurrence
        bio[effect_index] = "B-E"
        return bio

    if sentence.startswith("PRAYER REQUEST 😳 Would you please remember my husband in prayer , he is in so much pain with his diabetic neuropathy and shoulder pain"):
        cause_index = tokens.index("diabetic")
        bio[cause_index] = "B-C"
        bio[cause_index+1] = "I-C"
        effect_index = tokens.index("pain") # only first occurrence
        bio[effect_index] = "B-E"
        return bio

    if sentence.startswith('" So , although he died of COVID - 19 , he also died because he had #diabetes But even more proximately , he died because he lost his job and access to healthcare'):
        cause_index = tokens.index("#diabetes") # only first occurrence
        bio[cause_index] = "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "died"]
        bio[effect_indices[1]] = "B-E"
        return bio

    if sentence.startswith("Of course I started to panic cause no one in my family have diabetes NO ONE and almost every person that has diabetes has it cause of their genes"):
        cause_index = tokens.index("genes")
        bio[cause_index] = "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "diabetes"]
        bio[effect_indices[1]] = "B-E"
        return bio

    if sentence.startswith("USER just a pity lawyers are stupid , I would have thrown in mix , diabetes and obesity , why haven't government banned sodas , sweets etc , that is leading cause for obesity and diabetes , these will also cause icu's to be flooded"):
        cause_index = tokens.index("sodas")
        bio[cause_index] = "B-C"
        cause_index = tokens.index("sweets")
        bio[cause_index] = "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "diabetes"]
        bio[effect_indices[1]] = "B-E"
        effect_indices = [i for i, x in enumerate(tokens) if x == "obesity"]
        bio[effect_indices[1]] = "B-E"
        return bio

    if sentence.startswith("Many died from heart attacks , cancer , diabetes , car accidents and because they showed covid in their test , that is what they had to say they died from"):
        cause_index = tokens.index("diabetes")
        bio[cause_index] = "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "died"]
        bio[effect_indices[0]] = "B-E"
        return bio

    if sentence.startswith("USER You , watching your neighbors die because they cannot afford insulin :"):
        cause_index = tokens.index("can")
        bio[cause_index] = "B-C"
        bio[cause_index+1] = "I-C"
        bio[cause_index+2] = "I-C"
        bio[cause_index+3] = "I-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "die"]
        bio[effect_indices[0]] = "B-E"
        return bio

    if sentence.startswith("Dad died of diabetes , aunt died of cervical cancer"):
        cause_index = tokens.index("diabetes")
        bio[cause_index] = "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "died"]
        bio[effect_indices[0]] = "B-E"
        return bio

    if sentence.startswith("I am diabetic on insulin and pills need the money for insulin"):
        cause_index = tokens.index("diabetic")
        bio[cause_index] = "B-C"
        effect_index = tokens.index("insulin") # first occurrence
        bio[effect_index] = "B-E"
        effect_index = tokens.index("pills") # first occurrence
        bio[effect_index] = "B-E"
        effect_index = tokens.index("insulin") # first occurrence
        bio[effect_index] = "B-E"
        effect_index = tokens.index("need") # first occurrence
        bio[effect_index] = "B-E"     # need
        bio[effect_index+1] = "I-E"   # the
        bio[effect_index+2] = "I-E"   # money
        bio[effect_index+3] = "I-E"   # for
        bio[effect_index+4] = "I-E"   # insulin
        return bio

    if sentence.startswith("I am not scared of the symptoms I have right now I am scared of a heart attack , I am scared of diabetes , stroke , lupus , permanent lung / heart damage and yea death due to my symptoms I have right now"):
        cause_index = tokens.index("diabetes")
        bio[cause_index] = "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "scared"]
        bio[effect_indices[-1]] = "B-E"
        return bio

    if sentence.startswith("USER lol cannot wait to die bc I cannot afford me insulin"):
        cause_index = tokens.index("afford")
        bio[cause_index-2] = "B-C"   # can
        bio[cause_index-1] = "I-C"   # not
        bio[cause_index] = "I-C"     # afford
        bio[cause_index+1] = "I-C"   # me
        bio[cause_index+2] = "I-C"   # insulin
        effect_index = tokens.index("die")
        bio[effect_index] = "B-E"
        return bio


    ################## Add BIO tags for causes and effects ########################

    if causes:
        for cause in causes: # cause can consist of several words
            if cause in sentence: # possible that cause is in another sentence of the tweet
                cause_words = normalizeTweet(cause).split(" ")
                cause_words_start = cause_words[0]
                try:
                    indices = [i for i, x in enumerate(tokens) if x == cause_words_start] # get all indices of the first word of the cause
                    if len(indices) > 1: # if several occurrences of the same cause start word in phrase
                        for cause_word_start_index in indices:
                            if len(cause_words) > 1:
                                if tokens[cause_word_start_index + 1] == cause_words[1]: # find right causal start word
                                    ind = cause_word_start_index
                                    break
                            else:
                                print("WARNING: cause", cause, "has several occurrences of this word are in sentence:")
                                print(sentence)
                    else:
                        ind = tokens.index(cause_words_start) # get index of causal word in tokens list
                    bio[ind] = "B-C"

                    i = 1
                    while i < len(cause_words):
                        if tokens[ind+i] == cause_words[i]:
                            bio[ind+i] = "I-C"
                        else:
                            print("Error: token and causal word don't match!")
                            print("ind:", ind, "i:", i, "token[ind+i]", tokens[ind+i], "cause_words[i]", cause_words[i])
                            print(sentence)
                        i += 1
                except ValueError:
                    print("\nINFO: cause word '{}' does not exist in sentence: \n'{}', but should be in other sentence of the tweet".format(cause_words_start, tokens))


    if effects:
        for effect in effects: # effects can consist of several words
            if effect in sentence:
                effect_words = normalizeTweet(effect).split(" ")
                effect_words_start = effect_words[0]
                try:
                    indices = [i for i, x in enumerate(tokens) if x == effect_words_start]
                    if len(indices) > 1: # if several occurrences of the same cause start word in phrase
                        for effect_word_start_index in indices:
                            if len(effect_words) > 1:
                                if tokens[effect_word_start_index + 1] == effect_words[1]:
                                    ind = effect_word_start_index
                                    break
                            else:
                                print("WARNING: effect", effect, "has several occurrences of this word are in sentence:")
                    else:
                        ind = tokens.index(effect_words_start) # get index of c_word in tokens list
                    bio[ind] = "B-E"

                    i = 1
                    while i < len(effect_words):
                        if tokens[ind+i] == effect_words[i]:
                            bio[ind+i] = "I-E"
                        else:
                            print("Error: token and effect word don't match!")
                            print("ind:", ind, "i:", i, "token[ind+i]", tokens[ind+i], "effect_words[i]", cause_words[i])
                        i += 1

                except ValueError:
                    print("\nError: effect word '{}' does not exist in sentence: \n'{}', but should be in other sentence of the tweet".format(effect_words_start, tokens))

    return bio


def split_into_sentences(text, min_words_in_sentences=3):
    """ Split tweet into sentences """

    text = " " + text + "  "
    text = text.replace("\n"," ")
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    sentences = [s  for s in sentences if s != ""]
    sentences = [s for s in sentences if len(s.split(" ")) >= min_words_in_sentences] #keep only sentences with minimal number of words
    return sentences


def create_training_data(data, min_words_in_sentences=3):
    """
        Transform tweets into sentences and associate to each sentence a causal
        label (0,1) and its BIO tags.
    """
    tweets = []
    causal_labels = []
    bio_labels = []

    for i, row in data.iterrows():
        sentences = split_into_sentences(row["full_text"], min_words_in_sentences=min_words_in_sentences)
        intents = set(str(row["Intent"]).strip().split(";"))
        causes = str(row["Cause"]).strip().split(";")
        effects = str(row["Effect"]).strip().split(";")
        #print("\n", row["full_text"])
        #print("\tintents:", intents)
        #print("\tcauses: '{}'".format(causes))
        #print("\teffects: '{}'".format(effects))

        if causes or effects: # if there are causes or effects
            tokens = normalizeTweet(row["full_text"]).split(" ")
            #print(tokens)

        # single sentences
        if set({"nan"}) == intents or set({" "}) == intents:
            tweets.append(row["full_text"])
            causal_labels.append(row["Causal association"])
            bio_labels.append(bio_tagging(row["full_text"], causes, effects))
            #print("A: single sentence => causality: {}".format(row["Causal association"]))
            #print(bio_labels[-1])
            if len(bio_labels[-1]) != len(normalizeTweet(row["full_text"]).split(" ")):
                print("ERROR 1: N tokens should be equal to N BIO tags")

        # to be ignored
        elif (
             set({"q"}) == intents
          or set({"joke"}) == intents
          or set({"q", "joke"}) == intents
          or set({"joke", "mS"}) == intents
          or set({"neg"}) == intents
          or set({"neg", "msS"}) == intents
          or set({"neg", "mS"}) == intents
          or set({"neg", "msS", "mE"}) == intents
          or set({"q", "joke", "mS"}) == intents
          or set({"q", "msS", "neg"}) == intents
          or set({"neg", "mC"}) == intents
          or set({"mC", "joke", "msS"}) == intents
          or set({"joke", "mE"}) == intents
        ):
            #print("B ignore")
            continue

        # multiple sentences
        elif (
             set({"mS"}) == intents # multiple sentences (possible that cause and effect in different sentences -> ignore)
          or set({"q", "mS"}) == intents # multiple sentences or question
          or set({"mS", "mE"}) == intents
          or set({"mC", "mS"}) == intents
          or set({"mC", "mS", "mE"}) == intents
          or set({"q", "mC", "mS"}) == intents
          or set({"q", "mC", "mS", "mE"}) == intents

        ):
            for sent in sentences:
#                print(sent)
                if sent[-1] != "?": # ignore questions
                    tweets.append(sent)
                    causal_labels.append(0)
                    bio_labels.append(bio_tagging(sent, causes, effects))
                    #print("\tC: C, E in multiple sentences, causality => 0")
                    #print(bio_labels[-1])
                    if len(bio_labels[-1]) != len(normalizeTweet(sent).split(" ")):
                        print("ERROR 2: N tokens should be equal to N BIO tags")

        # multiple sentences with cause and effect pairs in a single sentence
        elif (
            set({"msS"}) == intents # multiple sentences with cause and effect in single sentence
         or set({"q", "msS"}) == intents # msS and a question
         or set({"msS", "mE"}) == intents # msS with several effects
         or set({"mC", "msS"}) == intents
         or set({"mE"}) == intents # multiple effects
         or set({"mC"}) == intents # multiple causes
         or set({"mC", "msS", "mE"}) == intents
         or set({"mC", "mE"}) == intents
         or set({"q", "mC", "mE"}) == intents
         or set({"q", "mC", "msS"}) == intents
        ):

            if row["Causal association"] != 1: #TEST
                print(sentences)
                print("1) ERROR: Causal association should be 1 !!!!")
                print()

            for sent in sentences:
                #print("D => C,E in single sentence")
                #print("\t:", sent)
                if sent[-1] != "?": # ignore question
                    existCause = False
                    for cause in causes:
                        if cause in sent:
                            existCause = True

                    existEffect = False
                    for effect in effects:
                        if effect in sent:
                            existEffect = True

                    if existCause and existEffect:
                        tweets.append(sent)
                        causal_labels.append(row["Causal association"])
                        bio_labels.append(bio_tagging(sent, causes, effects))
                        #print("\tCausal:", causal_labels[-1])
                        #print("\t:", bio_labels[-1])
                        if len(bio_labels[-1]) != len(normalizeTweet(sent).split(" ")):
                            print("ERROR 3: N tokens should be equal to N BIO tags")
#                        print("E: add with Cause + effect => association: {}".format(row["Causal association"]))
                    else:
                        tweets.append(sent)
                        causal_labels.append(0)
                        bio_labels.append(bio_tagging(sent, causes, effects))
                        #print("\tCausal:", causal_labels[-1])
                        #print("\t:", bio_labels[-1])
                        if len(bio_labels[-1]) != len(normalizeTweet(sent).split(" ")):
                            print("ERROR 4: N tokens should be equal to N BIO tags")

                #else:
                #    print("H: question in sentence => ignore")
            if row["Causal association"] == 0:
                print(sentences)
                print("H: should not have causality == 0")

    return pd.DataFrame({"tweet" : tweets, "Causal association" : causal_labels, "BIOtags": bio_labels})

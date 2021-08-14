# Normalisation for BertTweet
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re
import pandas as pd
import numpy as np
import torch

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
    normTweet = normTweet.split()

    ### tokens like "word1.word2" are separated to "word1", ".", "word2"
    normTweetList = []
    for token in normTweet:
        if "." in token and len(token) > 1 and token != ".." and token != "...": # to avoid that "." is taken. Only focus on "word1.word2"
            try:
                if token == "a.m." or token == "p.m.":
                    normTweetList.append(token)
                else:
                    token1, token2 = token.split(".")
                    normTweetList.extend([token1, ".", token2])
            except:
                print("Error: token: {}".format(token))
                print(normTweet)
        else:
            normTweetList.append(token)
    ###
    normTweet = normTweetList

    return " ".join(normTweet)


def manual_tagging_of_some_special_tweets(tweet, tokens, bio, ioTagging=True):
    """
        Some tweets which require manual tagging.
        The causes or effects often occur several times in the tweet.
        Manual tagging allows to tag the right cause and effect

        ioTagging = True : Only tokens "I-C", "I-E"
        ioTagging = False : Tokens: "B-C", "I-C", "B-E", "I-E"
    """
        ########### SPECIAL CASES (often: cause or effect word occur several times in sentence) ############
    if "my dads diabetic but i'm really bad at working out if he's having a diabetic hypo" in tweet:
        effect_index = tokens.index("hypo")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        bio[effect_index-1] = "I-C" if ioTagging else "B-C"
        return bio

    if "USER I'm diabetic I'm diabetic H Y P O G L U C E M Y" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "diabetic"]
        for cause_index in cause_indices:
            bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_start_index = tokens.index("H")
        bio[effect_start_index] = "I-E" if ioTagging else "B-E"
        for i in range(1,11):
            bio[effect_start_index + i] = "I-E"
        return bio

    if "‘ I lost my leg to diabetes ' HTTPURL | Pakistan's soaring" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "diabetes"]
        for cause_index in cause_indices:
            bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_start_index = tokens.index("lost")
        bio[effect_start_index] = "I-E" if ioTagging else "B-E"
        bio[effect_start_index+1] = "I-E"
        bio[effect_start_index+2] = "I-E"
        effect_start_index = tokens.index("amputations")
        bio[effect_start_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER USER I had three kids and a major weight loss due to diabetes" in tweet:
        cause_index = tokens.index("diabetes") # index searches first occurrence of token
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("weight")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        bio[effect_index+1] = "I-E"
        return bio

    if "It just seems odd that he would go into a diabetic coma" in tweet:
        cause_index = tokens.index("diabetic") # index returns first occurrences
        effect_index = tokens.index("coma")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "Fat diabetic Aboriginals die Fat diabetic " in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "diabetic"]
        for cause_index in cause_indices:
            bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "die"]
        for effect_index in effect_indices:
            bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER that people think only fat people have medical" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "fat"]
        bio[cause_indices[-1]] = "I-C" if ioTagging else "B-C"
        cause_index = tokens.index("eat")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        bio[cause_index+1] = "I-C"
        effect_index = tokens.index("diabetic")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "Using an insulin with a curve you're not familiar with is dangerous business but I guess potentially dying from a hypo is better than dying from DKA" in tweet:
        cause_index = tokens.index("hypo")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        cause_index = tokens.index("DKA")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "dying"]
        for effect_index in effect_indices:
            bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER As a Type 1 Diabetic I need my insulin chilled in a fridge at all cost and since my electricity out in my neighborhood my insulin will become bad from the heat I need assistance for my power IMMEDIATELY" in tweet:
        cause_index = tokens.index("Type")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        bio[cause_index+1] = "I-C"
        bio[cause_index+2] = "I-C"
        effect_index = tokens.index("insulin") # only first occurrence
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "PRAYER REQUEST 😳 Would you please remember my husband in prayer , he is in so much pain with his diabetic neuropathy and shoulder pain" in tweet:
        cause_index = tokens.index("diabetic")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        bio[cause_index+1] = "I-C"
        effect_index = tokens.index("pain") # only first occurrence
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if '" So , although he died of COVID - 19 , he also died because he had #diabetes But even more proximately , he died because he lost his job and access to healthcare' in tweet:
        cause_index = tokens.index("#diabetes") # only first occurrence
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "died"]
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"
        return bio

    if "Of course I started to panic cause no one in my family have diabetes NO ONE and almost every person that has diabetes has it cause of their genes" in tweet:
        cause_index = tokens.index("genes")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "diabetes"]
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER just a pity lawyers are stupid , I would have thrown in mix , diabetes and obesity , why haven't government banned sodas , sweets etc , that is leading cause for obesity and diabetes , these will also cause icu's to be flooded" in tweet:
        cause_index = tokens.index("sodas")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        cause_index = tokens.index("sweets")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "diabetes"]
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"
        effect_indices = [i for i, x in enumerate(tokens) if x == "obesity"]
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"
        return bio

    if "Many died from heart attacks , cancer , diabetes , car accidents and because they showed covid in their test , that is what they had to say they died from" in tweet:
        cause_index = tokens.index("diabetes")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "died"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER You , watching your neighbors die because they cannot afford insulin :" in tweet:
        cause_index = tokens.index("can")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        bio[cause_index+1] = "I-C"
        bio[cause_index+2] = "I-C"
        bio[cause_index+3] = "I-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "die"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"
        return bio

    if "Dad died of diabetes , aunt died of cervical cancer" in tweet:
        cause_index = tokens.index("diabetes")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "died"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"
        return bio

    if "I am diabetic on insulin and pills need the money for insulin" in tweet:
        cause_index = tokens.index("diabetic")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("insulin") # first occurrence
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        effect_index = tokens.index("pills") # first occurrence
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        effect_index = tokens.index("insulin") # first occurrence
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        effect_index = tokens.index("need") # first occurrence
        bio[effect_index] = "I-E" if ioTagging else "B-E"     # need
        bio[effect_index+1] = "I-E"   # the
        bio[effect_index+2] = "I-E"   # money
        bio[effect_index+3] = "I-E"   # for
        bio[effect_index+4] = "I-E"   # insulin
        return bio

    if "I am not scared of the symptoms I have right now I am scared of a heart attack , I am scared of diabetes , stroke , lupus , permanent lung / heart damage and yea death due to my symptoms I have right now" in tweet:
        cause_index = tokens.index("diabetes")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "scared"]
        bio[effect_indices[-1]] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER lol cannot wait to die bc I cannot afford me insulin" in tweet:
        cause_index = tokens.index("afford")
        bio[cause_index-2] = "I-C" if ioTagging else "B-C"   # can
        bio[cause_index-1] = "I-C"   # not
        bio[cause_index] = "I-C"     # afford
        bio[cause_index+1] = "I-C"   # me
        bio[cause_index+2] = "I-C"   # insulin
        effect_index = tokens.index("die")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER The other big lie is being fat gives you diabetes , when the opposite is true . Having out of control blood sugar is what makes you fat . The" in tweet:
        cause_index = tokens.index("out")
        bio[cause_index] = "I-C" if ioTagging else "B-C"   # out
        bio[cause_index+1] = "I-C" # of
        bio[cause_index+2] = "I-C" # control
        bio[cause_index+3] = "I-C" # blood
        bio[cause_index+4] = "I-C" # sugar
        effect_indices = [i for i, x in enumerate(tokens) if x == "fat"]
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER For me , it's when I'm really sick of diabetes but can't make myself care , and if I stop & think I feel some anxiety but still don't do anything . It's things like feeling hypo" in tweet:
        cause_index = tokens.index("diabetes")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        cause_index = tokens.index("hypo")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "sick"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"
        effect_index = tokens.index("biscuit")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER USER USER USER Yeah , the same goes for my diabetes . I don't have an exact reason for mine . I was severely underweight when I was diagnosed . They think type 1 diabetes could have genetic links but don't know for sure" in tweet:
        cause_index = tokens.index("diabetes") # first occurence diabetes
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("underweight")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "Cut down on carbs ! Don't get insulin resistance and a bunch of other health issues . We didn't evolve to eat so much carbs ." in tweet:
        cause_index = tokens.index("carbs") # first occurence carbs
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("insulin")
        bio[effect_index] = "I-E" if ioTagging else "B-E" # insulin
        bio[effect_index+1] = "I-E" # resistance
        return bio

    if "USER Slightly more worried than I was to be honest . My husband is a mid - 40s well controlled T1 diabetic . Also" in tweet:
        cause_index = tokens.index("T1")
        bio[cause_index] = "I-C" if ioTagging else "B-C"   # T1
        bio[cause_index+1] = "I-C" # diabetic
        effect_index = tokens.index("worried")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "My grandpa died of heart failure , grandma was diagnosed with diabetes in her 70s . All of dads siblings have diabetes and HTN ( so do all my siblings , no amount of lifestyle changes affect it , we are all on meds ) . My mom" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "diabetes"]
        bio[cause_indices[1]] = "I-C" if ioTagging else "B-C"   # 2nd occurrence of diabetes
        effect_index = tokens.index("lifestyle")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # lifestyle
        bio[effect_index+1] = "I-E" # changes
        effect_index = tokens.index("on")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # on
        bio[effect_index+1] = "I-E" # meds
        return bio

    if "USER USER And keto is terrible as well but that's a different story . True ketosis will help with diabetes" in tweet:
        cause_index = tokens.index("keto")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        cause_index = tokens.index("ketosis")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("terrible")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        effect_index = tokens.index("help")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # help
        bio[effect_index+1] = "I-E" # with
        bio[effect_index+2] = "I-E" # diabetes
        return bio

    if "USER USER I'm on keto b / c I have T2D . Don't think of keto" in tweet:
        cause_index = tokens.index("T2D")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "keto"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"   # 1st occurrence of keto
        return bio

    if "USER already stated back then that a imbalanced insulin could cause cancer . Something tha" in tweet:
        cause_index = tokens.index("imbalanced")
        bio[cause_index] = "I-C" if ioTagging else "B-C"   # imbalanced
        bio[cause_index+1] = "I-C" # insulin
        effect_indices = [i for i, x in enumerate(tokens) if x == "cancer"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"   # 1st occurrence of cancer
        return bio

    if "USER I understand that progesterone increases insulin resistance . My question is : if women who are taking progesterone" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "progesterone"]
        bio[cause_indices[0]] = "I-C" if ioTagging else "B-C"   # 1st occurrence of progesterone
        effect_index = tokens.index("increases")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # increases
        bio[effect_index+1] = "I-E" # insulin
        bio[effect_index+2] = "I-E" # resistance
        return bio

    if "My diabetic , overweight uncle says Jaggery is healthy and doesn't cause weight gain but sugar does . So" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "sugar"]
        bio[cause_indices[0]] = "I-C" if ioTagging else "B-C"   # 1st occurrence of progesterone
        effect_index = tokens.index("weight")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # weight
        bio[effect_index+1] = "I-E" # gain
        return bio

    if "USER #Insulin4all read the stories . 45 % of diabetics have rationed insulin . My" in tweet:
        cause_index = tokens.index("rationed")
        bio[cause_index] = "I-C" if ioTagging else "B-C"   # rationed
        bio[cause_index+1] = "I-C" # insulin
        effect_indices = [i for i, x in enumerate(tokens) if x == "#Insulin4all"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"   # 1st occurrence of #Insulin4all
        return bio

    if "USER USER USER USER I avoid most carbs because they tend to adversely affect my glucose" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "carbs"]
        bio[cause_indices[0]] = "I-C" if ioTagging else "B-C"   # 1st occurrence of #Insulin4all
        effect_index = tokens.index("adversely")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # adversely
        bio[effect_index+1] = "I-E" # affect
        bio[effect_index+2] = "I-E" # my
        bio[effect_index+3] = "I-E" # glucose
        return bio

    if "USER Yes ! Usually when u pee a lot it could be diabetes or pregnancy but I'm pretty sure that's not th" in tweet:
        cause_index = tokens.index("diabetes")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "pee"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"   # 1st occurrence of pee
        bio[effect_indices[0]+1] = "I-E" # a
        bio[effect_indices[0]+2] = "I-E" # lot
        return bio

    if "Ugh surfing the hypo line most of the night and high ketones this morning . F must be getting sick #T1D #T1Dparenting #T1D" in tweet:
        cause_index = tokens.index("#T1D")
        bio[cause_index] = "I-C" if ioTagging else "B-C" # #T1D
        bio[cause_index+2] = "I-C" if ioTagging else "B-C" # #T1D
        effect_index = tokens.index("hypo")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # hypo
        return bio

    if "When I ask about my glucose ups and downs , my doctor keeps talking about stress . Isn't this" in tweet:
        cause_index = tokens.index("stress") # first occurence
        bio[cause_index] = "I-C" if ioTagging else "B-C" # stress
        effect_index = tokens.index("glucose")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # glucose
        bio[effect_index+1] = "I-E"   # ups
        bio[effect_index+2] = "I-E"   # and
        bio[effect_index+3] = "I-E"   # downs
        return bio

    if "USER I used to be skinny fat . So I went keto / carnivore . No longer skinny fat . But fasting insulin did" in tweet:
        cause_index = tokens.index("eating")
        bio[cause_index] = "I-C" if ioTagging else "B-C"  # eating
        bio[cause_index+1] = "I-C"# too
        bio[cause_index+2] = "I-C"# much
        bio[cause_index+2] = "I-C"# fat
        effect_indices = [i for i, x in enumerate(tokens) if x == "fat"]
        bio[effect_indices[3]] = "I-E" if ioTagging else "B-E"   # fat
        bio[effect_indices[3]+1] = "I-E"   # gain
        return bio

    if "So many people suffer diabetes" in tweet:
        cause_index = tokens.index("diabetes") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("suffer")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER 3 . In our oil example , on Friday you have just found out you are seriously diabetic and cannot eat the sweets ." in tweet:
        cause_index = tokens.index("diabetic") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("can")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # can
        bio[effect_index+1] = "I-E" # not
        bio[effect_index+2] = "I-E" # eat
        bio[effect_index+3] = "I-E" # the
        bio[effect_index+4] = "I-E" # sweets
        return bio

    if "USER I have diabetes . I and about 30M people in the US use insulin ." in tweet:
        cause_index = tokens.index("diabetes") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("use")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # use
        bio[effect_index+1] = "I-E" # insulin
        return bio

    if "I just heard my neighbours sharing wrong information to themselves that sugar is the main cause of diabetes" in tweet:
        cause_index = tokens.index("sugar") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("diabetes")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # use
        return bio

    if "Went to my kids school 3 times today . #T1D is being an asshole . Finally picked him and took him out to lunch . Hopefully he won't go low again today" in tweet:
        cause_index = tokens.index("#T1D") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("go")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # go
        bio[effect_index+1] = "I-E"   # low
        return bio

    if "USER Disabled \ Medicaid here . It saved my life I would have died from hitting diabetic keto-acidosis . I would have died for not being able to afford insulin" in tweet:
        cause_index = tokens.index("keto")
        bio[cause_index] = "I-C" if ioTagging else "B-C"     # keto
        bio[cause_index+1] = "I-C"   # acidosis
        cause_index = tokens.index("not") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"   # not
        bio[cause_index+1] = "I-C" # being
        bio[cause_index+2] = "I-C" # able
        bio[cause_index+3] = "I-C" # to
        bio[cause_index+4] = "I-C" # afford
        bio[cause_index+5] = "I-C" # insulin
        effect_indices = [i for i, x in enumerate(tokens) if x == "died"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER This makes me want to become a drug mule just for insulin . The US prices are ridiculous . I can see high drug prices for shots of botox to eliminate wrinkles , but people need insulin to survive" in tweet:
        causes_indices = [i for i, x in enumerate(tokens) if x == "insulin"]
        bio[causes_indices[-1]] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("survive")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER This makes me want to become a drug mule just for insulin . The US prices are ridiculous . I can see high drug prices for shots of botox to eliminate wrinkles , but people need insulin to survive" in tweet:
        causes_indices = [i for i, x in enumerate(tokens) if x == "diabetic"]
        bio[cause_indices[0]] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("Splenda")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER This makes me want to become a drug mule just for insulin . The US prices are ridiculous . I can see high drug prices for shots of botox to eliminate wrinkles , but people need insulin to survive" in tweet:
        causes_indices = [i for i, x in enumerate(tokens) if x == "diabetes"]
        bio[cause_indices[0]] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("drugs")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "TELLING ME THAT HIS MOTHER WAS A DIABETIC OR WAS SHE A DRUG ADDICT , WHO USED NEEDLES" in tweet:
        cause_index = tokens.index("DIABETIC")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "NEEDLES"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"
        return bio

    if "My diabetic friend is currently in hospital and during the night she had a hypo " in tweet:
        cause_index = tokens.index("hypo") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("hospital") # first occ
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER oh ok . regarding the failing liver thing , when my grandma was hospitalized due to diabetes " in tweet:
        cause_index = tokens.index("diabetes") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("failing") # first occ
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # failin
        bio[effect_index+1] = "I-E" # liver
        return bio

    if "I couldn't bare my diabetic nerve pain any longer and so I talked to my doctor and he prescribed Lyrica now I" in tweet:
        cause_index = tokens.index("diabetic") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"   # diabetic
        bio[cause_index+1] = "I-C" # nerve
        bio[cause_index+2] = "I-C" # pain
        effect_index = tokens.index("prescribed") # first occ
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # prescribed
        bio[effect_index+1] = "I-E" # Lyrica
        return bio

    if "USER USER USER USER So people who can't afford insulin can just go to the ER twice a day" in tweet:
        cause_index = tokens.index("can't") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"   # can't
        bio[cause_index+1] = "I-C" # afford
        bio[cause_index+2] = "I-C" # insulin
        effect_index = tokens.index("ER") # first occ
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # ER
        return bio

    if "Was speaking to a relative who's not been able to fast due to diabetes" in tweet:
        cause_index = tokens.index("diabetes") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"   # can
        effect_index = tokens.index("fast") # first occ
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # fast
        return bio

    if "A hypo at 6pm , a hypo at 3pm and another at 10am . Can I please just have one day with a working pancre" in tweet:
        cause_index = tokens.index("#T1D") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"   # can
        effect_indices = [i for i, x in enumerate(tokens) if x == "hypo"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER Well you should know that people that dies in México from diabetes" in tweet:
        cause_index = tokens.index("diabetes") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"   # diabetes
        effect_index = tokens.index("dies") # first occ
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # dies
        return bio

    if "Boba tea is not the only thing that can give you diabetes . You going to mcdonald's everyday ordering coke , ice cream , fries & burger can give you diabetes" in tweet:
        cause_index = tokens.index("mcdonald") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        cause_index = tokens.index("coke")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        cause_index = tokens.index("ice") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        bio[cause_index+1] = "I-C"
        cause_index = tokens.index("fries")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        cause_index = tokens.index("burger")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "diabetes"]
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER USER USER Your friend could die from diabetes" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "diabetes"]
        bio[cause_indices[0]] = "I-C" if ioTagging else "B-C"   # diabetes
        bio[cause_indices[1]] = "I-C" if ioTagging else "B-C"   # diabetes
        effect_indices = [i for i, x in enumerate(tokens) if x == "die"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"   # dies
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"   # dies
        return bio

    if "As a #t1d I have spent a lot of time at the hospital . Surrounded by others with a similar health illness / disease whatever you want to call it . I am friends with people in USA that have been taken to hospital in critical condition more than once because obtaining insulin" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "hospital"]
        bio[cause_indices[1]] = "I-C" if ioTagging else "B-C"   # hospital
        effect_index = tokens.index("obtaining") # first occ
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        bio[effect_index+1] = "I-E"
        return bio

    if "USER Pay all this money for her insulin every single month plus other medications . Her insulin is the worse one cost me 300 and the other" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "insulin"]
        bio[cause_indices[1]] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("cost") # first occ
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "oddly enough my regular stress and anxiety seem to have ease slightly the last few weeks . ( my diabetes is stress related and my blood sugars have been at near normal level" in tweet:
        cause_index = tokens.index("diabetes")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "stress"]
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER Very lucky to have all my insulin , test strips , Libre etc on the wonderful NHS #longlivetheNHS " in tweet:
        cause_index = tokens.index("NHS")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("insulin")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        effect_index = tokens.index("test")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # test
        bio[effect_index+1] = "I-E" # strips
        effect_index = tokens.index("Libre")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER I love 💗 McDonalds too but it's not good for my diabetes . It's also " in tweet:
        cause_index = tokens.index("McDonalds")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("diabetes")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER The true cause of diabetes : 1 ) soft drinks 2 ) soft drinks 3 ) soft drinks 4 ) everything else " in tweet:
        cause_index = tokens.index("sugar")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        cause_indices = [i for i, x in enumerate(tokens) if x == "soft"]
        bio[cause_indices[0]] = "I-C" if ioTagging else "B-C"
        bio[cause_indices[0]+1] = "I-C"
        bio[cause_indices[1]] = "I-C" if ioTagging else "B-C"
        bio[cause_indices[1]+1] = "I-C"
        bio[cause_indices[2]] = "I-C" if ioTagging else "B-C"
        bio[cause_indices[2]+1] = "I-C"
        effect_index = tokens.index("diabetes")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio


    if "Sick . Tired . Sick and tired . Cancer . Overweight . Diabetic" in tweet:
        cause_index = tokens.index("Diabetic")
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "Sick"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"
        effect_index = tokens.index("Tired")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        effect_index = tokens.index("tired")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        effect_index = tokens.index("Overweight")
        bio[effect_index] = "I-E" if ioTagging else "B-E"
        return bio

    if "USER I have high blood pressure also and I'm a diabetic so a vaccine has to be really safe for me . I've been a diabetic all my life and my blood sugar gets really low" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "diabetic"]
        bio[cause_indices[1]] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "blood"]
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"    # blood
        bio[effect_indices[1]+1] = "I-E"  # sugar
        bio[effect_indices[1]+2] = "I-E"  # gets
        bio[effect_indices[1]+3] = "I-E"  # really
        bio[effect_indices[1]+4] = "I-E"  # low
        return bio

    if "USER I'd say it is , yes . No insulin spike because it's all Keto" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "Keto"]
        bio[cause_indices[0]] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("No")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # no
        bio[effect_index+1] = "I-E" # insulin
        bio[effect_index+2] = "I-E" # spike
        return bio

    if "USER India is a diabetes capital u cant say he dies of diabetes" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "diabetes"]
        bio[cause_indices[1]] = "I-C" if ioTagging else "B-C"
        effect_indices = [i for i, x in enumerate(tokens) if x == "dies"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"    # blood
        return bio

    if "USER USER USER Wrong . I don't care what diet people use to reverse T2 diabetes . I car" in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "diet"]
        bio[cause_indices[0]] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("reverse")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # reverse
        bio[effect_index+1] = "I-E"   # T2
        bio[effect_index+2] = "I-E"   # diabetes
        return bio

    if "Just realized I need to change my diet my dad had type 1 diabetes and my mom had diabetes as well when she was pregnant with me . The chances of me getting diabetes is high if I don't change my diet RN RN" in tweet:
        cause_index = tokens.index("getting")
        bio[cause_index] = "I-C" if ioTagging else "B-C"     # getting
        bio[cause_index+1] = "I-C"     # diabetes
        effect_indices = [i for i, x in enumerate(tokens) if x == "change"]
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"    # change
        bio[effect_indices[1]+1] = "I-E"    # my
        bio[effect_indices[1]+2] = "I-E"    # diet
        return bio

    if "USER USER My son is type 1 Diabetic and I've coached another type 1 and these kids wear that badge on their sleeve ." in tweet:
        cause_index = tokens.index("type") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"     # type
        bio[cause_index+1] = "I-C"     # 1
        bio[cause_index+2] = "I-C"     # Diabetic
        effect_index = tokens.index("wear")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # wear
        bio[effect_index+1] = "I-E"   # that
        bio[effect_index+2] = "I-E"   # badge
        return bio

    if "USER USER I thank you for that , I to have diabetes and what I'm going through health wise is because of the diabetes ! ! I truly understand , I to have heart issues as well as the eyes I was g" in tweet:
        cause_index = tokens.index("diabetes") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"     # diabetes
        effect_index = tokens.index("heart")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # heart
        bio[effect_index+1] = "I-E"   # issues
        return bio

    if "Soooo along with diabetes , I'm getting nerve pain because of my fibromyalgia and yesterday found out I'm getting nerve pain because of the problems with my spine narrowing and disc degeneration How awesome ! Ugh HTTPURL" in tweet:
        cause_index = tokens.index("fibromyalgia") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        cause_index = tokens.index("spine") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"    # spine
        bio[cause_index+1] = "I-C"  # narrowing
        effect_indices = [i for i, x in enumerate(tokens) if x == "nerve"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"
        bio[effect_indices[0]+1] = "I-E"
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"
        bio[effect_indices[1]+1] = "I-E"
        return bio

    if "USER USER USER Or I ate too much carbs that resulted in insulin spike after insulin spike and now my cells won't listen to insulin any more . Insulin resistant , hypoglymecic , pre-diabetes , carb intolerant etc ." in tweet:
        cause_index = tokens.index("ate") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"   # ate
        bio[cause_index+1] = "I-C" # too
        bio[cause_index+2] = "I-C" # much
        bio[cause_index+3] = "I-C" # carbs
        effect_indices = [i for i, x in enumerate(tokens) if x == "insulin"]
        bio[effect_indices[0]] = "I-E" if ioTagging else "B-E"    # insulin
        bio[effect_indices[0]+1] = "I-E"  # spike
        bio[effect_indices[1]] = "I-E" if ioTagging else "B-E"    # insulin
        bio[effect_indices[1]+1] = "I-E"  # spike
        return bio

    if "USER if you die from a gunshot wound but had diabetes , the trauma is what killed you , not the diabetes . i have to believe you know this . if you die from multi system organ failure after contracting this dangerous virus , the virus killed you . please understand ." in tweet:
        cause_index = tokens.index("diabetes") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"   # ate
        effect_index = tokens.index("die")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # heart
        return bio

    if "I took cinnamon when I had \" pre-diabetes . \" Or as I like to call it , early-stage diabetes . It made my burps taste good . It didn't stop the diabetes from murdering my pancreas to the point where I became insulin-dependent ." in tweet:
        cause_indices = [i for i, x in enumerate(tokens) if x == "diabetes"]
        bio[cause_indices[-1]] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("murdering")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # murdering
        bio[effect_index+1] = "I-E"   # my
        bio[effect_index+2] = "I-E"   # pancreas
        return bio

    if "USER That makes no sense . My doctor , with a straight face , said getting diabetes from the statins was good for me . How can that be good because diabetes is INFLAMMATORY to the arteries ? It's just nonsensical ." in tweet:
        cause_index = tokens.index("statins") # first occ
        bio[cause_index] = "I-C" if ioTagging else "B-C"
        effect_index = tokens.index("diabetes")
        bio[effect_index] = "I-E" if ioTagging else "B-E"   # diabetes
        return bio
    else:
        return ["O"] * len(tokens)


def bio_tagging(tweet, causes, effects):
    """
    Each token gets associated to one of the following labels:
    B-C : Begin cause
    I-C : Inside cause
    B-E : Begin effect
    I-C : Inside effect
    O   : Outside
    """

    tokens = normalizeTweet(tweet).split(" ")
    causes = str(causes).strip().split(";")
    effects = str(effects).strip().split(";")
    bio = ["O"] * len(tokens)

    # if no cause and no effect return BIO tags of "O"
    if (not causes and not effects) or (causes == ["nan"] and effects == ["nan"]):
        return bio

    # if only cause and no effect
    if (causes or causes != ["nan"]) and not (effects or effects !=["nan"]):
        print("ERROR: only cause and no effect exists\n\n -------------\n\n")

    # if only effect and no cause
    if not (causes or causes != ["nan"]) and  (effects or effects !=["nan"]):
        print("ERROR: only effect and no cause exists\n\n -------------\n\n")


    ########### SPECIAL CASES (Some causes or effects may occur several times in a tweet. Check here, if the current tweet is such a tweet and return the manually pre-labeled bio tag) ############
    manual_bio_tag = manual_tagging_of_some_special_tweets(tweet, tokens, bio)
    # usually 'bio' has only "O", except if it got altered in manual_tagging_of_some_special_tweets
    if manual_bio_tag.count("O") != len(manual_bio_tag):
        return manual_bio_tag


    ################## Add BIO tags for causes and effects ########################

    for cause in causes: # possible to have several causes

        cause_words = normalizeTweet(cause).split(" ") # a cause may consist of several words
        cause_words_start = cause_words[0]
        try:
            ### Find index of first word of cause -> label with "B-C"
            indices = [i for i, x in enumerate(tokens) if x == cause_words_start] # get all indices of the first word of the cause
            N_cause_words = len(cause_words)
            if len(indices) > 1 and N_cause_words > 1: # if several occurrences of the same cause start word in phrase
                for cause_word_start_index in indices:
                    causeIndexFound = all([tokens[cause_word_start_index+word_i] == cause_words[word_i] for word_i in range(N_cause_words)])
                    if causeIndexFound:
                        ind = cause_word_start_index
                        break
            else:
                ind = tokens.index(cause_words_start) # get index of causal word in tokens list
            bio[ind] = "B-C"

            ### If cause consists of several words -> label those words with "I-C"
            i = 1
            while i < len(cause_words):
                if tokens[ind+i] == cause_words[i]:
                    bio[ind+i] = "I-C"
                else:
                    print("Error: token and causal word don't match!\Tind:", ind, "i:", i, "token[ind+i]:", tokens[ind+i], "cause_words[i]:", cause_words[i])
                i += 1
        except ValueError:
            print("\nINFO: cause word '{}' does not exist in sentence: \n'{}', but should be in other sentence of the tweet".format(cause_words_start, tokens))


    for effect in effects: # possible to have several effects

        effect_words = normalizeTweet(effect).split(" ") # a effect may consist of several words
        effect_words_start = effect_words[0]
        try:
            ### Find index of first word of effect -> label with "B-E"
            indices = [i for i, x in enumerate(tokens) if x == effect_words_start]
            N_effect_words = len(effect_words)
            if len(indices) > 1 and N_effect_words > 1: # if several occurrences of the same cause start word in phrase
                for effect_word_start_index in indices:
                    effectIndexFound = all([tokens[effect_word_start_index+word_i] == effect_words[word_i] for word_i in range(N_effect_words)])
                    if effectIndexFound:
                        ind = effect_word_start_index
                        break
            else:
                ind = tokens.index(effect_words_start) # get index of c_word in tokens list
            bio[ind] = "B-E"

            ### If effect consists of several words -> label those words with "I-E"
            i = 1
            while i < len(effect_words):
                if tokens[ind+i] == effect_words[i]:
                    bio[ind+i] = "I-E"
                else:
                    print("Error: token and effect word don't match! \tind:", ind, "i:", i, "token[ind+i]", tokens[ind+i], "effect_words[i]", cause_words[i])
                i += 1

        except ValueError:
            print("\nError: effect word '{}' does not exist in sentence: \n'{}', but should be in other sentence of the tweet".format(effect_words_start, tokens))

    return bio

def split_into_sentences(text):
    """ Split tweet into sentences """

    text = " " + text + "  "
    text = text.replace("\n"," ")
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace("..", "<POINTPOINT>")
    text = text.replace(".",".<stop>")
    text = text.replace("<POINTPOINT>", "..")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    sentences = [s  for s in sentences if s != ""]
    return sentences


def create_training_data(data):
    """
        Transform tweets into sentences and associate to each sentence a causal
        label (0,1) and its BIO tags.
    """
    tweets = []
    causal_labels = []
    bio_labels = []

    for i, row in data.iterrows():
        sentences = split_into_sentences(row["full_text"])
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




class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Class taken from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

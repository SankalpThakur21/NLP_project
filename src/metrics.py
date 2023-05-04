from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu
import nltk
import sklearn
from utils import remove_unwanted
import scipy
import pandas as pd

def normalize(data):
      return sklearn.preprocessing.normalize(data)

def get_BLEU_score(sentence, reference):
    return sentence_bleu([reference], sentence,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2)

def rouge_n(sentence , reference,n):
    r_ngrams = list(ngrams(reference ,n ))
    s_ngrams = list(ngrams(sentence , n))
    matches =0
    if(len(s_ngrams) ==0 or len(r_ngrams) ==0 ):
        return 0
    for x in s_ngrams:
        if(x in r_ngrams):
            matches +=1
    recall = matches / len(r_ngrams)
    precision = matches/len(s_ngrams)
    if(precision + recall ==0) :
        return 0
    return 2 * (precision*recall)  /(precision+recall)

def get_ROUGE_L(trans_words, ref_words):
    m, n = len(trans_words), len(ref_words)
    lcs = [[None]*(n+1) for i in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                lcs[i][j] = 0
            elif trans_words[i-1] == ref_words[j-1]:
                lcs[i][j] = lcs[i-1][j-1] + 1
            else:
                lcs[i][j] = max(lcs[i-1][j] , lcs[i][j-1])
    ans = lcs[m][n]
    return ans

def ngram_overlap(trans_words, ref_words, n):
    trans_ngrams = list(ngrams(trans_words, n=n))
    ref_ngrams = list(ngrams(ref_words, n=n))
    intersect = len([i for i in ref_ngrams if i in trans_ngrams])
    union = len(ref_ngrams) + len(trans_ngrams) - intersect
    if union == 0:
        return 0
    return intersect / union



data  =pd.read_table('data/data.tsv',usecols=['Source' , 'Shortening' , 'AverageMeaning','AverageGrammar'])
cdata = data[data['AverageMeaning']<=3 ][data['AverageGrammar']<=3]
data.dropna(inplace=True)
ref = list(cdata['Source'])
mt = list(cdata['Shortening'])
avgm = list(cdata['AverageMeaning'])
avgmg = list(cdata['AverageGrammar'])
src , trans , mng , grm = remove_unwanted(mt ,ref ,avgm,avgmg)

print("loaded data calulating metrics ")

bleu_scores = []
rougel_scores = []
rouge2_scores = []
rouge3_scores = []
ngoverlap2_score = []
ngoverlap3_score = []
features =[]
for x,y in zip(trans , src):
    bleu_scores.append(get_BLEU_score(x,y))
    rougel_scores.append(get_ROUGE_L(x,y))
    rouge2_scores.append(rouge_n(x,y,2))
    rouge3_scores.append(rouge_n(x,y,3))
    ngoverlap2_score.append(ngram_overlap(x,y,2))
    ngoverlap3_score.append(ngram_overlap(x,y,3))
    features.append([bleu_scores[-1] , rougel_scores[-1] ,rouge2_scores[-1] ,rouge3_scores[-1] ,ngoverlap2_score[-1] ,ngoverlap3_score[-1]])



bleu_cor = []
rougel_cor = []
rogue2_cor = []
rogue3_cor = []
ngoverlap2_cor = []
ngoverlap3_cor = []


for x in range(11):
    w1 = x/10
    w2 = round(1 - w1,2)
    fluency = [w1* a +  w2*b for a,b in zip(mng ,grm)]
    
    bleu_cor.append(scipy.stats.pearsonr(bleu_scores , fluency)[0])
    rougel_cor.append(scipy.stats.pearsonr(rougel_scores , fluency)[0])
    rogue2_cor.append(scipy.stats.pearsonr(rouge2_scores , fluency)[0])
    rogue3_cor.append(scipy.stats.pearsonr(rouge3_scores , fluency)[0])
    ngoverlap2_cor.append(scipy.stats.pearsonr(ngoverlap2_score , fluency)[0])
    ngoverlap3_cor.append(scipy.stats.pearsonr(ngoverlap3_score , fluency)[0])
cor = [bleu_cor,rougel_cor,rogue2_cor,rogue3_cor,ngoverlap2_cor,ngoverlap3_cor]
for x in range(11):
    w1 = x/10
    w2 = round(1 - w1,2)
    fluency = [w1* a +  w2*b for a,b in zip(mng ,grm)]
    tot = 0
    for z in cor:
        tot+=z[x]
    tot = tot /len(cor)
    print("averageMeaning weight : {} , averageGrammar weight : {}  correlation to fluency : {}".format(w1 ,w2 , tot))
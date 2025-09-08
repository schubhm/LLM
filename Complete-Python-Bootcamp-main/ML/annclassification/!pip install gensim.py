!pip install gensim
import gensim
from gensim.models import Word2Vec, KeyedVectors
## References: https://stackoverflow.com/questions/46433778/import-googlenews-vectors-negative300-bin

import gensim.downloader as api

wv = api.load('word2vec-google-news-300')

vec_king = wv['king']
vec_king
vec_king.shape
wv['cricket']
wv.most_similar('cricket')
wv.most_similar('happy')
wv.similarity("hockey","sports")
vec=wv['king']-wv['man']+wv['woman']
vec
wv.most_similar([vec])

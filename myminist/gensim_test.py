from gensim import corpora


texts = [['范德萨', 'interface', 'computer'],
['survey', 'user', 'computer', 'system', 'response', 'time'],
['eps', 'user', 'interface', 'system'],
['system', 'human', 'system', 'eps'],
['user', '范德萨', 'time'],
['trees'],
['graph', 'trees'],
['graph', 'minors', 'trees'],
['graph', '范德萨', 'survey']]


for word in texts[4]:
    print(word)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus[0]) # [(0, 1), (1, 1), (2, 1)]


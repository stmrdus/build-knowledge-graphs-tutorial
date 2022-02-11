# Tutorial: Build a Knowledge Graph and apply KGE Techniques for Link Prediction

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/nhutnamhcmus/build-knowledge-graphs-tutorial"><a href="https://github.com/nhutnamhcmus/build-knowledge-graphs-tutorial/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/nhutnamhcmus/build-knowledge-graphs-tutorial"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/nhutnamhcmus/build-knowledge-graphs-tutorial">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/nhutnamhcmus/build-knowledge-graphs-tutorial">
<a href="https://github.com/nhutnamhcmus/build-knowledge-graphs-tutorial/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/nhutnamhcmus/build-knowledge-graphs-tutorial"></a>
<a href="https://github.com/nhutnamhcmus/build-knowledge-graphs-tutorial/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/nhutnamhcmus/build-knowledge-graphs-tutorial"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/nhutnamhcmus/build-knowledge-graphs-tutorial">


## A brief introduction to Web Scraping


## Introduction to Knowledge Graphs


## Build a simple Knowledge Graph with Python

### Install dependencies

```
!pip install wikipedia-api pandas spacy networkx scipy
```

```
!python -m spacy download en
```

### Scrape data

Import section

```python
import wikipediaapi
import pandas as pd
import concurrent.futures
from tqdm import tqdm
```

```python
def scrape_wikipedia(name_topic, verbose=True):
   def link_to_wikipedia(link):
       try:
           page = api_wikipedia.page(link)
           if page.exists():
               return {'page': link, 'text': page.text, 'link': page.fullurl, 'categories': list(page.categories.keys())}
       except:
           return None
      
   api_wikipedia = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
   name_of_page = api_wikipedia.page(name_topic)
   if not name_of_page.exists():
       print('Page {} is not present'.format(name_of_page))
       return
  
   links_to_page = list(name_of_page.links.keys())
   procceed = tqdm(desc='Scraped links', unit='', total=len(links_to_page)) if verbose else None
   origin = [{'page': name_topic, 'text': name_of_page.text, 'link': name_of_page.fullurl, 'categories': list(name_of_page.categories.keys())}]
  
   with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
       links_future = {executor.submit(link_to_wikipedia, link): link for link in links_to_page}
       for future in concurrent.futures.as_completed(links_future):
           info = future.result()
           origin.append(info) if info else None
           procceed.update(1) if verbose else None
   procceed.close() if verbose else None
  
   namespaces = ('Wikipedia', 'Special', 'Talk', 'LyricWiki', 'File', 'MediaWiki', 'Template', 'Help', 'User', 'Category talk', 'Portal talk')
   origin = pd.DataFrame(origin)
   origin = origin[(len(origin['text']) > 20) & ~(origin['page'].str.startswith(namespaces, na=True))]
   origin['categories'] = origin.categories.apply(lambda a: [b[9:] for b in a])

   origin['topic'] = name_topic
   print('Scraped pages', len(origin))
  
   return origin
```

```python
wiki_data_covid = scrape_wikipedia('COVID 19')
wiki_data_covid.to_csv('scraped_covid_data.csv')
```

### Sentence segmentation

Import libraries

```python
import requests

import pandas as pd

import spacy
from spacy import displacy
from spacy.tokens import Span
from spacy.matcher import Matcher
 
import matplotlib.pyplot as plot
import networkx as ntx

from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')
```

```python
data = pd.read_csv('scraped_covid_data.csv')
document = nlp(data['text'][10])
for tokn in document:
   print(tokn.text, "---", tokn.dep_)
```

### Entity extraction

```python
def extract_entities(sents):
   # chunk one
   enti_one = ""
   enti_two = ""
  
   dep_prev_token = "" # dependency tag of previous token in sentence
  
   txt_prev_token = "" # previous token in sentence
  
   prefix = ""
   modifier = ""
  
   for tokn in nlp(sents):
       # chunk two
       ## move to next token if token is punctuation
       if tokn.dep_ != "punct":
           #  check if token is compound word or not
           if tokn.dep_ == "compound":
               prefix = tokn.text
               # add the current word to it if the previous word is 'compound’
               if dep_prev_token == "compound":
                   prefix = txt_prev_token + " "+ tokn.text
                  
           # verify if token is modifier or not
           if tokn.dep_.endswith("mod") == True:
               modifier = tokn.text
               # add it to the current word if the previous word is 'compound'
               if dep_prev_token == "compound":
                   modifier = txt_prev_token + " "+ tokn.text
                  
           # chunk3
           if tokn.dep_.find("subj") == True:
               enti_one = modifier +" "+ prefix + " "+ tokn.text
               prefix = ""
               modifier = ""
               dep_prev_token = ""
               txt_prev_token = ""
              
           # chunk4
           if tokn.dep_.find("obj") == True:
               enti_two = modifier +" "+ prefix +" "+ tokn.text
              
           # chunk 5
           # update variable
           dep_prev_token = tokn.dep_
           txt_prev_token = tokn.text
          
   return [enti_one.strip(), enti_two.strip()]
```

```python
pairs_of_entities = []
for i in tqdm(data['text'][:800]):
   pairs_of_entities.append(extract_entities(i))
```

### Relation extraction

```python
def obtain_relation(sent):
  
   doc = nlp(sent)
  
   matcher = Matcher(nlp.vocab)
  
   pattern = [[{'DEP':'ROOT'},
           {'DEP':'prep','OP':"?"},
           {'DEP':'agent','OP':"?"}, 
           {'POS':'ADJ','OP':"?"}]]
  
   #matcher.add("matching_1", None, pattern)
   matcher.add('matching_1', pattern)
  
   matcher = matcher(doc)
   h = len(matcher) - 1
  
   span = doc[matcher[h][1]:matcher[h][2]]
  
   return (span.text)

relations = [obtain_relation(j) for j in (data['text'][:800])]
```

### Build a Knowledge Graph

```python
# subject extraction
source = [j[0] for j in pairs_of_entities]

# object extraction
target = [k[1] for k in pairs_of_entities]

# graph dataframe
data_kgf = pd.DataFrame({'source':source, 'edge':relations, 'target':target})
data_kgf.to_csv('graph.csv', index=False)
```

```python
# Create DG from the dataframe
graph = ntx.from_pandas_edgelist(data_kgf, "source", "target", edge_attr=True, create_using=ntx.MultiDiGraph())
```

```python
# plotting the network
plot.figure(figsize=(14, 14))
posn = ntx.spring_layout(graph)
ntx.draw(graph, with_labels=True, node_color='green', edge_cmap=plot.cm.Blues, pos = posn)
plot.show()
```

```python
graph = ntx.from_pandas_edgelist(data_kgf[data_kgf['edge']=="Retrieved"], "source", "target", edge_attr=True, create_using=ntx.MultiDiGraph())
plot.figure(figsize=(14,14))
pos = ntx.spring_layout(graph, k = 0.5) # k regulates the distance between nodes
ntx.draw(graph, with_labels=True, node_color='green', node_size=1400, edge_cmap=plot.cm.Blues, pos = posn)
plot.show()
```

```python
graph = pd.read_csv('graph.csv', sep=',', names=['from', 'to', 'rel'])
graph.head()
```
## Apply KGE model to solve link prediction problem

TransE (translations in the embedding space) is a method which models relationships by interpreting them as translations operating on the low-dimensional embeddings of the entities.

```python
# You must install pytorch with CUDA support, version depends on your choice.
!pip install torchkge==0.16.25 -q
```

```python
import torch
from torch import nn
from torch import cuda
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from torchkge import KnowledgeGraph
from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.models.interfaces import TranslationModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
```

#### Define base class for TransE

```python
class BaseTransE(TranslationModel):
    def __init__(self, num_entities, num_relations, dim=100):
        super(BaseTransE, self).__init__(num_entities, num_relations, dissimilarity_type='L2')
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        self.ent_embeddings = nn.Embedding(num_entities, self.dim)
        self.rel_embeddings = nn.Embedding(num_relations, self.dim)
        
        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_embeddings.weight.data)

        self.normalize_parameters()
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)

    def normalize_parameters(self):
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)

    def get_embeddings(self):
        self.normalize_parameters()
        return self.ent_embeddings.weight.data, self.rel_embeddings.weight.data
        
    def lp_prep_cands(self, h_idx, t_idx, r_idx):
        b_size = h_idx.shape[0]

        h_emb = self.ent_embeddings(h_idx)
        t_emb = self.ent_embeddings(t_idx)
        r_emb = self.rel_embeddings(r_idx)

        candidates = self.ent_embeddings.weight.data.view(1, self.num_entities, self.dim)
        candidates = candidates.expand(b_size, self.num_entities, self.dim)

        return h_emb, t_emb, candidates, r_emb
    
    def forward(self, h, t, nh, nt, r):
        return self.scoring_function(h, t, r), self.scoring_function(nh, nt, r)

    @staticmethod
    def l2_dissimilarity(a, b):
        assert len(a.shape) == len(b.shape)
        return (a-b).norm(p=2, dim=-1)**2

    @staticmethod
    def l1_dissimilarity(a, b):
        assert len(a.shape) == len(b.shape)
        return (a-b).norm(p=1, dim=-1)
```

#### Define class for TransE

```python
class TransE(BaseTransE):
    def scoring_function(self, h, t, r):
        h = F.normalize(self.ent_embeddings(h), p=2, dim=1)
        t = F.normalize(self.ent_embeddings(t), p=2, dim=1)
        r = self.rel_embeddings(r)
        scores = -torch.norm(h + r - t, 2, -1)
        return scores
```

#### Define the MarginLoss class

```python
class MarginLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=margin, reduction='mean')
    def forward(self, positive_scores, negative_scores):
        return self.loss(positive_scores, negative_scores, target=torch.ones_like(positive_scores))
```

#### Training model

```python
# Define model
model = TransE(kg_train.n_ent, kg_train.n_rel, dim=64)

# Define criterion for training model
criterion = MarginLoss(margin=0.5)
```

```python
# Move everything to CUDA if available
if cuda.is_available():
    cuda.empty_cache()
    model.cuda()
    criterion.cuda()
```

```python
# Define the torch optimizer to be used
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# Define negative sampler
sampler = BernoulliNegativeSampler(kg_train)

# Define Dataloader
dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda='all')

# Training loop
iterator = tqdm(range(n_epochs), unit='epoch')
for epoch in iterator:
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        h, t, r = batch[0], batch[1], batch[2]
        n_h, n_t = sampler.corrupt_batch(h, t, r)

        optimizer.zero_grad()

        # forward + backward + optimize
        pos, neg = model(h, t, n_h, n_t, r)
        loss = criterion(pos, neg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    iterator.set_description(
        'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, running_loss / len(dataloader)))

model.normalize_parameters()
```

#### Evaluation model

```python
# Define evaluator
evaluator = LinkPredictionEvaluator(model, kg_test)

# Run evaluator
evaluator.evaluate(b_size=128)
```

```python
# Show results
print("----------------Overall Results----------------")
print('Hit@10: {:.4f}'.format(evaluator.hit_at_k(k=10)[0]))
print('Hit@3: {:.4f}'.format(evaluator.hit_at_k(k=3)[0]))
print('Hit@1: {:.4f}'.format(evaluator.hit_at_k(k=1)[0]))
print('Mean Rank: {:.4f}'.format(evaluator.mean_rank()[0]))
print('Mean Reciprocal Rank : {:.4f}'.format(evaluator.mrr()[0]))
```
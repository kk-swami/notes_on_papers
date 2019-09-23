# Citation  

Distant Supervision for relation extraction without labeled data  
Mintz, Bills, Snow, Jurafsky  ACL 2009  

# Tags  

distant supervision, relation extraction  

# Significance

Avoid need for manual labeling of sentences for building ML models for relation extraction, by using an external knowledge base (freebase was used by the authors)
to provide weak supervision to sentences. This ensures that instead of having 1000 labeled examples by hand, we are able to obtain orders of magnitude more (noisy) labeled examples automatically


# Context and summary  

Previously Existing Techiques :  
1) Purely supervised classifiers for relation extraction need manual annotation which is expensive, and can be biased to domain. 
2) Purely Unsupervised techniques extract strings between entity pairs in a large corpus of text, and clusters these strings to produce relation strings, however the resulting relations
may be hard to map to a database  
3) Bootstrap learning - Using a small set of seeds (seeds are tuples of entity pairs and relation between them) on a corpus, get a new set of patterns, use patterns to extract more tuples, continue iteratively  
Patterns thus extracted can suffer from low precision and semantic drift  

In contrast, distant supervision can label millions of sentences containing given entity pair automatically (if noisily) using the knowledge base.  The noisiness of labels  is overcome by the orders of magnitude of data for which supervision is 
thus generated

  
For example , if (born_in,barack_obama,hawai) is there in the knowledge base, where born_in is the relation, and barack_obama and hawai are the two entities,  then we search of all sentences in text corpus where barack_obama and hawaii are present,
and label the relation between these entity pairs as born_in.
This is done for every entity pair in the KB; and so we have millions of labeled examples which can be use used to train an ML classifer; so that given an unknown pair of entities and an unknown sentence, the trained ML classifier can be used to predict the relation

Also, there are two kinds of slightly different relation extraction problems we are interested in  
1) Given an entity pair and given a sentence in which the entity pair occurs, we want to predict the relation between the entity pairs in _that sentence_  
Eg : sentence 1 : India won the world cup in 2011  entity pair : (India,2011) -> relation predicted should be won  
2) Given an entity pair, we want to predict relation between them to populate a knowledge base. Here , unlike example 1), the relation is a global relation between the two entities independent of context  (essentially static)
Eg : we want to fill an entry (is_born,hawai,barack_obama) in the DB which we don't know before.
So we look at ALL sentences where (hawai,barack_obama) exists in corpus; pool features from all these sentences to get essentially one set of pooled features for the tuple agnostic of the context, and give a global prediction  
This paper focuses on task 2)  


# Method in more detail  

1) Use the KB freebase for distant supervision. At the time of this paper (July 2008), Freebase had 116 million instances of 7300 relations between 9 million entities. After some filtering,
this was narrowed to to 1.8 million instances of 102 relations connecting 940000 entities
2) Training - For every entity pair and relation tuple which exists in KB
        1) Identify all sentences which contain the entity pair (Entity mapping is done by an NER tagger)
        2) Get features from each of these sentences, and concat to form features for that entity pair
        3) Have label of training data as the relation which exists in KB  
        4) For negative classes, select entity pairs randomly which do not occur in the KB, and extract features for them. There could be false negatives (two entities selected randomly are actually related, but relation is not present in KB); but this problem is estimated to be small
        5) train ML model of choice (this paper uses a logistic regression model with gaussian/L2 regularization). Note that this is a multi-label, multi class classifer (ie : two entities can be related by more than one relationship)
        For example, both (is_director,steven_spielberg, saving_private_ryan) and (is_writer,steven_spielberg,saving_private_ryan) could be true. Therefore, given entity pair (steven_spielberg,saving_private_ryan); both is_writer and is_director should be predicted  
        
3) Testing
        1) Identify all entities in corpus using NER tagger
        2) For every pair of entities which we want a label for (for which label is not already present in KB :) ) - get features from each and every sentence which holds that entity pair; pool
        (Example) : Given entity pair (dickens,david copperfield); we find all sentences where both dickens and david copperfield are present; Let's say there are 10 such sentences in corpus
        If we extract 3 features from each sentence, totally, we get 3*10 = 30 features which we pool  
        3) Apply model, get prediction of relation between entity pair     
        4) Since features from multiple sentences are potentially combined for each prediction, we get information which may not have been present in a single sentence  
        Eg: in paper : Given tuple (steven_spielberg, saving_private_ryan), we want to predict relation is_director  
        First sentence is : Steven Spielberg’s film Saving Private Ryan is loosely based on the brothers’ story.  
        Second sentence is : Allison co-produced the Academy Award- winning Saving Private Ryan, directed by Steven Spielberg  
        First sentence could mean that spielberg is a writer or a director or a producer; but mentions that saving private ryan is a film
        second sentence does not mention that saving private ryan is a film; but mentions that Spielberg is director.  
        If we combine features from both, we identify that (steven_spielberg, saving_private_ryan) should give a relation is_director  
        5) Features used - Lexical and syntactic features
                Eg : Given entity pair, for each sentence in which that pair appears  
                Lexical Features  
                    1) sequence of words between entities (bow)  
                    2) POS tags of these words  
                    3) flag indicating which entity came first  (note - directionality is important in relation extraction)  
                    4) window of k words to left of entity 1 and their POS tags  
                Syntactic features  
                    1) Dependency path between both entities
                    2) For each entity, 1 node it is connected to which is NOT a part of the dependency part between both entities  
                NER features
                    1) entity name for both the entities found by NER. This paper uses 4 classes - person/location/organization/miscellaneous/None  
        See table 3 in paper for examples of features in one particular context              

4) Evaluation  

      1) Done in two ways -
          i) Hold out a part of the KB from training, compute metrics automatically  
                  See figure 2 in paper
          ii) Use human evaluation  
                  See Table 5 in paper
        
        

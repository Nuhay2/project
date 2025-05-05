from abc import ABC
from abc import abstractmethod
import networkx as nx
import numpy as np
import random

# Ce programme calcul les métriques pour un graphe (pour calculer sur plusiseurs graphes j'ai relancé ce programme avec des graphes différents
G1=nx.read_gml("../../fb100/data/Caltech36.gml")

class LinkPrediction(ABC):
    def __init__(self,graph):
        self.graph=graph
        self.N=len(graph)
    def neighbors(self, v):
        neighbors_list=self.graph.neighbors(v)
        return list(neighbors_list)


    @abstractmethod
    def score(self):
        pass


class CommonNeighbors(LinkPrediction):
    def __init__(self, graph):
        super(CommonNeighbors, self).__init__(graph)
    def score(self,u,v):
        neighbors_u=set(self.neighbors(u))
        neighbors_v=set(self.neighbors(v))
        return (neighbors_u & neighbors_v)

class Jaccard(LinkPrediction):
    def __init__(self, graph):
        super(Jaccard,self).__init__(graph)

    def score(self, u, v):
        neighbors_u = set(self.neighbors(u))
        neighbors_v = set(self.neighbors(v))
        intersection=(neighbors_u & neighbors_v)
        union=(neighbors_u | neighbors_v)
        if not union:
            return 0
        return len(intersection)/len(union)

class AdamicAdar(LinkPrediction):
    def __init__(self, graph):
        super(AdamicAdar,self).__init__(graph)

    def score(self, u, v):
        neighbors_u = set(self.neighbors(u))
        neighbors_v = set(self.neighbors(v))
        intersection= (neighbors_u & neighbors_v)
        score=0.0
        for z in intersection:
            deg = len(self.neighbors(z))
            if deg>1:
                score+=1/np.log(deg)
        return score

# return le graph avec les edges removed et la liste des edges removed
def remove_random_edges(G, fraction):
    G_removed=G.copy()
    edges=list(G.edges())
    num_remove=int(len(edges)*fraction)
    removed_edges=random.sample(edges, num_remove)
    G_removed.remove_edges_from(removed_edges)
    return G_removed, removed_edges

# return les couples de noeuds non connectés
def get_non_edges(G):
    return list(nx.non_edges(G))

# création du tableau avec les couples et leur score
def get_prediction_scores(predictor, non_edges):
    scores=[]
    for u, v in non_edges:
        score=predictor.score(u,v)
        scores.append(((u,v), score))
    # on trie dans l'ordre décroissant des scores
    return sorted(scores, key=lambda x: x[1], reverse=True)

# calcul des critères d'évaluation
def evaluate_predictions(predicted_edges, removed_edges, ks=[50, 100, 200, 400]):
    removed_set=set((min(u,v), max(u,v)) for u,v in removed_edges)
    results={}

    for k in ks:
        # k meilleurs scores
        top_k=set((min(u,v), max(u,v)) for (u,v), p in predicted_edges[:k])
        tp=len(top_k & removed_set)
        fp= k - tp
        fn=len(removed_set)-tp

        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        top_k_rate=tp/k

        results[k]={
            "TP": tp,
            "Précision": precision,
            "Recall": recall,
            "Top_rate": top_k_rate
        }
    return results

G_partial, E_removed=remove_random_edges(G1, fraction=0.1)
# On choisi la méthode de prédiction de lien entre (CommonNeighbors, Jaccard ou AdamicAdar
predictor=Jaccard(G_partial)
non_edges=get_non_edges(G_partial)
predicted_scores=get_prediction_scores(predictor,non_edges)
metrics=evaluate_predictions(predicted_scores, E_removed)

for k in metrics:
    print(f"k={k} : {metrics[k]}")




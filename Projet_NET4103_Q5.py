import networkx as nx
import random
import torch

# Ce code permet de faire du label propagation sur les attributs demandés et comparer avec des bouts supprimés de différentes tailles
# Il ne charge que 1 graphe en même temps
G=nx.read_gml("../../fb100/data/Swarthmore42.gml")

# méthode pour supprimer une partie des attributs
def remove_attributes(G, attribute, fraction):
    removed_nodes = []
    nodes_with_attr1=list(G.nodes(data=True))
    nodes_with_attr2=[]
    for n, data in nodes_with_attr1:
        if not attribute in data:
            continue
        nodes_with_attr2.append((n,data[attribute]))

    num_to_remove=int(len(nodes_with_attr2)*fraction)
    nodes_to_remove=random.sample(nodes_with_attr2, num_to_remove)
    for node, k in nodes_to_remove:
        del G.nodes[node][attribute]
        removed_nodes.append(node)

    return removed_nodes

# méthode de label propagation (facultatif)
def label_propagation(G, max_iter=100):
    nodes = list(G.nodes)
    N = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}


    labels = torch.arange(N)

    for k in range(max_iter):
        prev_labels = labels.clone()

        random.shuffle(nodes)

        for node in nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            neighbors_index = [node_index[n] for n in neighbors]
            neighbor_labels = labels[neighbors_index]

            unique, counts = torch.unique(neighbor_labels, return_counts=True)
            max_count = counts.max()
            top_labels = unique[counts == max_count]
            labels[node_index[node]] = random.choice(top_labels).item()

        if torch.equal(prev_labels, labels):
            break

    return {node: labels[node_index[node]].item() for node in nodes}

# méthode de label propagation pour les attributs en particulier
def label_propagation_for_attributes(G, attribute, max_iter=100):
    nodes = list(G.nodes)
    N = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    labels = torch.full((N,), -1)  # Valeur initiale = -1 (pour indiquer "manquant")
    is_known = (labels != -1)

    # Initialiser les labels avec les valeurs existantes pour l'attribut
    for i, node in enumerate(nodes):
        if attribute in G.nodes[node]:
            labels[i] = G.nodes[node][attribute]

    for k in range(max_iter):
        prev_labels = labels.clone()
        random.shuffle(nodes)

        for node in nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue
            neighbors_index = [node_index[n] for n in neighbors]
            neighbor_labels = labels[neighbors_index]

            unique, counts = torch.unique(neighbor_labels, return_counts=True)
            max_count = counts.max()
            top_labels = unique[counts == max_count]
            if not is_known[node_index[node]]:  # On ne met à jour que si inconnu
                labels[node_index[node]] = random.choice(top_labels).item()
        if torch.equal(prev_labels, labels):
            break

    # Assigner les labels récupérés aux nœuds du graph
    for i, node in enumerate(nodes):
        if labels[i] != -1:
            G.nodes[node][attribute] = labels[i]

    return G

# Calcul la précision selon notre modèle
def evaluate_recovery(original_G, recovered_G, attribute, nodes_to_check):
    correct = 0
    total = 0

    for node in nodes_to_check:
        if attribute in original_G.nodes[node] and attribute in recovered_G.nodes[node]:
            total += 1
            if original_G.nodes[node][attribute] == recovered_G.nodes[node][attribute]:
                correct += 1

    precision = correct / total if total > 0 else 0
    print(f"Précision pour {attribute}: {precision:.4f}")


# Attributs à traiter
attributes = ["dorm", "major_index", "gender"]
fractions = [0.1, 0.2, 0.3]


# Appliquer la propagation des labels pour chaque attribut et fraction
for attribute in attributes:
    for fraction in fractions:
        print(f"Suppression de {fraction * 100}% de {attribute}")

        # Supprimer les attributs de manière aléatoire
        G1_copy = G.copy()  # Créer une copie pour ne pas altérer l'original
        removed_nodes = remove_attributes(G1_copy, attribute, fraction)

        # Appliquer l'algorithme de propagation des labels
        G1_copy = label_propagation_for_attributes(G1_copy, attribute)
        evaluate_recovery(G, G1_copy, attribute, removed_nodes)



import networkx as nx
import matplotlib.pyplot as plt

# Chargement des 3 graphes
G1=nx.read_gml("../../fb100/data/Caltech36.gml")
G2=nx.read_gml("../../fb100/data/MIT8.gml")
G3=nx.read_gml("../../fb100/data/Johns Hopkins55.gml")


degrees1 = [deg for node, deg in G1.degree()]
degrees2 = [deg for node, deg in G2.degree()]
degrees3 = [deg for node, deg in G3.degree()]

# Pour afficher les trois figures en même temps
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 ligne, 3 colonnes

# Graphe 1
axes[0].hist(degrees1, bins=range(min(degrees1), max(degrees1)+2), align='left',rwidth=0.8, color='skyblue', edgecolor='black')
axes[0].set_title("Graphe 1")
axes[0].set_xlabel("Degré")
axes[0].set_ylabel("Nombre de nœuds")
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Graphe 2
axes[1].hist(degrees2, bins=range(min(degrees2), max(degrees2)+2), align='left',rwidth=0.8, color='red', edgecolor='black')
axes[1].set_title("Graphe 2")
axes[1].set_xlabel("Degré")
axes[1].set_ylabel("Nombre de nœuds")
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Graphe 3
axes[2].hist(degrees3, bins=range(min(degrees3), max(degrees3)+2), align='left',rwidth=0.8, color='green', edgecolor='black')
axes[2].set_title("Graphe 3")
axes[2].set_xlabel("Degré")
axes[2].set_ylabel("Nombre de nœuds")
axes[2].grid(axis='y', linestyle='--', alpha=0.7)


# Mise en forme finale
plt.suptitle("Distributions des degrés des 3 graphes", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Laisser un peu de place pour le titre
plt.show()

# Affichage données de clustering et de densités de chaque graphe
print("[LOCAL CLUSTERING] Graphe 1: ", nx.average_clustering(G1) )
print("[GLOBAL CLUSTERING] Graphe 1: ", nx.transitivity(G1))
print("[GRAPH DENSITY] Graphe 1: ", nx.density(G1))

print("[LOCAL CLUSTERING] Graphe 2: ", nx.average_clustering(G2) )
print("[GLOBAL CLUSTERING] Graphe 2: ", nx.transitivity(G2))
print("[GRAPH DENSITY] Graphe 2: ", nx.density(G2))

print("[LOCAL CLUSTERING] Graphe 3: ", nx.average_clustering(G3) )
print("[GLOBAL CLUSTERING] Graphe 3: ", nx.transitivity(G3))
print("[GRAPH DENSITY] Graphe 3: ", nx.density(G3))

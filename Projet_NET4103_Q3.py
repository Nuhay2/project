import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import seaborn as sns
from tqdm import tqdm  # Pour une barre de progression

# Configuration
DATA_DIR = "fb100/data/"
ATTRIBUTES = ["student_fac", "gender", "major_index", "dorm", "year"]
RESULTS_FILE = "assortativity_results.csv"


def process_single_graph(file_path):
    """Charge un graphe, calcule les métriques et le libère de la mémoire"""
    try:
        name = os.path.splitext(os.path.basename(file_path))[0]
        G = nx.read_gml(file_path)
        n = G.number_of_nodes()


        results = []
        for attr in ATTRIBUTES:
            if attr in G.nodes[list(G.nodes)[0]]:  # Vérifie si l'attribut existe
                r = nx.attribute_assortativity_coefficient(G, attr)
            else:
                r = np.nan
            results.append({"Network": name, "Attribute": attr, "Assortativity": r, "Size": n})

        del G  # Libère explicitement la mémoire
        return results

    except Exception as e:
        print(f"Erreur avec {file_path}: {str(e)}")
        return []


def process_all_graphs():
    """Traite tous les graphes et sauvegarde les résultats"""
    all_results = []
    file_paths = glob.glob(os.path.join(DATA_DIR, "*.gml"))

    for file_path in tqdm(file_paths, desc="Traitement des graphes"):
        all_results.extend(process_single_graph(file_path))

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    return df


# Charger les résultats (ou les calculer si nécessaire)
if os.path.exists(RESULTS_FILE):
    df = pd.read_csv(RESULTS_FILE)
else:
    df = process_all_graphs()

# Visualisations (identique à votre version originale)
# (A) Scatter plot
fig, ax = plt.subplots(figsize=(12, 6))
for attr in ATTRIBUTES:
    subset = df[df["Attribute"] == attr]
    ax.scatter(subset["Size"], subset["Assortativity"], label=attr, alpha=0.6)

ax.axhline(0, color="black", linestyle="--", label="No assortativity")
ax.set_xscale("log")
ax.set_xlabel("Network Size (log scale)")
ax.set_ylabel("Assortativity Coefficient")
ax.set_title("Assortativity vs. Network Size (FB100 Networks)")
ax.legend()
plt.show()

# (B) Density plot
fig, ax = plt.subplots(figsize=(12, 6))
for attr in ATTRIBUTES:
    subset = df[df["Attribute"] == attr]
    sns.kdeplot(subset["Assortativity"], label=attr, fill=True, alpha=0.3)

ax.axvline(0, color="black", linestyle="--", label="No assortativity")
ax.set_xlabel("Assortativity Coefficient")
ax.set_ylabel("Density")
ax.set_title("Distribution of Assortativity (FB100 Networks)")
ax.legend()
plt.show()

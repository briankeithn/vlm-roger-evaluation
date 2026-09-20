"""Narrative Maps linear-program extraction, adapted for image collections.

Derived from `narrative_maps.py` in faustogerman/ROGER-Concept-Narratives
(MIT License, Copyright (c) 2025 Fausto German), the code accompanying:

    German, F.; Keith, B.; Matus, M.; Urrutia, D.; Meneses, C.
    "Semi-Supervised Image-Based Narrative Extraction: A Case Study with
    Historical Photographic Records." arXiv:2501.09884, 2025.

which in turn implements the Narrative Maps algorithm of:

    Keith Norambuena, B.F.; Mitra, T. "Narrative Maps: An Algorithmic Approach
    to Represent and Extract Information Narratives."
    Proc. ACM Hum.-Comput. Interact. 2021, 4, 228.

Changes made for this work:
  - `window_i_j` / `window_j_i` are module-level globals instead of being
    threaded through `create_LP` as parameters.
  - Added `solve_LP`, which builds the angular-similarity and cluster-similarity
    tables from the CLIP embeddings in `query["embed"]` and drives `create_LP`.
  - Removed the unused `start_time` parameter and `verbose` flag.

See NOTICE for the full attribution.
"""

import math
from math import ceil, exp, pi, sqrt
from urllib.parse import urlparse

import networkx as nx
import numpy as np
import pandas as pd
from pulp import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

window_i_j = {}
window_j_i = {}

# Computing Similarity Tables
def compute_temp_distance_table(query):
    dates = query['date'].to_numpy()

    pairwise_diff = dates[:, None] - dates
    pairwise_diff_days = pairwise_diff.astype('timedelta64[D]').astype(int)
    pairwise_diff_days = np.maximum(pairwise_diff_days, pairwise_diff_days.T)
    temporal_distance_table = pairwise_diff_days

    return temporal_distance_table

# Linear Program Construction
def create_LP(query, sim_table, membership_vectors, clust_sim_table, exp_temp_table, ent_table, numclust, relevance_table,
              K, mincover, sigma_t, credibility=[], bias=[], operations=[],
              has_start=True, has_end=False, window_time=None, cluster_list=[], start_nodes=[], end_nodes=[], force_cluster=True, previous_varsdict=None):
    n = len(query.index)
    
    # Variable names and indices
    var_i = []
    var_ij = []
    var_k = [str(k) for k in range(0, numclust)]

    for i in range(0, n):
        var_i.append(str(i))
        for j in window_i_j[i]:
            if i == j:
                print("ERROR IN WINDOW - BASE")
            var_ij.append(str(i) + "_" + str(j))

    # Linear program variable declaration
    minedge = LpVariable("minedge", lowBound=0, upBound=1)
    node_act_vars = LpVariable.dicts("node_act", var_i, lowBound=0, upBound=1)
    node_next_vars = LpVariable.dicts("node_next", var_ij, lowBound=0, upBound=1)

    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("StoryChainProblem", LpMaximize)
    # The objective function is added to 'prob' first
    prob += minedge, "WeakestLink"

    # Chain restrictions
    if has_start:
        num_starts = len(start_nodes)
        if num_starts == 0:
            prob += node_act_vars[str(0)] == 1, 'InitialNode'
        else:
            initial_energy = 1.0 / num_starts
            earliest_start = min(start_nodes)
            for node in start_nodes:
                prob += node_act_vars[str(node)] == initial_energy, 'InitialNode' + str(node)
            for node in range(0, earliest_start):
                prob += node_act_vars[str(node)] == 0, 'BeforeStart' + str(node)
                
    if has_end:
        num_ends = len(end_nodes)
        if num_ends == 0:
            prob += node_act_vars[str(n - 1)] == 1, 'FinalNode'
        else:
            final_energy = 1.0 / num_ends
            latest_end = min(end_nodes)
            for node in end_nodes:
                prob += node_act_vars[str(node)] == final_energy, 'FinalNode' + str(node)
            for node in range(latest_end + 1, n):
                prob += node_act_vars[str(node)] == 0, 'AfterEnd' + str(node)

    prob += lpSum([node_act_vars[i] for i in var_i]) == K, 'KNodes'

    if has_start:
        for j in range(1, n):
            if j not in start_nodes:
                prob += lpSum([node_next_vars[str(i) + "_" + str(j)]
                              for i in window_j_i[j]]) == node_act_vars[str(j)], 'InEdgeReq' + str(j)
            else:
                prob += lpSum([node_next_vars[str(i) + "_" + str(j)]
                              for i in window_j_i[j]]) == 0, 'InEdgeReq' + str(j)
    else:
        for j in range(1, n):
            prob += lpSum([node_next_vars[str(i) + "_" + str(j)]
                          for i in window_j_i[j]]) <= node_act_vars[str(j)], 'InEdgeReq' + str(j)

    if has_end:
        for i in range(0, n - 1):
            if i not in end_nodes:
                prob += lpSum([node_next_vars[str(i) + "_" + str(j)]
                              for j in window_i_j[i]]) == node_act_vars[str(i)], 'OutEdgeReq' + str(i)
            else:
                prob += lpSum([node_next_vars[str(i) + "_" + str(j)]
                              for j in window_i_j[i]]) == 0, 'OutEdgeReq' + str(i)
    else:
        for i in range(0, n - 1):
            prob += lpSum([node_next_vars[str(i) + "_" + str(j)]
                          for j in window_i_j[i]]) <= node_act_vars[str(i)], 'OutEdgeReq' + str(i)

    # Objective
    for i in range(0, n):
        for j in window_i_j[i]:
            coherence_weights = [0.5, 0.5]
            entity_multiplier = min(1 + ent_table[i, j], 2)
            relevance_multiplier = (relevance_table[i] * relevance_table[j]) ** 0.5
            coherence = (sim_table[i, j] ** coherence_weights[0]) * \
                (clust_sim_table[i, j] ** coherence_weights[1])
            weighted_coherence = min(coherence * entity_multiplier * relevance_multiplier, 1.0)
            prob += minedge <= 1 - node_next_vars[str(i) + "_" + str(j)] + \
                weighted_coherence, "Objective" + str(i) + "_" + str(j)

    if previous_varsdict:
        current_names = [v.name for v in prob.variables() if "node_act" in v.name]
        for k, v in previous_varsdict.items():
            if "node_act" in k and k in current_names:
                node_act_vars[k.replace("node_act_", "")].setInitialValue(v)

    return prob

# Useful function to build the graph from the LP output
def extract_varsdict(prob):
    # We get all the node_next variables in a dict.
    varsdict = {}
    for v in prob.variables():
        if "node_next" in v.name or "node_act" in v.name:
            varsdict[v.name] = np.clip(v.varValue, 0, 1)  # Just to avoid negative rounding errors.
    return varsdict

def build_graph_df_multiple_starts(query, varsdict, prune=None, threshold=0.01, cluster_dict={}, start_nodes=[]):
    n = len(query)
    # This has some leftover stuff that is not really useful now.
    if 'bias' in query.columns:
        graph_df = pd.DataFrame(columns=['id', 'adj_list', 'adj_weights',
                                'date', 'publication', 'title', 'text', 'url', 'bias', 'coherence'])
    else:
        graph_df = pd.DataFrame(columns=['id', 'adj_list', 'adj_weights',
                                'date', 'publication', 'title', 'text', 'url', 'coherence'])

    already_in = []
    for i in range(0, n):
        prob = []
        coherence = varsdict["node_act_" + str(i)]
        if coherence <= threshold:
            continue
        coherence_list = []
        index_list = []
        for j in window_i_j[i]:
            name = "node_next_" + str(i) + "_" + str(j)
            prob.append(varsdict[name])
            coherence_list.append(varsdict["node_act_" + str(j)])
        idx_list = [window_i_j[i][idx] for idx, e in enumerate(prob) if round(
            e, 8) != 0 and e > threshold and coherence_list[idx] > threshold]  # idx + i + 1
        nz_prob = [e for idx, e in enumerate(prob) if round(
            e, 8) != 0 and e > threshold and coherence_list[idx] > threshold]
        if prune:
            if len(idx_list) > prune:
                top_prob_idx = sorted(range(len(nz_prob)), key=lambda k: nz_prob[k])[-prune:]
                idx_list = [idx_list[j] for j in top_prob_idx]
                nz_prob = [nz_prob[idx] for idx in top_prob_idx]
        sum_nz = sum(nz_prob)
        nz_prob = [nz_prob[j] / sum_nz for j in range(0, len(nz_prob))]
        # If we haven't checked this one before we add it to the graph.
        url = str(query.iloc[i]['url'])
        if i in already_in or sum_nz > 0:
            if len(url) > 0:
                url = urlparse(url).netloc
            if not (graph_df['id'] == i).any():
                title = query.iloc[i]['title']
                for key, value in cluster_dict.items():
                    if str(i) in value:
                        title = "[" + str(key) + "] " + title
                outgoing_edges = [idx_temp for idx_temp in idx_list]
                # coherence = varsdict["node_act_" + str(i)]
                if 'bias' in query.columns:
                    graph_df.loc[len(graph_df)] = [i, outgoing_edges, nz_prob, query.iloc[i]['date'], query.iloc[i]['publication'],
                                                   title, '', query.iloc[i]['url'], query.iloc[i]['bias'], coherence]
                else:
                    graph_df.loc[len(graph_df)] = [i, outgoing_edges, nz_prob, query.iloc[i]['date'], query.iloc[i]['publication'],
                                                   title, '', query.iloc[i]['url'], coherence]

            already_in += [i] + idx_list
    return graph_df


# Building NetworkX graph.
def build_graph(graph_df):
    G = nx.DiGraph()
    for index, row in graph_df.iterrows():
        G.add_node(str(row['id']), coherence=max(-math.log(row['coherence']), 0))
        for idx, adj in enumerate(row['adj_list']):
            G.add_edge(str(row['id']), str(adj), weight=max(-math.log(row['adj_weights'][idx]), 0))
    return G


# Recursively Extract Storylines
def get_shortest_path(G):
    sources = [node for node, in_degree in G.in_degree() if in_degree == 0]
    targets = [node for node, out_degree in G.out_degree() if out_degree == 0]
    best_st = (sources[0], targets[0])
    try:
        best_val = nx.shortest_path_length(
            G, best_st[0], best_st[1], weight='weight') + G.nodes[best_st[0]]['coherence']  # Check? + vs *
    except nx.NetworkXNoPath:
        best_val = 100000
    for s in sources:
        for t in targets:
            try:
                current_val = nx.shortest_path_length(G, s, t, weight='weight') + G.nodes[s]['coherence']
            except nx.NetworkXNoPath:
                current_val = 100000
            if current_val < best_val:
                best_st = (s, t)
                best_val = current_val
    sp = nx.shortest_path(G, best_st[0], best_st[1], weight='weight')
    return sp


def normalize_graph(G):
    for node in G.nodes():
        llhs = [edge[2]['weight'] for edge in G.out_edges(node, data=True)]
        probabilities = [exp(-llh) for llh in llhs]
        sum_prob = sum(probabilities)
        probs = [prob / sum_prob for prob in probabilities]
        for idx, edge in enumerate(G.out_edges(node)):
            attrs = {edge: {'weigth': llhs[idx], 'prob': probs[idx]}}
            nx.set_edge_attributes(G, attrs)
    return G


def graph_stories(G, start_nodes=[], end_nodes=[]):
    # Base case, return the nodes if there is 1 or fewer nodes left.
    if len(G.nodes()) == 0:
        return []
    if len(G.nodes()) == 1:
        return [list(G.nodes())]
    # Base case, return node singletons if there are no edges left.
    if len(G.edges()) == 0:
        return [[node] for node in G.nodes()]
    # Main case.
    # Get maximum likelihood chain.
    if len(start_nodes) > 0 and len(end_nodes) > 0:  # First, special case when there is a start and end node.
        if nx.has_path(G, str(start_nodes[0]), str(end_nodes[0])):
            mlc = nx.shortest_path(G, str(start_nodes[0]), str(end_nodes[0]), weight='weight')
        else:
            mlc = get_shortest_path(G)  # Case where no paths exist.
    else:
        mlc = get_shortest_path(G)
    # Remove all nodes and adjacent edges to the maximum likelihood chain.
    H = G.copy()
    for node in mlc:
        H.remove_node(node)

    # Normalize outgoing edges to sum up to 1.
    H = normalize_graph(H)

    # Recursive call (special case if multiple connected components)
    # if nx.is_connected(H):
    # print("Connected. No issues.")
    return [mlc] + graph_stories(H)


# Extract Representative Landmarks
def get_representative_landmarks(G, storylines, query, mode="ranked"):
    antichain = []
    # first is default.
    if mode == "last":
        antichain = [story[-1] for story in storylines]  # get the last element in the story
    elif mode == "degree":
        degree_story = [[v for k, v in G.degree(story)] for story in storylines]
        max_degree_idx_list = [degrees.index(max(degrees)) for degrees in degree_story]
        antichain = [storylines[idx][max_degree_idx]
                     for idx, max_degree_idx in enumerate(max_degree_idx_list)]
    elif mode == "centrality":
        explanation = []
        antichain = []
        # Count non-singleton storylines.
        num_landmarks = len([story for story in storylines if len(story) > 1])
        # g_distance_dict = {(e1, e2): 1 / weight for e1, e2, weight in G.edges(data='weight')}
        # nx.set_edge_attributes(g, g_distance_dict, 'distance')
        # centrality = nx.closeness_centrality(G, distance='weight')
        centrality = nx.degree_centrality(G)
        centrality_df = pd.DataFrame.from_dict(
            {'node': list(centrality.keys()), 'centrality': list(centrality.values())})
        centrality_df = centrality_df.sort_values('centrality', ascending=False)
        for idx in range(num_landmarks):  # Get the top N landmarks based on centrality, where N = num_landmarks
            antichain.append(centrality_df.iloc[idx]['node'])
            explanation.append(
                "This event was marked as important due to its position as a relevant -hub- in the map.")
    elif mode == "centroid":
        explanation = []
        antichain = []
        for story in storylines:
            if len(story) > 1:  # We exclude singleton storyline (no relevant landmarks)
                node_embedding_list = [query.loc[query['id'] == node]['embed'].item() for node in story]
                node_embeddings = np.stack(node_embedding_list)
                centroid = node_embeddings.mean(axis=0)
                l1_distance = np.linalg.norm(node_embeddings - centroid, axis=1)
                idx_closest_node = np.argmin(l1_distance)
                antichain.append(story[idx_closest_node])  # need the ID, not the index
                explanation.append(
                    "This event was marked as important due to its -content- being representative of its corresponding storyline.")
    elif mode == "ranked":
        centrality = nx.betweenness_centrality(G, weight='weight')
        explanation = []
        for story in storylines:
            if len(story) > 1:  # We exclude singleton storyline (no relevant landmarks)
                node_embedding_list = [query.loc[query['id'] == node]['embed'].item() for node in story]
                node_embeddings = np.stack(node_embedding_list)
                centroid = node_embeddings.mean(axis=0)
                l1_distance = np.linalg.norm(node_embeddings - centroid, axis=1)
                temp = l1_distance.argsort()
                ranks_dist = np.empty_like(temp)
                ranks_dist[temp] = np.arange(len(l1_distance))
                # idx_closest_node = np.argmin(l1_distance)
                # antichain.append(story[idx_closest_node])
                centrality_array = np.array([centrality[node] for node in story])
                temp = centrality_array.argsort()
                ranks_centrality = np.empty_like(centrality_array)
                ranks_centrality[temp] = np.arange(len(centrality_array))

                average_ranks = np.average([ranks_dist, ranks_centrality], axis=0)
                idx_min = np.where(average_ranks == average_ranks.min())[0]
                lowest_dist_rank_idx = idx_min[0]
                lowest_dist_rank = ranks_dist[lowest_dist_rank_idx]
                # This doesn't matter if there's only one min node.
                # If there are many we give priority to dist to break ties.
                for idx in idx_min:
                    if lowest_dist_rank > ranks_dist[idx]:
                        lowest_dist_rank_idx = idx
                        lowest_dist_rank = ranks_dist[idx]
                antichain.append(story[lowest_dist_rank_idx])
                if ranks_dist[lowest_dist_rank_idx] < ranks_centrality[lowest_dist_rank_idx]:
                    explanation.append(
                        "This event was marked as important due to its -content- being representative of its corresponding storyline.")
                elif ranks_dist[lowest_dist_rank_idx] == ranks_centrality[lowest_dist_rank_idx]:
                    explanation.append(
                        "This event was marked as important based on its -content- being representative of its corresponding storyline and acting as a relevant -hub- in the map.")
                else:
                    explanation.append(
                        "This event was marked as important due to its position as a relevant -hub- in the map.")
    else:  # first
        antichain = [story[0] for story in storylines]  # get the first element in the story
    return antichain

def solve_LP(
    query,
    dataset,
    membership_vectors,
    K=6,
    mincover=0.20,
    sigma_t=30,
    start_nodes=[],
    end_nodes=[],
    force_cluster=True,
    use_entities=True,
    use_temporal=True,
    strict_start=False,
):

    n = len(query.index)
    
    # Compute temporal distance table
    temporal_distance_table = compute_temp_distance_table(query)
    
    if sigma_t != 0 and use_temporal:
        exp_temp_table = np.exp(-temporal_distance_table / sigma_t)
    else:
        exp_temp_table = np.ones(temporal_distance_table.shape)

    window_time = None
    if sigma_t != 0 and use_temporal:
        window_time = sigma_t * 3  # Days

    if window_time is None:
        for i in range(0, n):
            window_i_j[i] = list(range(i + 1, n))
        for j in range(0, n):
            window_j_i[j] = list(range(0, j))
    else:
        for j in range(0, n):
            window_j_i[j] = []
        for i in range(0, n):
            window_i_j[i] = []
        for i in range(0, n - 1):
            window = 0
            for j in range(i + 1, n):
                if temporal_distance_table[i, j] <= window_time:
                    window += 1
            window = max(min(5, n - i), window)
            window_i_j[i] = list(range(i + 1, min(i + window, n)))
            for j in window_i_j[i]:
                window_j_i[j].append(i)

    # Compute similarity tables from the embeddings
    similarities = np.clip(cosine_similarity(np.array(query["embed"].tolist())), -1, 1)
    sim_table = (1 - np.arccos(similarities) / np.pi)
    mask = np.ones(sim_table.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    max_value = sim_table[mask].max()
    min_value = sim_table[mask].min()
    sim_table = (sim_table - min_value) / (max_value - min_value)
    sim_table = np.clip(sim_table, 0, 1)
    
    # Compute cluster similarity
    clust_sim = np.zeros((membership_vectors.shape[0], membership_vectors.shape[0]))
    
    if len(membership_vectors.shape) > 1:
        numclust = membership_vectors.shape[1]
        membership_vectors[membership_vectors < 1 / numclust] = 0
        membership_vectors[np.all(membership_vectors == 0, axis=1)] = np.ones(numclust) / numclust
        row_sums = membership_vectors.sum(axis=1)
        membership_vectors = membership_vectors / row_sums[:, np.newaxis]
        
        from scipy.spatial.distance import jensenshannon
        clust_sim = distance.cdist(
            membership_vectors,
            membership_vectors,
            lambda u, v: jensenshannon(u, v, base=2.0)
        )
    else:
        numclust = 1
        membership_vectors = np.ones((membership_vectors.shape[0], 1))
    
    clust_sim_table = 1 - clust_sim

    ent_table = np.zeros((n, n))
    actual_ent_table = ent_table
    ent_doc_list = None

    relevance_table = [1.0] * membership_vectors.shape[0]

    has_start = False
    if start_nodes is not None:
        has_start = (len(start_nodes) > 0)
    has_end = False
    if end_nodes is not None:
        has_end = (len(end_nodes) > 0)

    previous_varsdict = None

    print("Creating LP with length " + str(K))
    prob = create_LP(
        query,
        sim_table,
        membership_vectors,
        clust_sim_table,
        exp_temp_table,
        actual_ent_table,
        numclust,
        relevance_table,
        K=K,
        mincover=mincover,
        sigma_t=sigma_t,
        has_start=has_start,
        has_end=has_end,
        start_nodes=start_nodes,
        end_nodes=end_nodes,
        force_cluster=force_cluster,
        previous_varsdict=previous_varsdict
    )
    print("Created LP with length " + str(K))
    prob.solve(PULP_CBC_CMD(mip=False, warmStart=True))

    varsdict = extract_varsdict(prob)

    graph_df = build_graph_df_multiple_starts(query, varsdict, prune=int(np.ceil(np.sqrt(K))), threshold=0.1 / K, cluster_dict={})

    if strict_start and has_start:
        # `graph_clean_up` is referenced here in the upstream Narrative Maps code
        # (faustogerman/ROGER-Concept-Narratives) but is not defined in it, so this
        # branch has never been executable. The experiments in this repository all
        # call solve_LP with strict_start=False, so the branch is never taken.
        # Fail loudly instead of raising a bare NameError if anyone enables it.
        raise NotImplementedError(
            "solve_LP(strict_start=True) is not supported: the upstream "
            "`graph_clean_up` helper it depends on was never published. "
            "All experiments in this repository use strict_start=False."
        )

    return [graph_df, (numclust, LpStatus[prob.status]), sim_table, clust_sim_table, ent_table, ent_doc_list]

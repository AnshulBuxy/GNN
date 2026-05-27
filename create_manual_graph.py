import pandas as pd
import numpy as np

def create_manual_graph():
    """
    Manually define the graph connections between your 8 stations.
    Edit the 'connections' list below to define which stations are connected.
    """
    
    # Your 8 station IDs
    station_ids = [1,2, 3, 4, 5, 6, 7, 8]
    n = len(station_ids)
    
    # Initialize adjacency matrix with zeros (no connections)
    graph_matrix = np.zeros((n, n))
    
    # =========================================================================
    # DEFINE NEIGHBORS FOR EACH STATION
    # =========================================================================
    # Edit this dictionary to define which stations connect to which
    # Format: station_id: [list of connected stations]
    
    neighbors = {
        1: [6,5],           # Station 1 connects to 2 and     
        2: [4,8,3],        # Station 2 connects to 1, 4, and 8
        3: [2,4,8],     # Station 3 connects to 1, 2, 4, and 5
        4: [2,3,8,5],        # Station 4 connects to 2, 3, and 5
        5: [1,4,6],        # Station 5 connects to 3, 4, and 6
        6: [1,7,5],           # Station 6 connects to 5 and 7
        7: [8,6],           # Station 7 connects to 6 and 8
        8: [2,3,4,7],              # Station 8 connects to 7
    }
    
    # Convert neighbors dict to connections list
    connections = []
    for station, neighbor_list in neighbors.items():
        for neighbor in neighbor_list:
            connections.append((station, neighbor))
    
    # =========================================================================
    
    # Build the adjacency matrix from connections
    print(f"Creating graph with {len(connections)} edge(s)...")
    for station_from, station_to in connections:
        # Convert station IDs to indices (assuming IDs start from 1)
        i = station_ids.index(station_from)
        j = station_ids.index(station_to)
        
        # Add bidirectional edge
        graph_matrix[i, j] = 1
        graph_matrix[j, i] = 1
    
    # Remove self-loops (diagonal)
    np.fill_diagonal(graph_matrix, 0)
    
    # Convert to DataFrame
    graph_df = pd.DataFrame(graph_matrix, index=station_ids, columns=station_ids)
    
    # Print the adjacency matrix
    print("\n" + "=" * 60)
    print("ADJACENCY MATRIX")
    print("=" * 60)
    print("1 = Connected, 0 = Not Connected\n")
    print(graph_df.astype(int))
    
    # Print statistics
    num_edges = int(graph_matrix.sum() / 2)  # Divide by 2 because symmetric
    print("\n" + "=" * 60)
    print("GRAPH STATISTICS")
    print("=" * 60)
    print(f"Number of stations: {n}")
    print(f"Number of edges: {num_edges}")
    print(f"Average connections per station: {graph_matrix.sum(axis=0).mean():.1f}")
    
    # Show which stations are connected
    print("\n" + "=" * 60)
    print("CONNECTIONS LIST")
    print("=" * 60)
    for i, station_id in enumerate(station_ids):
        connected = [station_ids[j] for j in range(n) if graph_matrix[i, j] == 1]
        if connected:
            print(f"Station {station_id} ↔ {connected}")
        else:
            print(f"Station {station_id} ↔ [No connections]")
    
    # Save to pickle
    output_file = "data/graph.pkl"
    graph_df.to_pickle(output_file)
    print("\n" + "=" * 60)
    print(f"✓ Graph saved to {output_file}")
    print("=" * 60)
    
    return graph_df


if __name__ == "__main__":
    print("=" * 60)
    print("MANUAL GRAPH CREATION")
    print("=" * 60)
    print("\nEdit the 'connections' list in this script to define")
    print("which stations are connected to each other.\n")
    
    graph_df = create_manual_graph()
    
    print("\nYou can now run: python train_and_evaluate.py")
    print("And select option [1] GNN to use this graph\n")

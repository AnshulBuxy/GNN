import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from geopy.distance import geodesic

def create_graph_from_stations(stations_file, output_file, method='knn', k=3, distance_threshold=None):
    """
    Create a graph adjacency matrix from station GPS coordinates.
    
    Parameters:
    -----------
    stations_file : str
        Path to traffic_stations.csv with GPS coordinates
    output_file : str
        Path to save the graph.pkl file
    method : str
        'knn' - Connect each station to K nearest neighbors
        'threshold' - Connect stations within distance_threshold km
        'fully_connected' - Connect all stations
    k : int
        Number of nearest neighbors (only for 'knn' method)
    distance_threshold : float
        Maximum distance in km to connect stations (only for 'threshold' method)
    """
    
    # Read stations data
    stations_df = pd.read_csv(stations_file)
    station_ids = stations_df['id'].tolist()
    
    # Get GPS coordinates
    positions = stations_df[['latitude', 'longitude']].to_numpy()
    
    # Calculate distance matrix (in km)
    print("Calculating distances between stations...")
    distance_matrix = squareform(pdist(positions, metric=lambda lat_lon1, lat_lon2: 
                                       geodesic(lat_lon1, lat_lon2).km))
    
    # Create adjacency matrix
    n = len(station_ids)
    graph_matrix = np.zeros((n, n))
    
    if method == 'knn':
        print(f"Creating K-NN graph with K={k}...")
        for i in range(n):
            # Get K nearest neighbors (excluding itself)
            knn_indices = np.argsort(distance_matrix[i])[1:k+1]
            graph_matrix[i, knn_indices] = 1
        
        # Make symmetric (if A connects to B, B connects to A)
        graph_matrix = np.maximum(graph_matrix, graph_matrix.T)
        
    elif method == 'threshold':
        if distance_threshold is None:
            raise ValueError("distance_threshold must be specified for 'threshold' method")
        print(f"Creating threshold graph with max distance={distance_threshold} km...")
        graph_matrix = (distance_matrix < distance_threshold).astype(int)
        np.fill_diagonal(graph_matrix, 0)  # No self-loops
        
    elif method == 'fully_connected':
        print("Creating fully connected graph...")
        graph_matrix = np.ones((n, n))
        np.fill_diagonal(graph_matrix, 0)  # No self-loops
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to DataFrame with station IDs as index/columns
    graph_df = pd.DataFrame(graph_matrix, index=station_ids, columns=station_ids)
    
    # Print statistics
    num_edges = int(graph_matrix.sum())
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {n}")
    print(f"  Edges: {num_edges}")
    print(f"  Average degree: {num_edges/n:.1f}")
    print(f"\nDistance matrix (km):")
    distance_df = pd.DataFrame(distance_matrix, index=station_ids, columns=station_ids)
    print(distance_df.round(2))
    
    print(f"\nAdjacency Matrix:")
    print(graph_df.astype(int))
    
    # Save to pickle
    graph_df.to_pickle(output_file)
    print(f"\n✓ Graph saved to {output_file}")
    
    return graph_df, distance_df


if __name__ == "__main__":
    stations_file = "data/traffic_stations.csv"
    output_file = "data/graph.pkl"
    
    print("=" * 60)
    print("GRAPH CREATION OPTIONS")
    print("=" * 60)
    print("\n[1] K-Nearest Neighbors (KNN)")
    print("    Connect each station to its K closest neighbors")
    print("\n[2] Distance Threshold")
    print("    Connect stations within a certain distance")
    print("\n[3] Fully Connected")
    print("    Connect all stations to each other")
    print("\n" + "=" * 60)
    
    choice = input("\nSelect method (1/2/3): ").strip()
    
    if choice == '1':
        k = input("Enter K (number of nearest neighbors) [default: 3]: ").strip()
        k = int(k) if k else 3
        graph_df, dist_df = create_graph_from_stations(
            stations_file, output_file, method='knn', k=k
        )
        
    elif choice == '2':
        threshold = input("Enter maximum distance in km [default: 10]: ").strip()
        threshold = float(threshold) if threshold else 10.0
        graph_df, dist_df = create_graph_from_stations(
            stations_file, output_file, method='threshold', distance_threshold=threshold
        )
        
    elif choice == '3':
        graph_df, dist_df = create_graph_from_stations(
            stations_file, output_file, method='fully_connected'
        )
        
    else:
        print("Invalid choice!")
        exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Graph file created successfully!")
    print("=" * 60)
    print("\nYou can now run: python train_and_evaluate.py")
    print("And select option [1] GNN to use this pre-defined graph")

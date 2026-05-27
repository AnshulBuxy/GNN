import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from os.path import isfile 
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from geopy.distance import geodesic
from torch_geometric.data import Data
import torch_geometric.loader 
import torch_geometric.transforms as GT
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from config import use_global_features

class TrafficVolumeDataSet(Dataset):
    """
        Custom PyTorch dataset for traffic data.
    """
    def __init__(self, datafile):
        assert isfile(datafile), f"Error: Data file {datafile} not found! Please run preprocess_data.py first."
        self.datafile = datafile
        self.df = pd.read_pickle(self.datafile)
        self.len = len(self.df) - 1 # Since there is no row after the last row.
        self.column_names = self.df.columns
        self.timestamps = self.df.index
        print(f"Loaded datafile {self.datafile} with {self.len} rows...")

    def __getitem__(self, index):
        # Return two consecutive rows of traffic data (and date / time)
        # Replace NaNs with -1 
        data_now = self.df.iloc[index].replace(np.nan, -1)
        data_next = self.df.iloc[index + 1].replace(np.nan, -1)

        datetime = torch.Tensor(self.convert_time(data_now))

        volumes_now = torch.Tensor(data_now.to_numpy(dtype=np.float32))
        target = torch.Tensor(data_next.to_numpy(dtype=np.float32))
        data_now = torch.cat((datetime, volumes_now))

        return (data_now, target, index) 

    def __len__(self):
        return self.len

    def convert_time(self, data):
        timestamp = getattr(data, "name")
        month = timestamp.month
        weekday = timestamp.weekday()
        hour = timestamp.hour
        return [month, weekday, hour]
    
def TrafficVolumeDataLoader(datafile, batch_size=32, num_workers=4, shuffle=False, drop_last=False):
    dataset = TrafficVolumeDataSet(datafile)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return dataloader

def create_edge_index_and_features(stations_included_file, stations_data_file, graph_file=None, compute_edge_features=True):
    """
        Create adjacency matrix and return edge index in COO format.
        stations_included_file : File containing IDs of the stations included in the pre-processed data.
        graph_file : File containing adjacency matrix (for all stations). If graph_file is None, then create kNN graph.
        stations_data_file: File containing stations IDs and GPS coordinates (lat, lon)
        compute_edge_features : If false, then set all edge features to 1.0.
    """
    stations_included = pd.read_csv(stations_included_file).iloc[:, 1]
    stations_data_df = pd.read_csv(stations_data_file) 
    positions = stations_data_df.loc[stations_data_df["id"].isin(stations_included), ["latitude", "longitude"]].to_numpy()
    distance_matrix = squareform(pdist(positions, metric=lambda lat,lon: geodesic(lat,lon).km))

    if graph_file:
        print(f"Using pre-defined graph from file ({graph_file})...")
        graph_df = pd.read_pickle(graph_file).loc[stations_included, stations_included]
    else:
        # Create kNN graph
        K = 10 
        print(f"Using KNN graph with K={K}...")
        graph_df = pd.DataFrame(0, index=stations_included, columns=stations_included)
        for i, row in enumerate(distance_matrix):
            knn = np.argsort(row)[:K]
            graph_df.iloc[i, knn] = 1

    # Convert adjacency matrix to COO format for use with PyG
    num_nodes = len(graph_df)
    start_indices = []
    end_indices = []
    edge_features = []
    print(f"Creating edge index and edge features...")
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if graph_df.iloc[i,j]:
                # Add edge (both directions)
                start_indices.extend([i,j])
                end_indices.extend([j,i])
                # Add edge weights
                if compute_edge_features:
                    edge_feature = np.exp(-distance_matrix[i,j])
                else:
                    edge_feature = 1.0
                edge_features.extend(2 * [[edge_feature]])

    edge_index = torch.tensor([start_indices, end_indices], dtype=torch.long)
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    # print(edge_index[0], edge_features[0])
    return edge_index, edge_features

class TrafficVolumeGraphDataSet(TrafficVolumeDataSet):
    """
        Modified dataset for use with PyTorch Geometric GNN.
    """
    def __init__(self, datafile, edge_index, edge_attr):
        super().__init__(datafile)
        self.edge_index, self.edge_attr = edge_index, edge_attr

    def __getitem__(self, index):
        """
        Return PyG Data object with node, edge and graph features.
        """
        data_now = self.df.iloc[index].replace(np.nan, -1)
        data_next = self.df.iloc[index + 1].replace(np.nan, -1)

        # Global features toggle (controlled by config.use_global_features)
        if use_global_features:
            datetime = torch.Tensor(self.convert_time(data_now)).reshape(1, 3)
        else:
            datetime = torch.zeros(1, 3)  # All zeros - no temporal information
        volumes = torch.Tensor(data_now.to_numpy(dtype=np.float32)).unsqueeze(1)
        y = torch.Tensor(data_next.to_numpy(dtype=np.float32))

        data = Data(x=volumes, edge_index=self.edge_index, edge_attr=self.edge_attr, y=y, u=datetime)
        # print(f"Data at index {index}: {data}")
        return data

def TrafficVolumeGraphDataLoader(datafile, edge_index, edge_attr, batch_size=32, num_workers=4, shuffle=False, drop_last=False):
    dataset = TrafficVolumeGraphDataSet(datafile, edge_index, edge_attr)
    dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=True)
    return dataloader

class TrafficVolumeSequenceGraphDataSet(TrafficVolumeDataSet):
    """
        Modified dataset for use with temporal models (GRU-GNN) that need sequences.
        Returns past `sequence_length` hours to predict the next hour.
    """
    def __init__(self, datafile, edge_index, edge_attr, sequence_length=12):
        super().__init__(datafile)
        self.edge_index, self.edge_attr = edge_index, edge_attr
        self.sequence_length = sequence_length
        
        # Build valid indices: only sequences with continuous timestamps (1-hour gaps)
        self.valid_indices = []
        for i in range(len(self.df) - sequence_length):
            # Check if sequence + target are temporally continuous (1-hour gaps)
            timestamps = self.df.index[i : i + sequence_length + 1]
            is_valid = True
            for j in range(len(timestamps) - 1):
                time_diff = (timestamps[j+1] - timestamps[j]).total_seconds() / 3600  # hours
                if abs(time_diff - 1.0) > 0.1:  # Not a 1-hour gap
                    is_valid = False
                    break
            if is_valid:
                self.valid_indices.append(i)
        
        self.len = len(self.valid_indices)
        print(f"  → Found {self.len} valid continuous sequences (out of {len(self.df) - sequence_length} total)")

    def __getitem__(self, index):
        """
        Return PyG Data object with sequence of past hours as node features.
        x will have shape (num_nodes, sequence_length) - each node has past sequence_length hours
        """
        # Map to valid index
        actual_index = self.valid_indices[index]
        
        # Get sequence of past hours: [index, index+1, ..., index+sequence_length-1]
        sequence_data = self.df.iloc[actual_index : actual_index + self.sequence_length].replace(np.nan, -1)
        # Get target (next hour after sequence)
        data_next = self.df.iloc[actual_index + self.sequence_length].replace(np.nan, -1)
        
        # Global features toggle (controlled by config.use_global_features)
        if use_global_features:
            datetime = torch.Tensor(self.convert_time(sequence_data.iloc[-1])).reshape(1, 3)
        else:
            datetime = torch.zeros(1, 3)  # All zeros - no temporal information
        
        # Convert sequence to tensor: (sequence_length, num_nodes) -> transpose to (num_nodes, sequence_length)
        volumes_sequence = torch.Tensor(sequence_data.to_numpy(dtype=np.float32).T)  # Shape: (num_nodes, seq_len)
        y = torch.Tensor(data_next.to_numpy(dtype=np.float32))

        data = Data(x=volumes_sequence, edge_index=self.edge_index, edge_attr=self.edge_attr, y=y, u=datetime)
        return data

def TrafficVolumeSequenceGraphDataLoader(datafile, edge_index, edge_attr, sequence_length=12, batch_size=32, num_workers=4, shuffle=False, drop_last=False):
    dataset = TrafficVolumeSequenceGraphDataSet(datafile, edge_index, edge_attr, sequence_length)
    dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=True)
    return dataloader


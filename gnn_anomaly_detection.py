import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')

# ========= STRIPE LITE IMPLEMENTATION =========
class ActiveSet:
    """
    Maintains the active set of edges for online inference
    """
    def __init__(self):
        self.edges = set()  # Set of (u, v, r) tuples
        
    def apply(self, u, v, r, is_add):
        """Apply an event to the active set"""
        edge = (u, v, r)
        if is_add:
            self.edges.add(edge)
        else:
            self.edges.discard(edge)
    
    def get_active_edges(self):
        """Get all active edges as a list"""
        return list(self.edges)

class StripeLiteRGCN(torch.nn.Module):
    """
    STRIPE Lite RGCN model compatible with the downloaded weights
    """
    def __init__(self, num_nodes, num_rels, d_node=32, d_rel=16, d_hid=64):
        super(StripeLiteRGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.d_node = d_node
        self.d_rel = d_rel
        self.d_hid = d_hid

        # Node embeddings: [num_nodes, d_node]
        self.node_emb = torch.nn.Embedding(num_nodes, d_node)
        
        # Relation embeddings: [num_rels, d_rel]
        self.rel_emb = torch.nn.Embedding(num_rels, d_rel)
        
        # RGCN layer with exact structure from your model
        # rgcn.weight: [10, d_node, d_hid], rgcn.root: [d_node, d_hid], rgcn.bias: [d_hid]
        self.rgcn = torch.nn.Module()
        self.rgcn.weight = torch.nn.Parameter(torch.randn(10, d_node, d_hid))
        self.rgcn.root = torch.nn.Parameter(torch.randn(d_node, d_hid))
        self.rgcn.bias = torch.nn.Parameter(torch.randn(d_hid))
        
        # Decoders with exact structure from your model
        # dec_add.0.weight: [d_hid, 208], dec_add.0.bias: [d_hid]
        # dec_add.2.weight: [1, d_hid], dec_add.2.bias: [1]
        self.dec_add = torch.nn.Module()
        self.dec_add.add_module('0', torch.nn.Linear(208, d_hid))
        self.dec_add.add_module('2', torch.nn.Linear(d_hid, 1))
        
        self.dec_rem = torch.nn.Module()
        self.dec_rem.add_module('0', torch.nn.Linear(208, d_hid))
        self.dec_rem.add_module('2', torch.nn.Linear(d_hid, 1))

    def forward(self, edge_index, edge_type, timestamps):
        """Forward pass for STRIPE Lite model"""
        # Get node embeddings
        src_emb = self.node_emb(edge_index[0])  # [batch_size, d_node]
        dst_emb = self.node_emb(edge_index[1])  # [batch_size, d_node]
        
        # Get relation embeddings
        rel_emb = self.rel_emb(edge_type)  # [batch_size, d_rel]
        
        # Combine embeddings - need to match the expected input size of 208
        combined_emb = torch.cat([src_emb, dst_emb, rel_emb], dim=1)  # [batch_size, d_node + d_node + d_rel]
        
        # Pad or transform to get the expected 208 dimensions
        if combined_emb.size(1) < 208:
            # Pad with zeros
            padding = torch.zeros(combined_emb.size(0), 208 - combined_emb.size(1), device=combined_emb.device)
            combined_emb = torch.cat([combined_emb, padding], dim=1)
        elif combined_emb.size(1) > 208:
            # Truncate
            combined_emb = combined_emb[:, :208]
        
        return combined_emb

    def detect_anomalies(self, edge_index, edge_attr, timestamps):
        """Detect anomalies using the STRIPE Lite model"""
        with torch.no_grad():
            # Extract edge type from edge_attr (assuming one-hot encoding)
            edge_type = torch.argmax(edge_attr, dim=1)
            
            # Forward pass to get combined embeddings
            combined_emb = self(edge_index, edge_type, timestamps)
            
            # Use decoder to predict anomaly scores
            # dec_add: 208 -> d_hid -> 1
            x = torch.relu(self.dec_add._modules['0'](combined_emb))
            anomaly_scores = self.dec_add._modules['2'](x).squeeze()
            
        return anomaly_scores

def score_event(active_set, model, num_nodes, num_rels, u, v, r, is_add):
    """
    Score a single event using the STRIPE Lite model
    Returns plausibility score in [0,1]
    """
    with torch.no_grad():
        # Create edge index for the event
        edge_index = torch.tensor([[u], [v]], dtype=torch.long, device=model.device if hasattr(model, 'device') else 'cpu')
        
        # Create edge type tensor
        edge_type = torch.tensor([r], dtype=torch.long, device=model.device if hasattr(model, 'device') else 'cpu')
        
        # Create timestamps (dummy for now)
        timestamps = torch.tensor([0], dtype=torch.long, device=model.device if hasattr(model, 'device') else 'cpu')
        
        # Get embeddings
        combined_emb = model(edge_index, edge_type, timestamps)
        
        # Use appropriate decoder based on event type
        if is_add:
            decoder = model.dec_add
        else:
            decoder = model.dec_rem
            
        # Predict plausibility score
        x = torch.relu(decoder._modules['0'](combined_emb))
        plaus_score = torch.sigmoid(decoder._modules['2'](x)).item()
        
    return plaus_score

class STRIPELiteModel(torch.nn.Module):
    """
    STRIPE Lite model compatible with the downloaded weights
    Based on the actual state_dict structure from your model
    """

    def __init__(self, num_nodes, num_rels, d_node, d_rel, d_hid):
        super(STRIPELiteModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.d_node = d_node
        self.d_rel = d_rel
        self.d_hid = d_hid

        # Node embeddings: [4522, 32]
        self.node_emb = torch.nn.Embedding(num_nodes, d_node)
        
        # Relation embeddings: [5, 16]
        self.rel_emb = torch.nn.Embedding(num_rels, d_rel)
        
        # RGCN layer with exact structure from your model
        # rgcn.weight: [10, 32, 64], rgcn.root: [32, 64], rgcn.bias: [64]
        self.rgcn = torch.nn.Module()
        self.rgcn.weight = torch.nn.Parameter(torch.randn(10, d_node, d_hid))
        self.rgcn.root = torch.nn.Parameter(torch.randn(d_node, d_hid))
        self.rgcn.bias = torch.nn.Parameter(torch.randn(d_hid))
        
        # Decoders with exact structure from your model
        # dec_add.0.weight: [64, 208], dec_add.0.bias: [64]
        # dec_add.2.weight: [1, 64], dec_add.2.bias: [1]
        self.dec_add = torch.nn.Module()
        self.dec_add.add_module('0', torch.nn.Linear(208, 64))
        self.dec_add.add_module('2', torch.nn.Linear(64, 1))
        
        self.dec_rem = torch.nn.Module()
        self.dec_rem.add_module('0', torch.nn.Linear(208, 64))
        self.dec_rem.add_module('2', torch.nn.Linear(64, 1))

    def forward(self, edge_index, edge_type, timestamps):
        """Forward pass for STRIPE Lite model"""
        # Get node embeddings
        src_emb = self.node_emb(edge_index[0])  # [batch_size, d_node]
        dst_emb = self.node_emb(edge_index[1])  # [batch_size, d_node]
        
        # Get relation embeddings
        rel_emb = self.rel_emb(edge_type)  # [batch_size, d_rel]
        
        # Combine embeddings - need to match the expected input size of 208
        # Based on dec_add.0.weight: [64, 208], we need input of size 208
        # This suggests we need to concatenate and possibly pad/transform
        combined_emb = torch.cat([src_emb, dst_emb, rel_emb], dim=1)  # [batch_size, d_node + d_node + d_rel]
        
        # Pad or transform to get the expected 208 dimensions
        if combined_emb.size(1) < 208:
            # Pad with zeros
            padding = torch.zeros(combined_emb.size(0), 208 - combined_emb.size(1), device=combined_emb.device)
            combined_emb = torch.cat([combined_emb, padding], dim=1)
        elif combined_emb.size(1) > 208:
            # Truncate
            combined_emb = combined_emb[:, :208]
        
        return combined_emb

    def detect_anomalies(self, edge_index, edge_attr, timestamps):
        """Detect anomalies using the STRIPE Lite model"""
        with torch.no_grad():
            # Extract edge type from edge_attr (assuming one-hot encoding)
            edge_type = torch.argmax(edge_attr, dim=1)
            
            # Forward pass to get combined embeddings
            combined_emb = self(edge_index, edge_type, timestamps)
            
            # Use decoder to predict anomaly scores
            # dec_add: 208 -> 64 -> 1
            x = torch.relu(self.dec_add._modules['0'](combined_emb))
            anomaly_scores = self.dec_add._modules['2'](x).squeeze()
            
        return anomaly_scores

class TemporalGNNAnomalyDetector(torch.nn.Module):
    """
    Simplified Temporal Graph Neural Network for anomaly detection
    Uses basic node embeddings with temporal features
    """

    def __init__(self, num_nodes, edge_dim, msg_dim, hidden_dim=64):
        super(TemporalGNNAnomalyDetector, self).__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim

        # Graph convolution layers (no edge attributes for simplicity)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Temporal encoding layer
        self.temporal_encoder = torch.nn.Linear(1, hidden_dim)

        # Edge type embedding (for different relationship types)
        self.edge_embedding = torch.nn.Embedding(msg_dim, hidden_dim)

        # Anomaly detection head
        self.anomaly_head = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

        # Node embeddings
        self.node_embeddings = torch.nn.Embedding(num_nodes, hidden_dim)
        torch.nn.init.xavier_uniform_(self.node_embeddings.weight)

    def forward(self, edge_index, edge_attr, timestamps):
        # Get node embeddings
        x = self.node_embeddings.weight

        # Create temporal encoding for all nodes
        avg_timestamp = timestamps.float().mean().unsqueeze(0).unsqueeze(-1)
        t_encoded = self.temporal_encoder(avg_timestamp)  # Shape: [1, hidden_dim]
        t_encoded = t_encoded.repeat(x.size(0), 1)  # Repeat for all nodes

        # Add temporal encoding to node embeddings
        x = x + t_encoded

        # Apply graph convolutions (without edge attributes for now)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x

    def detect_anomalies(self, edge_index, edge_attr, timestamps):
        """Detect anomalies for given edges"""
        with torch.no_grad():
            # Get node embeddings with temporal encoding
            node_embeddings = self(edge_index, edge_attr, timestamps)

            # Calculate anomaly scores for each edge
            src_emb = node_embeddings[edge_index[0]]  # Source node embeddings
            dst_emb = node_embeddings[edge_index[1]]  # Destination node embeddings

            # Concatenate source and destination embeddings
            edge_embeddings = torch.cat([src_emb, dst_emb], dim=1)

            # Predict anomaly scores
            anomaly_scores = self.anomaly_head(edge_embeddings).squeeze()

        return anomaly_scores

class ModelFactory:
    """Factory class for loading pre-trained models"""

    @staticmethod
    def load_model(model_name, model_path=None, device='cpu'):
        """
        Load a pre-trained model

        Args:
            model_name: Name of the model ('STRIPE_Lite', 'TGN', etc.)
            model_path: Path to the .pth file (optional)
            device: Device to load the model on
        """
        if model_path is None:
            # Default model paths
            model_paths = {
                'STRIPE_Lite': '/Users/adam/Documents/swisscom-hackathon/models/stripe_lite_model.pth',
                'TGN': 'models/tgn_model.pth'
            }
            model_path = model_paths.get(model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load the model from the .pth file
        try:
            model_data = torch.load(model_path, map_location=device, weights_only=False)

            # Check if it's a complete model or a state_dict
            if hasattr(model_data, 'forward'):
                # It's a complete model
                model = model_data
                model.to(device)
                model.eval()
                print(f"✓ Loaded complete model {model_name} from {model_path}")
                return model
            elif isinstance(model_data, dict) and 'state_dict' in model_data:
                # It's a checkpoint with state_dict
                state_dict = model_data['state_dict']
                
                # Extract model parameters from checkpoint
                num_nodes = model_data.get('num_nodes', 4522)  # From your model
                num_rels = model_data.get('num_rels', 5)      # From your model (HAS_PORT, DEPENDS_ON, INSTALLED_AT, HAS, REFERS_TO)
                d_node = model_data.get('d_node', 32)         # From your model
                d_rel = model_data.get('d_rel', 16)          # From your model
                d_hid = model_data.get('d_hid', 64)          # From your model

                # Create appropriate model based on model_name and state_dict keys
                if model_name == 'STRIPE_Lite' or 'node_emb.weight' in state_dict:
                    # STRIPE Lite model using the correct StripeLiteRGCN class
                    model = StripeLiteRGCN(
                        num_nodes=num_nodes,
                        num_rels=num_rels,
                        d_node=d_node,
                        d_rel=d_rel,
                        d_hid=d_hid
                    )
                elif model_name == 'TGN' or 'memory' in state_dict:
                    # TGN model - use TemporalGNNAnomalyDetector for now
                    model = TemporalGNNAnomalyDetector(
                        num_nodes=num_nodes,
                        edge_dim=64,
                        msg_dim=3,
                        hidden_dim=64
                    )
                else:
                    # Default model
                    model = TemporalGNNAnomalyDetector(
                        num_nodes=num_nodes,
                        edge_dim=64,
                        msg_dim=5,  # Updated to include HAS and REFERS_TO
                        hidden_dim=64
                    )

                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                print(f"✓ Loaded model {model_name} from checkpoint")
                return model
            else:
                # Assume it's a direct state_dict
                state_dict = model_data

                # Create appropriate model based on state_dict keys
                if 'node_emb.weight' in state_dict:
                    # STRIPE Lite model using the correct StripeLiteRGCN class
                    num_nodes = state_dict['node_emb.weight'].shape[0]  # 4522
                    num_rels = state_dict['rel_emb.weight'].shape[0]    # 5
                    d_node = state_dict['node_emb.weight'].shape[1]     # 32
                    d_rel = state_dict['rel_emb.weight'].shape[1]       # 16
                    d_hid = state_dict['rgcn.bias'].shape[0]          # 64
                    
                    model = StripeLiteRGCN(
                        num_nodes=num_nodes,
                        num_rels=num_rels,
                        d_node=d_node,
                        d_rel=d_rel,
                        d_hid=d_hid
                    )
                elif 'memory' in state_dict:
                    # TGN model
                    model = TemporalGNNAnomalyDetector(
                        num_nodes=1000,
                        edge_dim=64,
                        msg_dim=5,  # Updated to include HAS and REFERS_TO
                        hidden_dim=64
                    )
                else:
                    # Default model
                    model = TemporalGNNAnomalyDetector(
                        num_nodes=1000,
                        edge_dim=64,
                        msg_dim=5,  # Updated to include HAS and REFERS_TO
                        hidden_dim=64
                    )

                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                print(f"✓ Loaded model {model_name} from state_dict")
                return model

        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(f"Could not load model {model_name} from {model_path}")

class GraphAnomalyDetectionPipeline:
    """
    Complete pipeline for GNN-based anomaly detection
    """

    def __init__(self, memory_dim=64, time_dim=64, device='cpu', model_name='STRIPE_Lite', model_path=None):
        self.device = device
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.node2idx = {}
        self.idx2node = {}

        # Load pre-trained model
        self.load_pretrained_model()

    def load_pretrained_model(self):
        """Load the pre-trained model"""
        try:
            self.model = ModelFactory.load_model(
                self.model_name,
                self.model_path,
                self.device
            )
            print(f"✓ Loaded pre-trained model: {self.model_name}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"⚠ Warning: {e}")
            print("Using untrained model instead")
            # Fallback to untrained model
            self.model = TemporalGNNAnomalyDetector(
                num_nodes=1000,
                edge_dim=64,
                msg_dim=5,
                hidden_dim=64
            ).to(self.device)
            print(f"✓ Created untrained model: {self.model_name}")

    def prepare_data(self, data_file, sequence_length=10):
        """
        Prepare temporal data for training
        """
        # Load data
        dataset = pd.read_csv(data_file)
        print(f"Loaded {len(dataset)} edges")

        # Create node mappings
        nodes = pd.unique(dataset[["src", "dst"]].to_numpy().ravel())
        self.node2idx = {n: i for i, n in enumerate(nodes)}
        self.idx2node = {i: n for n, i in self.node2idx.items()}

        # Map to indices
        src = dataset["src"].map(self.node2idx).to_numpy()
        dst = dataset["dst"].map(self.node2idx).to_numpy()
        t = dataset["timestamp"].astype(int).to_numpy()

        # Create message embeddings (one-hot for edge labels)
        edge_labels = dataset["label"].unique()
        label2idx = {label: idx for idx, label in enumerate(edge_labels)}
        msg = np.zeros((len(dataset), len(edge_labels)))

        for i, label in enumerate(dataset["label"]):
            msg[i, label2idx[label]] = 1

        # Create temporal sequences
        sequences = []
        for i in range(len(dataset) - sequence_length):
            seq = {
                'src': torch.tensor(src[i:i+sequence_length], dtype=torch.long),
                'dst': torch.tensor(dst[i:i+sequence_length], dtype=torch.long),
                't': torch.tensor(t[i:i+sequence_length], dtype=torch.long),
                'msg': torch.tensor(msg[i:i+sequence_length], dtype=torch.float)
            }
            sequences.append(seq)

        return sequences, len(nodes), len(edge_labels)

    def train_model(self, sequences, num_nodes, msg_dim, num_epochs=50, lr=0.001):
        """
        Train the TGN model for anomaly detection
        """
        print("=== Training Temporal GNN Anomaly Detector ===")

        self.model = TemporalGNNAnomalyDetector(
            num_nodes=num_nodes,
            edge_dim=64,
            msg_dim=msg_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        # Create training data (using autoencoder-like approach)
        train_sequences, val_sequences = train_test_split(sequences, test_size=0.2, random_state=42)

        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for seq in train_sequences:
                src, dst, t, msg = seq['src'], seq['dst'], seq['t'], seq['msg']
                src, dst, t, msg = src.to(self.device), dst.to(self.device), t.to(self.device), msg.to(self.device)

                # Create edge index for the model
                edge_index = torch.stack([src, dst], dim=0)

                # Forward pass - get node embeddings
                node_embeddings = self.model(edge_index, msg, t)

                # Get embeddings for source and destination nodes
                src_emb = node_embeddings[src]
                dst_emb = node_embeddings[dst]

                # Concatenate for anomaly prediction
                edge_embeddings = torch.cat([src_emb, dst_emb], dim=1)

                # Predict anomaly score
                anomaly_score = self.model.anomaly_head(edge_embeddings).squeeze()

                # Target: assume normal edges have low anomaly score (unsupervised)
                target = torch.zeros_like(anomaly_score)

                loss = criterion(anomaly_score, target)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            if epoch % 10 == 0:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for seq in val_sequences:
                        src, dst, t, msg = seq['src'], seq['dst'], seq['t'], seq['msg']
                        src, dst, t, msg = src.to(self.device), dst.to(self.device), t.to(self.device), msg.to(self.device)

                        # Create edge index
                        edge_index = torch.stack([src, dst], dim=0)

                        # Forward pass
                        node_embeddings = self.model(edge_index, msg, t)

                        # Get embeddings for source and destination nodes
                        src_emb = node_embeddings[src]
                        dst_emb = node_embeddings[dst]
                        edge_embeddings = torch.cat([src_emb, dst_emb], dim=1)

                        # Predict anomaly score
                        anomaly_score = self.model.anomaly_head(edge_embeddings).squeeze()
                        target = torch.zeros_like(anomaly_score)
                        val_loss += criterion(anomaly_score, target).item()

                avg_val_loss = val_loss / len(val_sequences)
                print(f"Epoch {epoch}, Train Loss: {total_loss/len(train_sequences):.4f}, Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(self.model.state_dict(), 'best_gnn_model.pth')

        print("Training completed!")
        return self.model

    def detect_anomalies(self, sequences, threshold=0.5):
        """
        Detect anomalies using trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.eval()
        anomaly_scores = []

        with torch.no_grad():
            for seq in sequences:
                src, dst, t, msg = seq['src'], seq['dst'], seq['t'], seq['msg']
                src, dst, t, msg = src.to(self.device), dst.to(self.device), t.to(self.device), msg.to(self.device)

                # Create edge index
                edge_index = torch.stack([src, dst], dim=0)

                # Get node embeddings
                node_embeddings = self.model(edge_index, msg, t)

                # Get embeddings for source and destination nodes
                src_emb = node_embeddings[src]
                dst_emb = node_embeddings[dst]
                edge_embeddings = torch.cat([src_emb, dst_emb], dim=1)

                # Predict anomaly score
                anomaly_score = self.model.anomaly_head(edge_embeddings).squeeze()
                anomaly_scores.extend(anomaly_score.cpu().numpy())

        # Convert to anomaly probabilities
        scores = np.array(anomaly_scores)
        anomalies = (scores > threshold).astype(int)

        return scores, anomalies

    def evaluate_model(self, clean_sequences, anomalous_sequences):
        """
        Evaluate model performance
        """
        print("=== Evaluating Model ===")

        # Get anomaly scores for both datasets
        clean_scores, clean_anomalies = self.detect_anomalies(clean_sequences)
        anomalous_scores, anomalous_anomalies = self.detect_anomalies(anomalous_sequences)

        # Create ground truth (0 for clean, 1 for anomalous)
        y_true = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(anomalous_scores))])
        y_scores = np.concatenate([clean_scores, anomalous_scores])

        # Calculate metrics
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        print(f"AUC: {auc:.4f}")
        print(f"Average Precision: {ap:.4f}")
        print(f"Anomaly detection rate: {np.mean(anomalous_anomalies):.2%}")

        return auc, ap

# Usage example
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = GraphAnomalyDetectionPipeline(device='cpu')

    # Prepare data
    clean_sequences, num_nodes, msg_dim = pipeline.prepare_data('./data/edge_events_clean.csv')
    anomalous_sequences, _, _ = pipeline.prepare_data('./data/edge_events.csv')

    # Train model
    model = pipeline.train_model(clean_sequences, num_nodes, msg_dim, num_epochs=30)

    # Evaluate
    auc, ap = pipeline.evaluate_model(clean_sequences, anomalous_sequences)

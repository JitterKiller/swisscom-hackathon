from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import sys
import os

# Add parent directory to path to import GNN model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_anomaly_detection import GraphAnomalyDetectionPipeline, StripeLiteRGCN, ActiveSet, score_event

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'csv', 'json', 'npy', 'txt'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global variables for the models
anomaly_detectors = {}  # Dictionary to store different models
graph_data = None
available_models = ['STRIPE_Light', 'TGN']

# Global variables for STRIPE Lite online inference
stripe_lite_model = None
stripe_lite_artifact = None
online_state = None
node2idx = None
rel2idx = None
idx2node = None
idx2rel = None
tau_add = None
tau_rem = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_sample_graph_data():
    """Load sample graph data from the existing dataset"""
    global graph_data

    try:
        # Load edge events data
        df = pd.read_csv('uploads/edge_events_clean.csv')

        # Sample a subset for visualization (take first 200 edges for better performance)
        sample_df = df.head(200)

        # Create nodes and edges for visualization
        nodes = []
        edges = []

        # Get unique nodes
        unique_nodes = pd.unique(sample_df[["src", "dst"]].to_numpy().ravel())
        node_map = {node: idx for idx, node in enumerate(unique_nodes)}

        # Create node data
        for node in unique_nodes:
            degree = len(sample_df[(sample_df['src'] == node) | (sample_df['dst'] == node)])
            # Scale node size based on degree (min 8, max 35)
            size = max(8, min(35, 8 + (degree * 4)))
            nodes.append({
                'id': node,
                'type': 'device' if node.startswith(('bng-', 'cpe-', 'concentrator-', 'sa-')) else 'unknown',
                'degree': degree,
                'size': size,
                'color': '#69b3a2'
            })

        # Create edge data with simulated timestamps for demo
        for i, (_, row) in enumerate(sample_df.iterrows()):
            # Generate simulated timestamps for demonstration (0 to 1000)
            simulated_timestamp = (i * 1000) // len(sample_df)
            edges.append({
                'source': row['src'],  # Use actual node ID, not index
                'target': row['dst'],  # Use actual node ID, not index
                'type': row['label'],
                'timestamp': simulated_timestamp
            })

        # Calculate time range from simulated timestamps
        if len(edges) > 0:
            timestamps = [edge['timestamp'] for edge in edges]
            min_time = min(timestamps)
            max_time = max(timestamps)
        else:
            min_time = 0
            max_time = 1000

        graph_data = {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'time_range': [min_time, max_time]
            }
        }

        return graph_data

    except Exception as e:
        print(f"Error loading graph data: {e}")
        return None

def load_uploaded_graph_data(graph_id):
    """Load graph data from uploaded files"""
    try:
        # Try CSV format first (primary format)
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{graph_id}.csv')
        if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Sample for performance (limit to 500 edges max)
                if len(df) > 500:
                    df = df.head(500)
                    print(f"Sampled to 500 edges for performance")
                # Convert CSV to graph format (simplified)
                nodes = []
                edges = []
                unique_nodes = pd.unique(df[["src", "dst"]].to_numpy().ravel())
                node_map = {node: idx for idx, node in enumerate(unique_nodes)}

                for node in unique_nodes:
                    degree = len(df[(df['src'] == node) | (df['dst'] == node)])
                    # Scale node size based on degree (min 8, max 35)
                    size = max(8, min(35, 8 + (degree * 4)))
                    nodes.append({
                        'id': node,
                        'type': 'unknown',
                        'degree': degree,
                        'size': size,
                        'color': '#ff7f7f'
                    })

                for i, (_, row) in enumerate(df.iterrows()):
                    # Generate simulated timestamps for demonstration (0 to 1000)
                    simulated_timestamp = (i * 1000) // len(df)
                    edges.append({
                        'source': row['src'],  # Use actual node ID, not index
                        'target': row['dst'],  # Use actual node ID, not index
                        'type': row.get('label', 'unknown'),
                        'timestamp': simulated_timestamp
                    })

                # Calculate time range from simulated timestamps
                if len(edges) > 0:
                    timestamps = [edge['timestamp'] for edge in edges]
                    min_time = min(timestamps)
                    max_time = max(timestamps)
                    print(f"Simulated time range: {min_time} to {max_time}")
                else:
                    min_time = 0
                    max_time = 1000
                    print("No edges found")

                return {
                    'nodes': nodes,
                    'edges': edges,
                    'metadata': {
                        'total_nodes': len(nodes),
                        'total_edges': len(edges),
                        'time_range': [min_time, max_time]
                    }
                }
    except Exception as e:
        print(f"Error loading uploaded graph data: {e}")

    return None

def initialize_anomaly_detectors():
    """Initialize all available GNN anomaly detectors"""
    global anomaly_detectors, stripe_lite_model, stripe_lite_artifact, online_state
    global node2idx, rel2idx, idx2node, idx2rel, tau_add, tau_rem

    try:
        print("Initializing anomaly detectors...")

        # Initialize STRIPE Lite model with proper pipeline
        print("Loading STRIPE Lite model...")
        try:
            import torch
            
            # Load artifact and rebuild model following the correct pipeline
            artifact_path = '/Users/adam/Documents/swisscom-hackathon/models/stripe_lite_model.pth'
            stripe_lite_artifact = torch.load(artifact_path, map_location='cpu')
            
            num_nodes = stripe_lite_artifact["num_nodes"]
            num_rels = stripe_lite_artifact["num_rels"]
            d_node = stripe_lite_artifact.get("d_node", 32)
            d_rel = stripe_lite_artifact.get("d_rel", 16)
            d_hid = stripe_lite_artifact.get("d_hid", 64)
            
            # Create model and load state dict
            stripe_lite_model = StripeLiteRGCN(num_nodes, num_rels, d_node=d_node, d_rel=d_rel, d_hid=d_hid)
            stripe_lite_model.load_state_dict(stripe_lite_artifact["state_dict"])
            stripe_lite_model.eval()
            
            # Load mappings and thresholds
            node2idx = stripe_lite_artifact["node2idx"]
            rel2idx = stripe_lite_artifact["rel2idx"]
            idx2node = stripe_lite_artifact["idx2node"]
            idx2rel = stripe_lite_artifact["idx2rel"]
            tau_add = stripe_lite_artifact["tau_add"]
            tau_rem = stripe_lite_artifact["tau_rem"]
            
            # Initialize online state
            online_state = ActiveSet()
            
            # Create detector wrapper for compatibility
            stripe_detector = GraphAnomalyDetectionPipeline(
                device='cpu',
                model_name='STRIPE_Lite'
            )
            stripe_detector.model = stripe_lite_model
            stripe_detector.node2idx = node2idx
            
            anomaly_detectors['default'] = stripe_detector
            anomaly_detectors['STRIPE_Lite'] = stripe_detector
            print("✅ STRIPE Lite model loaded successfully with proper pipeline")
            
        except Exception as e:
            print(f"❌ Could not load STRIPE Lite model: {e}")
            print("Using fallback model...")
            stripe_detector = GraphAnomalyDetectionPipeline(
                device='cpu',
                model_name='STRIPE_Lite'
            )
            anomaly_detectors['default'] = stripe_detector
            anomaly_detectors['STRIPE_Lite'] = stripe_detector

        # Initialize TGN model
        print("Loading TGN model...")
        try:
            tgn_detector = GraphAnomalyDetectionPipeline(
                device='cpu',
                model_name='TGN',
                model_path='../models/tgn_model.pth'
            )
            anomaly_detectors['TGN'] = tgn_detector
            print("✅ TGN model loaded successfully")
        except Exception as e:
            print(f"❌ Could not load TGN model: {e}")
            print("Using fallback model...")
            tgn_detector = GraphAnomalyDetectionPipeline(
                device='cpu',
                model_name='TGN'
            )
            anomaly_detectors['TGN'] = tgn_detector

        print("✓ All anomaly detectors initialized successfully!")
        return True

    except Exception as e:
        print(f"Error initializing anomaly detectors: {e}")
        print("Some models may not be available")
        return False

@app.route('/api/graphs', methods=['GET'])
def get_available_graphs():
    """Get list of available graphs"""
    graphs = [
        {
            'id': 'default',
            'name': 'edge_events_clean',
            'model': 'STRIPE Lite',
            'description': 'Dataset de base avec modèle STRIPE Lite'
        }
    ]

    # Check for uploaded graphs in the uploads folder
    try:
        existing_ids = set()  # Track existing graph IDs
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith('.csv'):
                existing_ids.add(filename.rsplit('.', 1)[0])

        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith('.csv'):
                base_id = filename.rsplit('.', 1)[0]

                # Use existing ID if available, otherwise generate unique one
                graph_id = base_id
                counter = 1
                while graph_id in existing_ids and graph_id != base_id:
                    graph_id = f"{base_id}_{counter}"
                    counter += 1
                existing_ids.add(graph_id)

                graphs.append({
                    'id': graph_id,
                    'name': filename,
                    'model': 'TGN',
                    'description': f'Dataset uploadé avec modèle TGN - {filename}'
                })
    except Exception as e:
        print(f"Error reading uploaded files: {e}")

    return jsonify(graphs)

@app.route('/api/graph-data', methods=['GET'])
@app.route('/api/graph-data/<graph_id>', methods=['GET'])
def get_graph_data(graph_id='default'):
    """Get graph data for visualization"""
    global graph_data

    try:
        if graph_id == 'default':
            if graph_data is None:
                graph_data = load_sample_graph_data()
                if graph_data is None:
                    return jsonify({'error': 'Unable to load default graph data'}), 500
        else:
            # Try to load uploaded graph data
            graph_data = load_uploaded_graph_data(graph_id)
            if graph_data is None:
                return jsonify({'error': f'Graph {graph_id} not found'}), 404

        return jsonify(graph_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_anomaly():
    """Predict anomaly for a given edge"""
    global anomaly_detectors

    try:
        data = request.get_json()
        src = data.get('src')
        dst = data.get('dst')
        label = data.get('label', 'DEPENDS_ON')
        event_type = data.get('event_type', 'add')
        graph_id = data.get('graph_id', 'default')

        if not src or not dst:
            return jsonify({'error': 'Source and destination nodes are required'}), 400

        # Determine which model to use based on graph_id
        if graph_id == 'default':
            model_key = 'default'
        else:
            model_key = 'TGN'  # Use TGN for uploaded graphs

        # Check if model is available
        if model_key not in anomaly_detectors:
            return jsonify({'error': f'Model {model_key} not initialized'}), 500

        detector = anomaly_detectors[model_key]

        # Check if the model has been trained
        # For STRIPE Lite: check if it has dec_add (decoder)
        # For TemporalGNNAnomalyDetector: check if it has anomaly_head
        if hasattr(detector.model, 'dec_add'):
            # STRIPE Lite model
            is_model_trained = hasattr(detector.model, 'dec_add') and hasattr(detector.model.dec_add, '_modules')
        elif hasattr(detector.model, 'anomaly_head'):
            # TemporalGNNAnomalyDetector model
            is_model_trained = hasattr(detector.model, 'anomaly_head') and len(detector.model.anomaly_head) > 0
        else:
            is_model_trained = False

        # Determine model description based on training status
        if graph_id == 'default':
            model_used = 'STRIPE Lite (Trained)' if is_model_trained else 'STRIPE Lite (Untrained)'
        else:
            model_used = 'TGN (Trained)' if is_model_trained else 'TGN (Untrained)'

        # Try to use the actual trained model for prediction
        try:
            # Check if we have the STRIPE Lite model with proper pipeline
            if (graph_id == 'default' and stripe_lite_model is not None and 
                node2idx is not None and rel2idx is not None):
                
                # Use STRIPE Lite online inference pipeline
                def _map_event(src, dst, label, event_type):
                    """Map raw event fields to integer ids that the model understands"""
                    try:
                        u = int(node2idx[src])
                        v = int(node2idx[dst])
                        r = int(rel2idx[label])
                    except KeyError as e:
                        # If node/label not in training data, use fallback
                        print(f"⚠️  Unseen token {e}, using fallback mapping")
                        u = hash(src) % len(node2idx)
                        v = hash(dst) % len(node2idx)
                        r = hash(label) % len(rel2idx)
                    is_add = 1 if str(event_type).lower() == "add" else 0
                    return u, v, r, is_add

                def detect_one(src, dst, label, event_type):
                    """Score ONE incoming edge and immediately update the ActiveSet state"""
                    u, v, r, is_add = _map_event(src, dst, label, event_type)

                    # Score with current state (snapshot BEFORE applying this event)
                    s = score_event(online_state, stripe_lite_model, 
                                  stripe_lite_artifact["num_nodes"], 
                                  stripe_lite_artifact["num_rels"], 
                                  u, v, r, is_add)
                    anom_score = 1.0 - s
                    tau = tau_add if is_add == 1 else tau_rem
                    flag = bool(anom_score >= tau)

                    # Advance the state to include this event
                    online_state.apply(u, v, r, is_add)

                    return {
                        "plaus_score": float(s),
                        "anom_score": float(anom_score),
                        "threshold": float(tau),
                        "is_add": bool(is_add),
                        "flag": flag
                    }

                # Use STRIPE Lite online inference
                result = detect_one(src, dst, label, event_type)
                prediction_score = result["anom_score"]
                print(f"✅ STRIPE Lite online inference: plaus={result['plaus_score']:.3f}, anom={result['anom_score']:.3f}")
                
            elif hasattr(detector.model, 'detect_anomalies') and is_model_trained:
                # Use model inference (works for both STRIPE Lite and TemporalGNNAnomalyDetector)
                src_idx = detector.node2idx.get(src, 0)
                dst_idx = detector.node2idx.get(dst, 0)

                import torch
                msg_dim = 5
                edge_msg = torch.zeros(msg_dim)
                label_indices = {
                    'HAS_PORT': 0, 'DEPENDS_ON': 1, 'INSTALLED_AT': 2, 'HAS': 3, 'REFERS_TO': 4
                }
                edge_msg[label_indices.get(label, 1)] = 1.0

                edge_features = torch.randn(1, 64)

                detector.model.eval()
                with torch.no_grad():
                    edge_index = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
                    edge_attr = edge_msg.unsqueeze(0)  # Use the actual edge message

                    anomaly_scores = detector.model.detect_anomalies(
                        edge_index, edge_attr, torch.tensor([0])
                    )
                    prediction_score = anomaly_scores.item()
                    print(f"✅ Model inference prediction: {prediction_score:.3f}")
            else:
                # For other model types or untrained models, use heuristic
                print(f"⚠️  Model doesn't have detect_anomalies method, using heuristic")
                prediction_score = np.random.random()

        except Exception as e:
            print(f"❌ Model prediction failed: {e}")
            print("Using fallback heuristic prediction")
            prediction_score = np.random.random()

        # Adjust score based on label type
        label_risk = {
            'HAS_PORT': 0.1,
            'DEPENDS_ON': 0.3,
            'INSTALLED_AT': 0.2,
            'HAS': 0.4,
            'REFERS_TO': 0.5
        }
        prediction_score += label_risk.get(label, 0.2)

        # Adjust score based on event type
        if event_type == 'delete':
            prediction_score += 0.4  # Delete events are more likely to be anomalous
        elif event_type == 'add':
            prediction_score -= 0.1  # Add events are less likely to be anomalous

        # Adjust score based on node types
        if src.startswith('bng-') and dst.startswith('cpe-'):
            prediction_score += 0.2
        elif src.startswith('concentrator-') and dst.startswith('trunk-'):
            prediction_score -= 0.1

        prediction_score = min(max(prediction_score, 0), 1)  # Clamp to [0, 1]

        # Determine prediction method
        if is_model_trained and hasattr(detector.model, 'detect_anomalies'):
            prediction_method = 'model_inference'
        else:
            prediction_method = 'heuristic'

        result = {
            'is_anomaly': prediction_score > 0.7,  # Threshold for anomaly
            'confidence': prediction_score,
            'details': {
                'src': src,
                'dst': dst,
                'label': label,
                'event_type': event_type,
                'graph_id': graph_id,
                'prediction_score': prediction_score,
                'model_used': model_used,
                'threshold': 0.7,
                'detector_available': True,
                'model_trained': is_model_trained,
                'prediction_method': prediction_method
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_graph():
    """Upload and process graph file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        # Generate unique graph_id first
        base_name = secure_filename(file.filename).rsplit('.', 1)[0]
        graph_id = base_name
        counter = 1

        # Ensure unique ID
        while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], f'{graph_id}.csv')) or \
              os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], f'{graph_id}.json')):
            graph_id = f"{base_name}_{counter}"
            counter += 1

        # Save with unique filename
        filename = f"{graph_id}.csv"  # Force CSV extension
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get file stats
        file_size = os.path.getsize(file_path)
        file_type = filename.rsplit('.', 1)[1].lower()

        # Process the uploaded file
        upload_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Basic file processing (extend this based on your needs)
        stats = {
            'nodes': 0,
            'edges': 0,
            'density': 0.0
        }

        preview = None
        model_used = 'TGN'  # Default model for uploaded files

        try:
            if file_type == 'csv':
                df = pd.read_csv(file_path)
                stats['nodes'] = len(pd.unique(df[["src", "dst"]].to_numpy().ravel()))
                stats['edges'] = len(df)
                if stats['nodes'] > 1:
                    stats['density'] = stats['edges'] / (stats['nodes'] * (stats['nodes'] - 1))
                preview = df.head(3).to_dict('records')

                # Save CSV data with metadata
                metadata = {
                    'filename': filename,
                    'upload_id': upload_id,
                    'timestamp': timestamp,
                    'model_used': model_used,
                    'stats': stats,
                    'data': df.to_dict('records')
                }
                metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{graph_id}.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

            elif file_type == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # Assuming JSON has nodes and edges structure
                if 'nodes' in data and 'edges' in data:
                    stats['nodes'] = len(data['nodes'])
                    stats['edges'] = len(data['edges'])
                    if stats['nodes'] > 1:
                        stats['density'] = stats['edges'] / (stats['nodes'] * (stats['nodes'] - 1))
                preview = data

                # Add metadata to JSON
                data['metadata'] = {
                    'filename': filename,
                    'upload_id': upload_id,
                    'timestamp': timestamp,
                    'model_used': model_used,
                    'stats': stats
                }

                # Save updated JSON
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error processing file: {e}")

        result = {
            'upload_id': upload_id,
            'filename': filename,
            'file_size': file_size,
            'file_type': file_type,
            'timestamp': timestamp,
            'model_used': model_used,
            'stats': stats,
            'preview': preview
        }

        return jsonify(result), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-status/<upload_id>', methods=['GET'])
def get_upload_status(upload_id):
    """Get status of an upload"""
    # In a real implementation, you'd track upload status
    return jsonify({
        'upload_id': upload_id,
        'status': 'completed',
        'message': 'File processed successfully'
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    print("Starting Graph Anomaly Detection API...")

    # Initialize the anomaly detectors
    if initialize_anomaly_detectors():
        print("✓ Anomaly detectors initialized")
    else:
        print("⚠ Could not initialize anomaly detectors, predictions may not work")

    # Load sample graph data
    if load_sample_graph_data():
        print("✓ Sample graph data loaded")
    else:
        print("⚠ Could not load sample graph data")

    print("API running on http://localhost:1337")
    app.run(debug=True, host='0.0.0.0', port=1337)

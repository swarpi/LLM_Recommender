# retrieval.py

import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

# Load the new model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-large")
model = AutoModel.from_pretrained("hyp1231/blair-roberta-large")
model.to(device)
model.eval()  # Set model to evaluation mode

def compute_embedding(text):
    """Compute embedding for a given text using the new model."""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {key: value.to(device, dtype=torch.float16) for key, value in inputs.items()}  # Move inputs to device with float16

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]  # CLS token
        embedding = embedding / embedding.norm(dim=1, keepdim=True)  # Normalize embedding

    embedding = embedding.cpu().numpy()[0]  # Convert to numpy array and select first (and only) embedding
    return embedding

def initialize_chromadb(db_path):
    """Initialize the ChromaDB client."""
    db = chromadb.PersistentClient(path=db_path)
    return db

def get_or_create_collection(db, collection_name):
    """Get or create a collection in ChromaDB without an embedding function."""
    if collection_name in [col.name for col in db.list_collections()]:
        collection = db.get_collection(name=collection_name)
    else:
        collection = db.create_collection(name=collection_name)
    return collection

def collect_results_alternating_shortest(product_names, collection):
    """Collect results by computing embeddings for product names and querying ChromaDB."""
    if not product_names:
        return -1  # Return -1 if product_names is empty

    doc_distance_map = defaultdict(list)
    final_result = []

    # Step 1: Collect results for each product name
    for product_name in product_names:
        # Compute embedding for the product name
        query_embedding = compute_embedding(product_name)
        # Query ChromaDB using the embedding
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=10
        )
        print(f'results {results}')
        # Check if results are empty or documents are None
        if not results['documents'] or not results['documents'][0]:
            continue
        # Store documents, distances, and metadatas
        doc_distance_map[product_name] = list(zip(results['documents'][0], results['distances'][0], results['metadatas'][0]))
    
    # Check if any results were found
    if not doc_distance_map:
        return -1  # Return -1 if no results were found

    # Step 2: Iteratively pick the closest documents
    while len(final_result) < 10:
        for product_name in product_names:
            if product_name in doc_distance_map and doc_distance_map[product_name]:
                # Sort by distance (ascending)
                sorted_results = sorted(doc_distance_map[product_name], key=lambda x: x[1])
                closest_document, distance, metadata = sorted_results.pop(0)
                if closest_document not in [doc for doc, _ in final_result]:
                    final_result.append((closest_document, metadata))
                doc_distance_map[product_name] = sorted_results
            if len(final_result) >= 10:
                break
        else:
            # If no more documents can be added, break the while loop
            break

    if not final_result:
        return -1  # Return -1 if no results were found

    return final_result  # Return list of tuples (document, metadata)

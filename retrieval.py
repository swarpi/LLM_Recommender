# retrieval.py

import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the new model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-large")
model = AutoModel.from_pretrained(
    "hyp1231/blair-roberta-large",
    torch_dtype=torch.float16  # Load model in float16
).to(device)
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
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to device without changing dtype

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

from collections import defaultdict

from collections import defaultdict

from collections import defaultdict

def collect_results_per_product(product_names, collection, max_products=20):
    if not product_names:
        print("Debug: No product names provided.")
        return -1  # Return -1 if product_names is empty

    # Clean product names
    product_names = [name.strip('*').strip() for name in product_names]
    print(f"Debug: Cleaned product names: {product_names}")

    doc_distance_map = defaultdict(list)
    final_result = []
    seen_documents = set()
    seen_ids = set()

    # Step 1: Collect results for each product name
    for product_name in product_names:
        print(f"\nDebug: Processing product name: '{product_name}'")
        # Compute embedding for the product name
        query_embedding = compute_embedding(product_name)
        print(f"Debug: Computed embedding for '{product_name}'")

        # Query ChromaDB using the embedding
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5  # Query 5 items per product name
        )
        print(f"Debug: Query results for '{product_name}': {results}")

        # Check if results are empty or documents are None
        if not results['documents'] or not results['documents'][0]:
            print(f"Debug: No documents found for '{product_name}'.")
            continue

        # Store documents, distances, and metadatas
        zipped_results = list(
            zip(results['documents'][0], results['distances'][0], results['metadatas'][0])
        )
        doc_distance_map[product_name] = zipped_results
        print(f"Debug: Stored {len(zipped_results)} results for '{product_name}'")

    # Check if any results were found
    if not doc_distance_map:
        print("Debug: No results found for any product names.")
        return -1  # Return -1 if no results were found

    # Step 2: Collect the best item from each product name
    print("\nDebug: Collecting the best item from each product name.")
    for product_name in product_names:
        if product_name in doc_distance_map:
            # Sort the results for this product_name by distance
            sorted_results = sorted(doc_distance_map[product_name], key=lambda x: x[1])  # x[1] is the distance
            for document, distance, metadata in sorted_results:
                product_id = metadata.get('metadata')  # Adjust key if necessary
                if product_id not in seen_ids:
                    final_result.append((document, distance, metadata))
                    seen_documents.add(document)
                    seen_ids.add(product_id)
                    print(f"Debug: Added best document '{document}' for '{product_name}'")
                    break  # Only take the first item for this product_name
                else:
                    print(f"Debug: Duplicate product ID '{product_id}' found. Skipping.")
        else:
            print(f"Debug: No results found for product name '{product_name}'")

    # If we have reached max_products, return
    if len(final_result) >= max_products:
        print(f"Debug: Reached max products limit of {max_products} after collecting best items.")
        return final_result[:max_products]

    # Step 3: Collect remaining items, sorted by distance, excluding already added products
    print("\nDebug: Collecting remaining items to fill up to max products.")
    remaining_items = []
    for product_name in product_names:
        if product_name in doc_distance_map:
            for document, distance, metadata in doc_distance_map[product_name]:
                product_id = metadata.get('metadata')
                if product_id not in seen_ids:
                    remaining_items.append((document, distance, metadata))

    # Sort remaining items by distance
    remaining_items_sorted = sorted(remaining_items, key=lambda x: x[1])

    # Calculate number of slots left
    slots_left = max_products - len(final_result)
    final_result.extend(remaining_items_sorted[:slots_left])

    print(f"\nDebug: Final result contains {len(final_result)} items.")
    return final_result  # Return list of tuples (document, distance, metadata)

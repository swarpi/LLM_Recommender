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

def collect_results_per_product(product_names, collection, user_history, max_products=20):
    if not product_names:
        print("Debug: No product names provided.")
        return -1  # Return -1 if product_names is empty

    # Clean product names
    product_names = [name.strip('*').strip() for name in product_names]
    print(f"Debug: Cleaned product names: {product_names}")

    doc_distance_map = defaultdict(list)
    final_result = []
    seen_documents = set()
    seen_ids = set(user_history)  # Initialize seen_ids with user_history

    # Step 1: Collect results for each product name
    for product_name in product_names:
        print(f"\nDebug: Processing product name: '{product_name}'")
        # Compute embedding for the product name
        query_embedding = compute_embedding(product_name)
        print(f"Debug: Computed embedding for '{product_name}'")

        # Query ChromaDB using the embedding
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=10  # Query 5 items per product name
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
                    print(f"Debug: Product ID '{product_id}' in user history or already seen. Skipping.")
        else:
            print(f"Debug: No results found for product name '{product_name}'")

    # If we have reached max_products, return
    if len(final_result) >= max_products:
        print(f"Debug: Reached max products limit of {max_products} after collecting best items.")
        return final_result[:max_products]

    # Step 3: Collect remaining items in a round-robin fashion
    print("\nDebug: Collecting remaining items in a round-robin fashion.")
    index_per_product = {product_name: 0 for product_name in product_names}  # Track indices for each product_name

    while len(final_result) < max_products:
        added_any = False  # Flag to check if any item was added in this iteration
        for product_name in product_names:
            if len(final_result) >= max_products:
                break
            if product_name in doc_distance_map:
                sorted_results = sorted(doc_distance_map[product_name], key=lambda x: x[1])
                idx = index_per_product[product_name]
                # Skip the first item as it's already taken in step 2
                while idx < len(sorted_results):
                    document, distance, metadata = sorted_results[idx]
                    idx += 1
                    product_id = metadata.get('metadata')  # Adjust key if necessary
                    if product_id not in seen_ids:
                        final_result.append((document, distance, metadata))
                        seen_documents.add(document)
                        seen_ids.add(product_id)
                        index_per_product[product_name] = idx  # Update index
                        added_any = True
                        print(f"Debug: Added document '{document}' for '{product_name}'")
                        break
                    else:
                        print(f"Debug: Product ID '{product_id}' in user history or already seen. Skipping.")
                index_per_product[product_name] = idx  # Update index even if no item was added
        if not added_any:
            print("Debug: No more unique items to add. Ending collection.")
            break  # Exit if no new items were added in the full cycle

    print(f"\nDebug: Final result contains {len(final_result)} items.")
    return final_result[:max_products]  # Return the list of tuples (document, distance, metadata)

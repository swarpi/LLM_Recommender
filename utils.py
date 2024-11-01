# utils.py
import re

def extract_latest_n_reviews(data, n):
    """Extract the latest 'n' reviews from the data."""
    review = []
    for user in data:
        reviews = user['reviews']
        # Ensure reviews are sorted by timestamp (earliest to latest)
        sorted_reviews = sorted(reviews, key=lambda x: x['timestamp'])
        # Get the last 'n' reviews
        #latest_reviews = sorted_reviews[-n:]
        review.extend(sorted_reviews)
    return review

def extract_product_names(response_text):
    """Extract product names from the response text."""
    product_names = []
    current_category = None
    # Split the response into lines
    lines = response_text.strip().split('\n')
    for line in lines:
        # Remove leading whitespace to check indentation
        stripped_line = line.lstrip()
        # Calculate indentation level
        indent_level = len(line) - len(stripped_line)
        # Match lines that start with a number and period
        match = re.match(r'^(\d+)\.\s*(.*)', stripped_line)
        if match:
            number = int(match.group(1))
            text = match.group(2).strip()
            if indent_level == 0:
                # This is a category title
                current_category = text
                # Optionally, store category names if needed
            else:
                # This is a product name under the current category
                product_name = text
                product_names.append(product_name)
    return product_names

def extract_ranked_products(response_text):
    ranked_products = []
    lines = response_text.strip().split('\n')
    for line in lines:
        match = re.match(r'^\d+\.\s*(.*)', line)
        if match:
            product_name = match.group(1).strip()
            ranked_products.append(product_name)
    return ranked_products

def remove_duplicate_products(final_results):
    """Remove duplicate products based on metadata IDs."""
    unique_products = []
    seen_ids = set()
    for document, distance, metadata in final_results:
        product_id = metadata.get('metadata')  # Adjust key if necessary
        if product_id not in seen_ids:
            unique_products.append((document, distance, metadata))
            seen_ids.add(product_id)
        else:
            print(f"Duplicate product ID '{product_id}' found. Removing duplicate.")
    return unique_products


def limit_products(final_results, max_products=20):
    """Limit the number of products to the top N based on distance."""
    # Sort the final_results based on distance (lower distance means higher relevance)
    sorted_results = sorted(final_results, key=lambda x: x[1])  # x[1] is the distance
    # Select the top N products
    limited_results = sorted_results[:max_products]
    return limited_results



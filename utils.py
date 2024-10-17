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
        latest_reviews = sorted_reviews[-n:]
        review.extend(latest_reviews)
    return review

def extract_product_names(response_text):
    """Extract product names from the response text."""
    product_names = []
    # Split the response into lines
    lines = response_text.strip().split('\n')
    for line in lines:
        # Match lines that start with a number and period
        match = re.match(r'^\d+\.\s*(.*)', line)
        if match:
            product_name = match.group(1).strip()
            # Remove "Name:" prefix if present
            if product_name.lower().startswith('name:'):
                product_name = product_name[5:].strip()
            product_names.append(product_name)
    return product_names

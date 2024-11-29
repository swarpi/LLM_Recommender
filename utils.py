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


import re

import re

def extract_product_names(response_text):
    """
    Extract product names from the response text after detecting keywords containing
    'candidate' or 'category'. Handles various list formats including items in individual
    square brackets separated by commas.
    """
    product_names = []

    # Split the text into lines
    lines = response_text.strip().split('\n')
    start_extracting = False

    for line in lines:
        stripped_line = line.strip()

        if not start_extracting:
            # Check if the line contains any form of 'candidate' or 'category' (case-insensitive)
            keyword_match = re.search(r'(candidate|category)', stripped_line, re.IGNORECASE)
            if keyword_match:
                start_extracting = True
                # Extract items from the line if any, excluding 'username'
                items_in_brackets = re.findall(r'\[([^\]]+)\]', stripped_line)
                if items_in_brackets:
                    for item in items_in_brackets:
                        item = item.strip()
                        if item.lower() != 'username':
                            product_names.append(item)
                continue  # Proceed to next line
            continue  # Skip to the next line until the keyword is found

        # Skip empty lines
        if not stripped_line:
            continue

        # Attempt to extract product names

        # Handle numbered list items with bolded product names
        match_numbered_bold_item = re.match(r'^\d+[\.\)-]?\s*\*\*(.+?)\*\*', stripped_line)
        if match_numbered_bold_item:
            item_text = match_numbered_bold_item.group(1).strip()
            product_names.append(item_text)
            continue

        # Handle lines with items in square brackets, excluding 'username'
        items_in_brackets = re.findall(r'\[([^\]]+)\]', stripped_line)
        if items_in_brackets:
            for item in items_in_brackets:
                item = item.strip()
                if item.lower() != 'username':
                    product_names.append(item)
            continue

        # Handle lines with semicolon-separated items
        if ';' in stripped_line:
            items = [item.strip() for item in stripped_line.split(';') if item.strip()]
            product_names.extend(items)
            continue

        # Handle numbered list items with optional dot and space
        match_numbered_item = re.match(r'^\d+[\.\)-]?\s*"?(.+?)"?$', stripped_line)
        if match_numbered_item:
            item_text = match_numbered_item.group(1).strip('"').strip()
            if item_text.lower() != 'username':
                product_names.append(item_text)
            continue

        # Handle bulleted list items starting with "-", "*", "+"
        match_bullet_item = re.match(r'^[-*+]\s+(.*)', stripped_line)
        if match_bullet_item:
            item_text = match_bullet_item.group(1).strip()
            product_names.append(item_text)
            continue

        # Handle standalone quoted items
        match_quoted_line = re.match(r'^"(.+?)"$', stripped_line)
        if match_quoted_line:
            item_text = match_quoted_line.group(1).strip()
            product_names.append(item_text)
            continue

        # Stop extraction if a new section is detected
        if re.match(r'^[A-Z][A-Za-z0-9_\s]*:$', stripped_line):
            break

    return product_names





import re

def extract_product_names_adapter(response_text):
    """Extract product names from the response text based on list indicators."""
    product_names = []
    lines = response_text.strip().split('\n')
    start_extracting = False

    # Pattern to detect list items (numbered or bullet points)
    list_item_pattern = re.compile(r'^(\d+[\.\)-]?|[-*+])\s+(.*)')

    for line in lines:
        stripped_line = line.strip()

        if not start_extracting:
            # Check if the line starts with a list indicator
            if list_item_pattern.match(stripped_line):
                start_extracting = True
            else:
                continue  # Skip lines until we find the start of the list

        # Now we are in extraction mode
        match_list_item = list_item_pattern.match(stripped_line)
        if match_list_item:
            # Extract the text after the list indicator
            item_text = match_list_item.group(2).strip()
            product_names.append(item_text)
        else:
            # Handle continuation lines
            if stripped_line == '':
                continue  # Skip empty lines
            else:
                # Append to the previous item if it's a continuation
                if product_names:
                    product_names[-1] += ' ' + stripped_line
                else:
                    continue  # Skip if there's no previous item

    return product_names



def extract_product_names_alpaca(response_text):
    """Extract product names from the response text, handling various formats."""
    product_names = []
    lines = response_text.strip().split('\n')
    collecting = False  # Flag to indicate if we are in the candidate items section
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        # Check for 'List of Candidate Items' or similar phrases
        if (re.search(r'List of candidate items', stripped_line, re.IGNORECASE) or re.search(r'candidate items', stripped_line, re.IGNORECASE)):
            collecting = True
            continue  # Move to the next line
        
        # If we're in the candidate items section
        if collecting:
            # Check for empty line indicating the end of the section
            if stripped_line == '':
                collecting = False
                continue
            # Match bullet points or numbered lists
            match = re.match(r'^[-+*]\s*(.*)', stripped_line)
            if match:
                product_name = match.group(1).strip()
                # Optionally remove parenthetical remarks
                product_name = re.sub(r'\s*\(.*?\)\s*', '', product_name)
                product_names.append(product_name)
            else:
                # Also handle numbered items
                match = re.match(r'^\d+\.\s*(.*)', stripped_line)
                if match:
                    product_name = match.group(1).strip()
                    product_name = re.sub(r'\s*\(.*?\)\s*', '', product_name)
                    product_names.append(product_name)
                else:
                    # Handle lines that may be continuations of the previous line
                    if product_names and stripped_line:
                        product_names[-1] += ' ' + stripped_line
            continue  # Move to the next line
        
        # Previous logic for handling other formats
        # Match lines that start with a number and a period
        match = re.match(r'^(\d+)\.\s*(.*)', stripped_line)
        if match:
            product_name = match.group(2).strip()
            product_name = re.sub(r'\s*\(.*?\)\s*', '', product_name)
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



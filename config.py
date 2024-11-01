# config.py

# Language model parameters
TOKENIZER_PATH = "models/hf-frompretrained-download/meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_PATH = "models/hf-frompretrained-downloadmeta-llama/Meta-Llama-3-8B-Instruct"
PIPELINE_PARAMS = {
    'max_length': 4096,
    'num_return_sequences': 1,
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 0.95,
    'repetition_penalty': 1.2,
    # 'pad_token_id' will be set in the code after tokenizer is initialized
}

# Prompts
USER_PROFILE_PROMPT = """
You are an e-commerce recommender specialist. Your task is to create a comprehensive user profile based on the following reviews, listed in chronological order (oldest to newest):

{reviews}

Analyze this information and create a user profile following these steps:

1. Long-term preferences: Identify themes and valued attributes.
2. Short-term interests: Note recent preferences or emerging interests.
3. Demographic information: Infer age range, possible gender, and lifestyle.
4. User profile summary: Combine insights into a concise profile.

Present your analysis in a structured format, using clear headings for each section. Do not include any code in your response. Focus on creating a vivid picture of the user's preferences, habits, and potential future interests. The created User Profile should not exceed 200 words.
"""

PRELIMINARY_RECOMMENDATIONS_PROMPT = """You are an expert recommendation system.

Based on the following user profile:

{user_profile}

Generate three general product categories that match the user's preferences and interests.

For each category, provide:

- **Category Title**
- A numbered list of three specific product types or items within that category that fit the user profile.

**Guidelines:**

**Now, generate the recommendations for the user profile provided, following the same format as the example. Do not include any introductions or explanations.**- Use plain text with a numbered list.
**Do not include any introductions, conclusions, or additional commentary**
**Do NOT include any python code or Json code**
**Example:**

1. Skincare Essentials
   1. Organic Face Cream with SPF
   2. Natural Exfoliating Scrub
   3. Hydrating Serum with Green Tea Extract

2. Eco-Friendly Hair Care
   1. Sulfate-Free Shampoo Bar
   2. Coconut Oil Leave-In Conditioner
   3. Bamboo Bristle Brush Set

3. Personal Grooming Tools
   1. Rechargeable Electric Razor
   2. Stainless Steel Nail Clippers
   3. Silicone Travel Toothbrush Holder

"""



# Additional configuration parameters can be added here.
RANKING_PROMPT = """
You are an expert recommendation system.

Based on the following user profile:

{user_profile}

And the following list of products:

{products}

Rank the products in the list from most relevant to least relevant for the user, based on the user profile.

Provide the ranked list of products in the following format:

1. Product Name
2. Product Name
3. Product Name
...

Do not include any explanations or additional text.
"""

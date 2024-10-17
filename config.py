# config.py

# Language model parameters
TOKENIZER_PATH = "models/hf-frompretrained-download/meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_PATH = "models/hf-frompretrained-downloadmeta-llama/Meta-Llama-3-8B-Instruct"
PIPELINE_PARAMS = {
    'max_length': 2048,
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

1. Name: **[Non-ambiguous product or category name]**
   Reason: [1-2 sentences explaining the fit]

**Guidelines:**

- Do not mention specific brands or product names, except if there is any preference mentioned in the User Profile.
- Use plain text with a numbered list.
- **Do not include any introductions, conclusions, or additional commentary.**
- **Do not repeat the guidelines or any part of this prompt in your response.**
- **Your entire response should consist only of the numbered list in the exact format specified.**
"""

# Additional configuration parameters can be added here.

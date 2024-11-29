# config.py

# Language model parameters
TOKENIZER_PATH = "models/hf-frompretrained-download/meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_PATH = "models/hf-frompretrained-downloadmeta-llama/Meta-Llama-3-8B-Instruct"

# Dynamic model path for experiments with different sample sizes
def get_model_path_user_profile(sample_size):
    return f"outputs/best_model_{sample_size}_samples"

def get_tokenizer_path_user_profile(sample_size):
    return f"outputs/best_model_{sample_size}_samples"

# Dynamic model path for experiments with different sample sizes
def get_model_path_user_profile_and_candidate_items(sample_size):
    return f"outputs/best_model_up_ci_{sample_size}_samples"

def get_tokenizer_path_user_profile_and_candidate_items(sample_size):
    return f"outputs/best_model_up_ci_{sample_size}_samples"

PIPELINE_PARAMS = {
    'max_length': 2048,
    'num_return_sequences': 1,
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 0.95,
    'repetition_penalty': 1.2,
            # 'pad_token_id' will be set in the code after tokenizer is initialized
}
# QLoRA Fine-tuning Parameters
QLORA_PARAMS = {
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'lora_target_modules': ['q_proj', 'v_proj'],
    'gradient_accumulation_steps' : 2,
    'lora_num_epochs': 2,
    'lora_val_iterations': 100,
    'lora_early_stopping_patience': 10,
    'lora_lr': 1e-4,
    'lora_micro_batch_size': 1
}

# Alpaca-LoRA Instruction Templates
ALPACA_LORA_PROMPTS_USER_PROFILE = {
        'instruction': "### Instruction:\n You are a recommender system specialized in creating user profiles. Your task is to create a comprehensive user profile and a list of candidate items based on a user's review. The user Profile consists of: 1. Short-term Interests: Examine the user's most recent items along with their personalized descriptions 2. Long-term Preferences: Analyze all reviews and items from the user’s entire history to capture deeper, stable interests that define the user’s lasting preferences. Use this comprehensive historical data to outline consistent themes and inclinations that have remained steady over time. 3. User Profile Summary: Synthesize these findings into a concise profile (max 200 words) that combines insights from both short-term interests and long-term preferences. Use the short-term interests as a query to refine or highlight relevant themes from the long-term history, and provide a cohesive picture of the user’s tastes, typical habits, and potential future interests.Present your analysis under clear section headings.Do Not include any Code in your response. Think step by step",
        'input': "{user_review}",
        'output': "### Response:"
    }

ALPACA_LORA_PROMPTS_USER_PROFILE_AND_CANDIDATE_ITEMS = {
        'instruction': "### Instruction:\n You are a recommender system specialized in creating user profiles. Your task is to create a comprehensive user profile and a list of candidate items based on a user's review. The user Profile consists of: 1. Short-term Interests: Examine the user's most recent items along with their personalized descriptions 2. Long-term Preferences: Analyze all reviews and items from the user’s entire history to capture deeper, stable interests that define the user’s lasting preferences. Use this comprehensive historical data to outline consistent themes and inclinations that have remained steady over time. 3. User Profile Summary: Synthesize these findings into a concise profile (max 200 words) that combines insights from both short-term interests and long-term preferences. Use the short-term interests as a query to refine or highlight relevant themes from the long-term history, and provide a cohesive picture of the user’s tastes, typical habits, and potential future interests. Afterwards generate five Candidate Items that are general product categories that align with the user's preferences and interests. Approach this task by treating these categories as a cohesive set, ensuring that they collectively reflect the user’s overall profile and maximize satisfaction. ",
        'input': "User Reviews: {user_review}",
        'output': "### Response:"
    }
ALPACA_LORA_PROMPTS_CANDIDATE_ITEMS = {
    "instruction":  "### Instruction:\n You are a recommender system specialized. Based on the following user profile text, generate a list of general product categories that align with the user's preferences and interests. Approach this task by treating these categories as a cohesive set, ensuring that they collectively reflect the user’s overall profile and maximize satisfaction. ",
        'input': "### Input \n User Profile: \n {user_profile}",
        'output': "### Response:"
}
# Prompts
USER_PROFILE_AND_CANDIDATE_ITEM = """
You are a recommender system specialized in creating user profiles and generating product categories.
Based on the following user reviews, listed in chronological order (oldest to newest):
{reviews}
**Task 1: User Profile Creation**
Analyze this information and create a user profile by following these steps:
1. **Short-term Interests:** Examine the user's most recent items along with their personalized descriptions.
2. **Long-term Preferences:** Analyze all reviews from the user's entire history to capture deeper, stable interests that define the user's lasting preferences. Outline consistent themes and inclinations that have remained steady over time.
3. **User Profile Summary:** Synthesize these findings into a concise profile (maximum 200 words) that combines insights from both short-term interests and long-term preferences. Provide a cohesive picture of the user's tastes, typical habits, and potential future interests.
**Task 2: Product Categories Generation**
Based on the user profile you've created, generate five general product categories that align with the user's preferences and interests. Treat these categories as a cohesive set to reflect the user's overall profile and maximize satisfaction.
- **Identify Unique Aspects:** Create five distinct categories that capture different aspects of the user's profile.
- **Rank by Relevance:** Order the categories by their relevance to the user's profile, from most to least relevant.

Present your analysis under clear section headings.Do not include any code in your response.Present the product categories as a numbered list that only contains the category name.

Example Output:
User Profile:
1."Short-Term Interests": "The user recently reviewed products focused on facial skincare, specifically creams and serums targeting anti-aging concerns such as fine lines, wrinkles, and dehydration. They showed interest in natural and organic ingredients, vegan-friendly options, and cruelty-free practices. Their preference leans towards face washes and moisturizers that cater to specific skin concerns like acne-prone, sensitive, and dry skin.",
2."Long-term Preferences": "Based on the entirety of their reviews, some common denominators emerge:\n* Focus on natural ingredients, particularly plant-based extracts, essential oils, and herbal remedies\n* Concern for anti-aging and rejuvenation, seeking effective solutions for maintaining healthy-looking skin\n* Interest in sheet masks, especially those with unique features like eye patches and targeted application areas\n* Appreciation for moisturized and hydrated skin, often mentioning specific requirements like deep hydration and non-sticky textures\n* Willingness to explore various brands and products, demonstrating adaptability and open-mindedness",
3."User Profile Summary": "Our user appears to be someone who prioritizes natural and organic approaches to skincare while focusing on addressing specific skin concerns. They exhibit a willingness to experiment with diverse products and techniques to achieve optimal results. As they continue exploring the world of skincare, they may gravitate toward more niche markets, such as bespoke formulas tailored to individual skin types or advanced technology-driven treatments. For now, our recommendation would focus on recommending products featuring natural, sustainable, and innovative formulations catering to their varied skin needs, including hydration, anti-aging, and sensitivity management. A suggested next step could involve introducing them to emerging trends in customized skincare regimens and cutting-edge technologies for enhanced customer experience."
Product Categories:
1. Sustainable Beauty Products
2. Luxury Skincare Essentials
3. Everyday Makeup Basics
4. Seasonal Self-Care Kits
5. Innovative Hair Care Solutions
"""

USER_PROFILE_PROMPT = """
You are a recommender system specialized in creating user profiles. Your task is to create a comprehensive user profile based on the following reviews, listed in chronological order (oldest to newest):

{reviews}

Analyze this information and create a user profile following these steps:

1. Short-term Interests: Examine the user's most recent items along with their personalized descriptions
2. Long-term Preferences: Analyze all reviews and items from the user’s entire history to capture deeper, stable interests that define the user’s lasting preferences. Use this comprehensive historical data to outline consistent themes and inclinations that have remained steady over time.
3. User Profile Summary: Synthesize these findings into a concise profile (max 200 words) that combines insights from both short-term interests and long-term preferences. Use the short-term interests as a query to refine or highlight relevant themes from the long-term history, and provide a cohesive picture of the user’s tastes, typical habits, and potential future interests

Present your analysis under clear section headings. Ensure the final profile reflects both the user’s stable and current interests for a comprehensive understanding of their preferences.Do Not include any Code in your response"""

PRELIMINARY_RECOMMENDATIONS_PROMPT = """You are a recommender system specialized in generating product categories based on user profiles.

Based on the following user profile:

{user_profile}

Generate five general product categories that align with the user's preferences and interests. Approach this task by treating these categories as a cohesive set, ensuring that they collectively reflect the user’s overall profile and maximize satisfaction.

Instructions:

    Identify Unique Aspects: Create five distinct categories that capture different aspects of the user’s profile, aiming to reflect a broad range of their preferences and interests.
    Enhance Cohesion: Ensure that each category complements the others, creating a set that feels cohesive and well-rounded.
    Rank by Relevance: Order the categories by their relevance to the user’s profile, from most to least relevant, while ensuring each adds meaningful variety to the list.
Present the result as a numbered list. Do not include any code in your response.
Example Output:
Candidate_Items:
1. Sustainable Beauty Products – Reflecting the user’s long-standing preference for eco-friendly and organic items.
2. Luxury Skincare Essentials – Highlighting interest in premium skincare brands, often favoring high-quality, indulgent products.
3. Everyday Makeup Basics – Covering frequently purchased items for daily use, aligned with recent purchases of affordable, versatile products.
4. Seasonal Self-Care Kits – Capturing a trend of seasonally themed sets, ideal for users who enjoy holiday promotions and gift-ready collections.
5. Innovative Hair Care Solutions – Targeting a new interest in specialized hair treatments, noted in recent product searches and reviews.

"""
PRELIMINARY_RECOMMENDATIONS_PROMPT_PRODUCT_NAME_ONLY = """You are a recommender system specialized in generating product categories based on user profiles.
Based on the following user profile:
{user_profile}
Generate five general product categories that align with the user's preferences and interests. Approach this task by treating these categories as a cohesive set, ensuring that they collectively reflect the user’s overall profile and maximize satisfaction.
Instructions:

    Identify Unique Aspects: Create five distinct categories that capture different aspects of the user’s profile, aiming to reflect a broad range of their preferences and interests.
    Enhance Cohesion: Ensure that each category complements the others, creating a set that feels cohesive and well-rounded.
    Rank by Relevance: Order the categories by their relevance to the user’s profile, from most to least relevant, while ensuring each adds meaningful variety to the list.

Present the result as a numbered list that only contains the category name. Do not include any code in your response.
Example Output:
Candidate_Items:
1. Sustainable Beauty Products 
2. Luxury Skincare Essentials 
3. Everyday Makeup Basics 
4. Seasonal Self-Care Kits 
5. Innovative Hair Care Solutions
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

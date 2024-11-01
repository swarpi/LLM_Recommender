# model_pipeline.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import TOKENIZER_PATH, MODEL_PATH, PIPELINE_PARAMS, USER_PROFILE_PROMPT, PRELIMINARY_RECOMMENDATIONS_PROMPT, RANKING_PROMPT
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class RecommenderModel:
    def __init__(self):
        # Initialize tokenizer and model
        print(TOKENIZER_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        
        # Initialize pipeline
        self.pipeline = self.initialize_pipeline()
    
    def initialize_pipeline(self):
        # Add pad_token_id to PIPELINE_PARAMS
        pipeline_params = PIPELINE_PARAMS.copy()
        pipeline_params['pad_token_id'] = self.tokenizer.eos_token_id
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            eos_token_id=self.tokenizer.eos_token_id,
            device_map="auto",
        )
        return pipe
    print(f"max length is {PIPELINE_PARAMS['max_length']}")
    def get_response(self, prompt):
        sequences = self.pipeline(
            prompt,
            max_length=PIPELINE_PARAMS['max_length'],
            num_return_sequences=PIPELINE_PARAMS['num_return_sequences'],
            temperature=PIPELINE_PARAMS['temperature'],
            top_k=PIPELINE_PARAMS['top_k'],
            top_p=PIPELINE_PARAMS['top_p'],
            repetition_penalty=PIPELINE_PARAMS['repetition_penalty'],
            pad_token_id=self.tokenizer.eos_token_id,
            truncation=True,
        )
        gen_text = sequences[0]["generated_text"]
        # Remove the input prompt from the generated text
        response = gen_text[len(prompt):].strip()
        return response
    
    def create_user_profile(self, reviews):
        prompt = USER_PROFILE_PROMPT.format(reviews=reviews)
        return self.get_response(prompt)
    
    def create_preliminary_recommendations(self, user_profile):
        prompt = PRELIMINARY_RECOMMENDATIONS_PROMPT.format(user_profile=user_profile)
        return self.get_response(prompt)
    
    def rank_recommendations(self, user_profile, products):
        prompt = RANKING_PROMPT.format(user_profile=user_profile, products=products)
        return self.get_response(prompt)


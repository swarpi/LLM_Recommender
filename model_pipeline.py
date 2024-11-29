# model_pipeline.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import *
import torch
from peft import PeftModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RecommenderModel:
    def __init__(self, sample_size=None, model_type='user_profile', adapter = False):
        # Dynamically get model and tokenizer paths based on sample size and model type
        if model_type == 'both' and sample_size != None:
            print("model is both")
            model_path = get_model_path_user_profile_and_candidate_items(sample_size)
            print(f"model path is {model_path}")
            tokenizer_path = get_tokenizer_path_user_profile_and_candidate_items(sample_size)
        else:
            model_path = MODEL_PATH
            tokenizer_path = TOKENIZER_PATH
                        # Initialize tokenizer and model with the dynamic paths
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.pipeline = self.initialize_pipeline()
        if(adapter):
            # Initialize pipeline
            self.adapter_path_user_profile = f"outputs/adapter_test_user_profile_epoch_{QLORA_PARAMS['lora_num_epochs']}_{sample_size}_samples"
            self.adapter_name_user_profile = "user_profile"
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path_user_profile, adapter_name=self.adapter_name_user_profile)

            print(f"Loaded user profile adapter: {self.adapter_name_user_profile}")

            # Load the candidate items adapter
            self.adapter_path_candidate_items = f"outputs/adapter_test_candidate_items_epoch_{QLORA_PARAMS['lora_num_epochs']}_{sample_size}_samples"
            self.adapter_name_candidate_items = "candidate_items"
            self.model.load_adapter(self.adapter_path_candidate_items, adapter_name=self.adapter_name_candidate_items)
            # Print the list of adapters loaded into the model
            print(f"Loaded candidate items adapter: {self.adapter_name_candidate_items}")
            print("Adapters loaded in the model:", list(self.model.peft_config.keys()))
        #self.model.eval()
        
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
    
    def create_user_profile_and_candidate_items(self, reviews):
        prompt = USER_PROFILE_AND_CANDIDATE_ITEM.format(reviews=reviews)
        return self.get_response(prompt)

    def create_user_profile_alpaca(self, reviews):
        prompt = (
            ALPACA_LORA_PROMPTS_USER_PROFILE['instruction'] + "\n\n" +
            ALPACA_LORA_PROMPTS_USER_PROFILE['input'].replace("{user_review}", reviews)+"\n"+
            ALPACA_LORA_PROMPTS_USER_PROFILE['output']
        )
        return self.get_response(prompt)
    
    def create_user_profile_alpaca_adapter(self, reviews):
        print(f"\nSetting active adapter to: {self.adapter_name_user_profile}")
        self.model.set_adapter(self.adapter_name_user_profile)
        print(f"Current active adapter: {self.model.active_adapter}")
        prompt = (
            ALPACA_LORA_PROMPTS_USER_PROFILE['instruction'] + "\n\n" +
            ALPACA_LORA_PROMPTS_USER_PROFILE['input'].replace("{user_review}", reviews)+"\n"+
            ALPACA_LORA_PROMPTS_USER_PROFILE['output']
        )
        return self.get_response(prompt)
    
    def create_preliminary_recommendations_alpaca_adapter(self, user_profile):
        print(f"\nSetting active adapter to: {self.adapter_name_candidate_items}")
        self.model.set_adapter(self.adapter_name_candidate_items)
        print(f"Current active adapter: {self.model.active_adapter}")
        prompt = (
            ALPACA_LORA_PROMPTS_CANDIDATE_ITEMS['instruction'] + "\n\n" +
            ALPACA_LORA_PROMPTS_CANDIDATE_ITEMS['input'].replace("{user_profile}", user_profile)+"\n"+
            ALPACA_LORA_PROMPTS_CANDIDATE_ITEMS['output']
        )
        print(prompt)
        return self.get_response(prompt)
    
    def create_user_profile_and_candidate_items_alpaca(self, user_review):
        prompt = (
            ALPACA_LORA_PROMPTS_USER_PROFILE_AND_CANDIDATE_ITEMS['instruction'] + "\n\n" +
            ALPACA_LORA_PROMPTS_USER_PROFILE_AND_CANDIDATE_ITEMS['input'].replace("{user_review}", user_review) +"\n"+ 
            ALPACA_LORA_PROMPTS_USER_PROFILE_AND_CANDIDATE_ITEMS['output']
        )
        return self.get_response(prompt)
    
    def create_preliminary_recommendations(self, user_profile):
        prompt = PRELIMINARY_RECOMMENDATIONS_PROMPT.format(user_profile=user_profile)
        return self.get_response(prompt)
    
    def create_preliminary_recommendations_product_name_only(self, user_profile):
        prompt = PRELIMINARY_RECOMMENDATIONS_PROMPT_PRODUCT_NAME_ONLY.format(user_profile=user_profile)
        return self.get_response(prompt)
    

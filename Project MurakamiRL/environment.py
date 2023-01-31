import gymnasium as gym
import numpy as np
import torch
from transformers import T5Tokenizer
import random

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-small")
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

class RLEnv(gym.Env):
    def __init__(self, model, tokenizer, observation_input=[], max_length=100):
        self.vocabs = list(dict(tokenizer.get_vocab().items()).keys())
        self.action_space =  gym.spaces.Discrete(len(self.vocabs))
        self.actions = self.vocabs
        self.model = model
        self.tokenizer = tokenizer
        self.observation_space = observation_input
        self.target_table = {}
        self.input_item = [""]
        self.predicted = []
        self.env_max_length = self.model.config.max_length
        self.gen_stop_toks = []
        self.reset()

        if self.tokenizer.sep_token:
            self.gen_stop_toks.append(self.tokenizer.sep_token)
        if self.tokenizer.eos_token:
            self.gen_stop_toks.append(self.tokenizer.eos_token)
    
    def get_obs_input(self, input_item):
        return input_item[0]

    def reset(self, input_item=None):
        self.predicted = []
        if input_item is None:
            self.input_item = random.choice(self.observation_space)
        else:
            self.input_item = input_item
        return self._get_obs()

    
    def _get_obs(self, predicted=[]):
        with torch.inference_mode():
            p_text = self.tokenizer.convert_tokens_to_string(predicted)
            feature_dict = self.tokenizer([[self.get_obs_input(self.input_item), p_text]], return_tensors="pt", add_special_tokens=False).to(self.model.device)
            prediction = self.model(**feature_dict, output_hidden_states=True)
            outputs = prediction.hidden_states[-1].squeeze(0)

            return outputs.data[-1]
    
    def _predict(self, vocab_id):
        predicted = self.predicted
        with torch.inference_mode():
            pred_word = self.actions[vocab_id]
            if pred_word in self.gen_stop_toks or len(pred_word) <1 or len(self.predicted) > self.env_max_length:
                
                return predicted, True, self.tokenizer.convert_tokens_to_string(predicted)

            else:
                predicted += [pred_word]

                return predicted, False, self.tokenizer.convert_tokens_to_string(predicted)

    def get_reward(self, input_item, predicted_list, finish):
        reward = 1
        return reward

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = np.argmax(action)
        
        predicted, finish, predicted_str = self._predict(vocab_id=action)
        reward = self.get_reward(self.input_item, predicted, finish)
        self.predicted = predicted
        print(predicted_str)
        info = {"predicted_str": predicted_str} 

        return self._get_obs(predicted), reward, finish, info
        


    
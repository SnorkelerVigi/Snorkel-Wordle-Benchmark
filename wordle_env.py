# Run this command to set the environment variable first
# export PYENCHANT_LIBRARY_PATH=/opt/homebrew/lib/libenchant-2.dylib
import ast
import os
import sys
sys.path.append('/home/ubuntu/')
import copy
from tqdm import tqdm
from typing import List, Optional, Any
from type import Word, Trajectory
from Data.FiendishWords import FIENDISH_WORDS
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from jinja2 import Environment, PackageLoader, select_autoescape
import json
from pathlib import Path
from datasets import Dataset
from enum import Enum
import yaml
# import nltk
# nltk.download('words')
from nltk.corpus import words

import enchant
import re
from tqdm import tqdm
import litellm
from litellm import completion, batch_completion
# litellm._turn_on_debug()
import hashlib
from dotenv import load_dotenv
import requests
from openai import OpenAI
from datetime import datetime

# Replace with your vLLM server URL
VLLM_SERVER_URL = "http://localhost:8000"

load_dotenv()

for key, value in os.environ.items():
    if key.endswith('_API_KEY'):
        os.environ[key] = value

class DummyChoice:
    def __init__(self, c):
        self.message = c["message"]
        self.finish_reason = c.get("finish_reason") or c.get("native_finish_reason")
        self.index = c.get("index", 0)

class DummyResponse:
    def __init__(self, raw):
        self.id = raw.get("id")
        self.model = raw.get("model")
        self.choices = [DummyChoice(c) for c in raw.get("choices", [])]
        self.usage = raw.get("usage")  # dict with prompt_tokens, completion_tokens, total_tokens, cost


class InferenceSource(Enum):
    LITELLM = "litellm"
    OPENROUTER = "openrouter"

def append_result_to_json(result_data: dict, results_file_path: str):
    """
    Append a new result to the results.json file as a list entry.
    If the file doesn't exist, create it with the first entry.
    If it exists, load the list and append the new result.
    """
    results_path = Path(results_file_path)
    
    # Ensure parent directory exists
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results or start with empty list
    results_list = []
    if results_path.exists() and results_path.stat().st_size > 0:
        try:
            with results_path.open('r', encoding='utf-8') as f:
                existing_data = json.load(f)
                # Handle case where file contains a single dict instead of list
                if isinstance(existing_data, list):
                    results_list = existing_data
                else:
                    results_list = [existing_data]  # Convert single result to list
        except json.JSONDecodeError:
            # File exists but has invalid JSON, start fresh
            results_list = []
    
    # Add timestamp to the result
    result_data['timestamp'] = datetime.now().isoformat()
    
    # Append new result
    results_list.append(result_data)
    
    # Write back to file
    with results_path.open('w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2)


def choose_inference_engine():
    """
    Allow user to choose between OpenRouter and LiteLLM inference engines.
    """
    print("\nChoose inference engine:")
    print("1. OpenRouter")
    print("2. LiteLLM")
    
    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == '1':
                return InferenceSource.OPENROUTER
            elif choice == '2':
                return InferenceSource.LITELLM
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}. Please try again.")

def get_model_id():
    try:
        response = requests.get(f"{VLLM_SERVER_URL}/v1/models")
        response.raise_for_status()
        models = response.json().get("data", [])
        if not models:
            print("No models are currently loaded.")
        else:
            print("Loaded models:")
            for model in models:
                print(f"- ID: {model.get('id')} | Owned by: {model.get('owned_by', 'unknown')}")
            return models[0].get('id')
    except requests.RequestException as e:
        print(f"Error querying vLLM server: {e}")

def hash_word(s: str, bits=16):
    hash_obj = hashlib.sha256(s.encode('utf-8'))
    return int(hash_obj.hexdigest(), 16) % (2 ** bits)

def parser(text):
    reasoning = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    guess = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    
    if reasoning and guess:
        return reasoning.group(1).strip(), guess.group(1).strip()
    elif not reasoning and guess:
        return None, guess.group(1).strip()
    elif reasoning and not guess:
        return reasoning.group(1).strip(), None
    else:
        return None, None

class GameState:
    def __init__(self, W: Word, messages: List[dict], model_config: dict = None):
        
        if model_config is None:
            self.model = 'gpt-4o'
            self.custom_llm_provider = 'openai'
            self.api_base = "https://api.openai.com/v1"
            self.inference_source = InferenceSource.LITELLM
        else:
            self.model = model_config['model']
            self.custom_llm_provider = model_config.get('custom_llm_provider', 'openai')
            self.api_base = model_config.get('api_base', "https://api.openai.com/v1")
            self.inference_source = InferenceSource(model_config.get('inference_source', 'litellm'))
            
        # Initialize OpenRouter client if needed
        if self.inference_source == InferenceSource.OPENROUTER:
            import httpx
            # Create a custom httpx client with SSL configuration
            httpx_client = httpx.Client(
                timeout=httpx.Timeout(60.0),  # 60 second timeout
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                verify=True  # Keep SSL verification enabled for security
            )
            
            self.openrouter_client = OpenAI(
                api_key=os.getenv('OPENROUTER_API_KEY'),
                base_url='https://openrouter.ai/api/v1',
                http_client=httpx_client,
                default_headers={
                    "HTTP-Referer": "https://github.com/wordle-game",
                    "X-Title": "Wordle AI Game"
                }
            )
        self.W = Word(**W)
        self.trajectory = Trajectory(
                                    word=self.W.word,
                                    word_hash=self.W.hash,
                                    messages=messages,
                                    completed_by=self.model,
                                    completion_cost=0.0,
                                    solved=False
                                )
        self.max_turns = 6
        self.current_turn = 0
        self.current_guess = None

        broker = enchant.Broker()
        self.d = broker.request_dict("en_US")
        self.local_dictionary = list(json.load(open('/Users/vigneshramesh/Desktop/Wordle/Data/English Words Dictionary.json')).keys())
        
        
        self.nltk_words = words.words()

    def get_feedback(self, guess: str):
        """
        trajectory: list of tuples (word, result)
        word: string
        guess: string
        """
        # if not self.d.check(guess) and guess not in self.nltk_words:
        if guess not in self.nltk_words and guess not in self.local_dictionary and not self.d.check(guess):
            return "Invalid word, not a valid English word"
        
        if len(guess) != 5:
            return "Invalid word, not a 5 letter word"
        
        response = ""
        response_text = ""
        for i, alphabet in enumerate(guess):
            if self.W.word[i] == alphabet:
                response += "G"
                response_text += f'{alphabet} is in the word and in the correct position.\n'
            elif alphabet in self.W.word:
                response += "Y"
                response_text += f'{alphabet} is in the word but in the wrong position.\n'
            else:
                response += "B"
                response_text += f'{alphabet} is not in the word.\n'
        return f"{guess} -> {response}"
    
    def generate_response(self):
        if self.inference_source == InferenceSource.LITELLM:
            # Use LiteLLM - it will automatically detect the provider based on model name and available API keys
            try:
                if self.custom_llm_provider == 'vllm':
                    # For vLLM, use api_base
                    response = completion(
                        model=self.model, 
                        api_base=self.api_base, 
                        messages=self.trajectory.messages
                    )
                else:
                    # For other providers, LiteLLM will auto-detect based on model name and API keys
                    response = completion(
                        model=self.model, 
                        messages=self.trajectory.messages
                    )
                
                # Handle cost tracking
                if hasattr(response, '_hidden_params') and response._hidden_params.get("response_cost") is not None:
                    self.trajectory.completion_cost += response._hidden_params["response_cost"]
                
                # Convert LiteLLM response to expected format
                response_dict = {
                    'choices': [{
                        'message': {
                            'content': response.choices[0].message.content
                        }
                    }]
                }
                return response_dict
                
            except Exception as e:
                print(f"LiteLLM inference failed: {e}")
                raise
                
        elif self.inference_source == InferenceSource.OPENROUTER:
            # Use OpenRouter with the OpenAI client for better SSL handling
            try:
                response = self.openrouter_client.chat.completions.create(
                    model=self.model,
                    messages=self.trajectory.messages
                )
                
                # Convert OpenAI response to expected dict format
                response = {
                    "choices": [{
                        "message": {
                            "content": response.choices[0].message.content
                        }
                    }],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0
                    }
                }
                
                # Handle usage tracking if available
                if 'usage' in response:
                    usage = response['usage']
                    if hasattr(self.trajectory, 'total_tokens'):
                        self.trajectory.total_tokens += usage.get('total_tokens', 0)
                    if hasattr(self.trajectory, 'prompt_tokens'):
                        self.trajectory.prompt_tokens += usage.get('prompt_tokens', 0)  
                    if hasattr(self.trajectory, 'completion_tokens'):
                        self.trajectory.completion_tokens += usage.get('completion_tokens', 0)
                    if hasattr(self.trajectory, 'completion_cost'):
                        self.trajectory.completion_cost += usage.get('cost', 0)
                        
            except Exception as e:
                print(f"OpenRouter API request failed: {e}")
                raise
            
        else:
            raise ValueError(f"Invalid inference source: {self.inference_source}")
        
        return response
    
    def step(self):
        try:
            data = self.generate_response()
            
            # Debug: Print response structure if there are issues
            if 'choices' not in data:
                print(f"Debug - Response missing 'choices': {data}")
                raise KeyError("Response missing 'choices' key")
                
            if not data['choices']:
                print(f"Debug - Empty choices array: {data}")
                raise IndexError("Empty choices array in response")
                
            if 'message' not in data['choices'][0]:
                print(f"Debug - Missing message in choice: {data['choices'][0]}")
                raise KeyError("Missing 'message' in choices[0]")
                
            answer = data['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"Error in step() method: {e}")
            print(f"Response data: {data if 'data' in locals() else 'No response data'}")
            raise

        reasoning, guess = parser(answer)
        self.trajectory.guesses.append(guess)

        self.trajectory.messages.append({'role': 'assistant', 'content': f'<think>{reasoning}</think><answer>{guess}</answer>'})
        
        feedback = None
        if guess:
            feedback = self.get_feedback(guess)
            self.trajectory.feedback.append(feedback)
            self.trajectory.messages.append({'role': 'user', 'content': feedback})
        else:
            self.trajectory.messages.append({'role': 'user', 'content': f'<think>Invalid format, stick to the <think>, </think> and <answer>, </answer> tags provided in the system prompt'})
            self.trajectory.feedback.append('INVALID')
        
        # If the feedback is 'GGGGG', break
        if feedback is not None and feedback.split('->')[-1].strip() == 'GGGGG':
            self.trajectory.solved = True
            self.trajectory.game_completed = True
            self.trajectory.messages[-1]['content'] = f'Success! You found the word {self.W.word} in {self.current_turn+1} turns.'
            return True
        
        return False
    
    def solve(self):
        while self.current_turn < self.max_turns and not self.trajectory.solved:
            self.step()
            self.current_turn += 1
        self.trajectory.game_completed = True
        self.trajectory.num_turns = self.current_turn
        return self.trajectory

class WordleEnv:
    def __init__(self, number_of_games: Optional[int] = None, model_config: dict = None):
        self.num_workers = 5
        self.number_of_games = len(FIENDISH_WORDS) if number_of_games is None else number_of_games
        
        self.dataset = self.get_dataset(number_of_games=self.number_of_games)
        self.model_config = model_config
        self.games_won = 0
        self.first_guesses = {}
        self.invalid_guesses = 0
        self.total_cost = 0.0
        self.lock = threading.Lock()  # Lock for synchronized writing
        self.jinja_env = Environment(
            loader=PackageLoader('Jinja'),
            autoescape=select_autoescape()
        )
        

        self.trajectory_output_dir = Path('/Users/vigneshramesh/Desktop/Wordle/Results')
        self.filename = 'trajectories.json'
        self.trajectory_output_path = self.trajectory_output_dir / self.filename
        self.reset()
        
        # Ensure the output directory exists
        self.trajectory_output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_dataset(self, number_of_games: Optional[int] = None):
        word_dicts = [word.model_dump() for word in FIENDISH_WORDS][-number_of_games:]
        return Dataset.from_list(word_dicts)
    
    def reset(self):
        self.system_prompt_template = self.jinja_env.get_template('system_prompt.txt')
        self.messages_template = [
                                    {'role': 'system', 'content': self.system_prompt_template.render()},
                                    {'role': 'user', 'content': 'Your first guess is?'}
                                ]

    def play(self):
        # Force load the nltk words
        _ = words.words()
        def play_game(i):
        # for i in range(self.number_of_games):

            messages = copy.deepcopy(self.messages_template)
            G = GameState(W=copy.deepcopy(self.dataset[i]), messages=messages, model_config=self.model_config)
            G.solve()
            
            
            # Append to file safely
            with self.lock:
                self.total_cost += G.trajectory.completion_cost
                for feedback in G.trajectory.feedback:
                    if feedback == 'INVALID':
                        self.invalid_guesses += 1
                if G.trajectory.guesses[0] not in self.first_guesses.keys():
                    self.first_guesses[G.trajectory.guesses[0]] = 1
                else:
                    self.first_guesses[G.trajectory.guesses[0]] += 1
                if G.trajectory.solved:
                    self.games_won += 1

                trajectories: list[dict] = []
                if self.trajectory_output_path.exists() and self.trajectory_output_path.stat().st_size:
                    try:
                        with self.trajectory_output_path.open("r", encoding="utf-8") as f:
                            trajectories = json.load(f)          # expect list-or-dict; default to []
                            if not isinstance(trajectories, list):
                                trajectories = [trajectories]
                    except json.JSONDecodeError:
                        trajectories = []


                trajectories.append(G.trajectory.model_dump())
                
                with self.trajectory_output_path.open("w", encoding="utf-8") as f:
                    json.dump(trajectories, f, indent=2)
                                

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(play_game, i): i for i in range(self.number_of_games)}
            for future in tqdm(as_completed(futures), total=self.number_of_games, desc="Solving games"):
                i = futures[future]
                try:
                    future.result()
                except Exception as e:
                    tqdm.write(f"Error in game {i}: {e}")
        
        tqdm.write(f'Total games played: {self.number_of_games}')
        tqdm.write(f'Games won: {self.games_won}')
        tqdm.write(f'Total cost: {self.total_cost}')
        return {
            'number_of_games': self.number_of_games,
            'games_won': self.games_won,
            'total_cost': self.total_cost,
            'first_guesses': self.first_guesses,
            'invalid_guesses_per_game': self.invalid_guesses / self.number_of_games
        }
        
if __name__ == '__main__':
    
    number_of_games = 100
    
    # Load models from YAML file
    models_file = '/Users/vigneshramesh/Desktop/Wordle/Data/models.yaml'
    with open(models_file, 'r') as f:
        all_models = [line.strip() for line in f if line.strip()]
    
    print(f"Testing {len(all_models)} models...")
    results_file_path = '/Users/vigneshramesh/Desktop/Wordle/Results/results.json'
    
    # Let user choose inference engine
    chosen_engine = choose_inference_engine()
    print(f"\nUsing {chosen_engine.value} for inference\n")
    
    for i, model in enumerate(tqdm(all_models)):
        tqdm.write(f"Testing model {model} {i+1}/{len(all_models)}")
        model_config = {
            'model': model,
            'inference_source': chosen_engine.value
        }
        
        try:
            env = WordleEnv(number_of_games=number_of_games, model_config=model_config)
            results = env.play()
            
            # Add metadata to results
            results['model'] = model_config['model']
            results['inference_source'] = model_config['inference_source']
            if 'custom_llm_provider' in model_config:
                results['LLM Provider'] = model_config['custom_llm_provider']
            
            tqdm.write(f"✓ Completed - Win rate: {results['games_won']}/{results['number_of_games']}")
            append_result_to_json(results, results_file_path)
            
        except Exception as e:
            tqdm.write(f"✗ Failed - Error: {str(e)}")
            # Log failure
            failure_result = {
                'model': model,
                'inference_source': chosen_engine.value,
                'number_of_games': number_of_games,
                'games_won': 0,
                'total_cost': 0.0,
                'error': str(e),
                'status': 'failed'
            }
        
            append_result_to_json(failure_result, results_file_path)
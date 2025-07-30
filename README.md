# Wordle AI Benchmark

A comprehensive benchmarking framework for evaluating Large Language Models (LLMs) on Wordle. This tests various AI models' ability to solve challenging Wordle puzzles using strategic reasoning and iterative feedback.

## 🎯 Features

- **Multi-Provider Support**: Works with OpenRouter, LiteLLM, and local vLLM servers
- **Comprehensive Testing**: Evaluate models on 100 carefully selected "fiendish" words
- **Detailed Analytics**: Track success rates, token usage, costs, and solving strategies
- **Concurrent Processing**: Multi-threaded execution for efficient batch testing
- **Rich Output**: JSON trajectories with complete reasoning chains and game states

## 🎯 Example Game Progression

Mode : claude-opus-4-20250514
Cost of this completion : $0.23
Solved in : 4 Turns

**Turn 1**
![Turn 1](Assets/Images/Image%201.png)

**Turn 2** 
![Turn 2](Assets/Images/Image%202.png)

**Turn 3**
![Turn 3](Assets/Images/Image%203.png)

**Turn 4**
![Turn 4](Assets/Images/Image%204.png)


## 📋 Requirements

- Python ≥ 3.12
- API keys for your chosen LLM providers
- PyEnchant library with English dictionary support

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/SnorkelerVigi/Snorkel-Wordle-Benchmark.git
cd Snorkel-Wordle-Benchmark

# Install dependencies using uv (recommended)
pip install uv
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root:

```bash
# Required: At least one API key
OPENROUTER_API_KEY=your_openrouter_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here
GEMINI_API_KEY=your_gemini_key_here

# Optional: For local vLLM server
VLLM_SERVER_URL=http://localhost:8000
```

### 3. Configure Models

Edit `Data/models.yaml` to specify which models to test:

```yaml
# Example models
gpt-4o
claude-3-5-sonnet-20241022
moonshotai/kimi-k2
meta-llama/llama-3.1-8b-instruct
```

### 4. System Dependencies

**macOS with Homebrew:**
```bash
brew install enchant
export PYENCHANT_LIBRARY_PATH=/opt/homebrew/lib/libenchant-2.dylib
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libenchant-2-2 libenchant-2-dev
```

**Windows:**
```bash
pip install pyenchant
```

### 5. Run Benchmark

```bash
python wordle_env.py
```

The script will:
1. Ask you to choose between OpenRouter or LiteLLM inference
2. Run tests on all models specified in `Data/models.yaml`
3. Save results to `Results/results.json` and trajectories to `Results/trajectories.json`

## 📁 Project Structure

```
wordle-ai-benchmark/
├── wordle_env.py           # Main benchmark script
├── type.py                 # Pydantic models for data structures
├── Assets/
│   └── Images/             # Documentation images
│       ├── Image 1.png
│       ├── Image 2.png
│       ├── Image 3.png
│       └── Image 4.png
├── Data/
│   ├── __init__.py         # Package initialization
│   ├── FiendishWords.py    # Curated list of challenging words
│   ├── data.py             # Complete word database (13K+ words)
│   ├── models.yaml         # Models to test
│   ├── all_models.yaml     # Extended model list
│   └── English Words Dictionary.json
├── Jinja/
│   └── templates/
│       └── system_prompt.txt  # System prompt template
├── Results/
│   ├── results.json        # Aggregated test results
│   └── trajectories.json   # Detailed game trajectories
├── pyproject.toml          # Python dependencies
└── README.md
```

## ⚙️ Configuration

### Model Configuration

The benchmark supports multiple inference sources:

**OpenRouter (Recommended)**
- Supports 200+ models from various providers
- Built-in cost tracking and usage analytics
- Reliable API with high uptime

**LiteLLM**
- Auto-detects provider based on model name and API keys
- Supports local deployments and custom endpoints
- Flexible routing and fallback options

**vLLM (Local)**
- For self-hosted model inference
- Configure `VLLM_SERVER_URL` in your `.env` file

### Customizing Test Parameters

Edit `wordle_env.py` to modify:

```python
# Number of games to run per model (default: 100)
number_of_games = 50

# Number of concurrent workers (default: 5)
self.num_workers = 10

# Maximum turns per game (default: 6)
self.max_turns = 6
```

### Adding Custom Word Lists

To test with different word sets:

1. Create your word list following the `Word` model structure in the `Data/` folder
2. Import in `wordle_env.py`:
   ```python
   from Data.your_module import YOUR_WORDS
   self.dataset = self.get_dataset_from_words(YOUR_WORDS)
   ```

## 📊 Output Format

### Results Summary (`Results/results.json`)

```json
{
  "model": "gpt-4o",
  "inference_source": "litellm",
  "number_of_games": 100,
  "games_won": 87,
  "total_cost": 0.0234,
  "first_guesses": {
    "slate": 23,
    "arose": 18,
    "audio": 15
  },
  "invalid_guesses_per_game": 0.02,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Game Trajectories (`Results/trajectories.json`)

```json
{
  "word": "eclat",
  "word_hash": 57956,
  "solved": true,
  "num_turns": 4,
  "completion_cost": 0.00123,
  "completed_by": "gpt-4o",
  "guesses": ["slate", "round", "chime", "eclat"],
  "feedback": ["BYBBB", "BBBYB", "YYBBB", "GGGGG"],
  "messages": [
    {
      "role": "system",
      "content": "You are a Wordle solver assistant..."
    },
    {
      "role": "assistant", 
      "content": "<think>I'll start with SLATE...</think><answer>slate</answer>"
    }
  ]
}
```

## 🔧 Troubleshooting

### Common Issues

**"Error in game X: 'choices'"**
- This indicates an API response format issue
- Check your API keys are valid and have sufficient quota
- Try switching inference sources (OpenRouter ↔ LiteLLM)

**PyEnchant Installation Issues**
```bash
# macOS
brew install enchant
export PYENCHANT_LIBRARY_PATH=/opt/homebrew/lib/libenchant-2.dylib

# Ubuntu
sudo apt-get install libenchant-2-dev

# If still having issues, try:
pip uninstall pyenchant
pip install --no-cache-dir pyenchant
```

**Rate Limiting**
- Reduce `num_workers` in `WordleEnv.__init__()`
- Add delays between requests in provider-specific code
- Check your API plan limits

**Memory Issues with Large Models**
- Reduce `number_of_games` parameter
- Process models sequentially instead of in parallel
- Clear trajectories periodically during long runs

### Debug Mode

Enable detailed logging:

```python
# In wordle_env.py, uncomment:
# litellm._turn_on_debug()

# Add this for more verbose output:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
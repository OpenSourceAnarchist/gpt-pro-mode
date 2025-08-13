# OpenRouter Pro Mode

This project, originally `gpt-pro-mode`, has been refactored to exclusively use the [OpenRouter API](https://openrouter.ai "null"). It provides a powerful "Pro Mode" for interacting with any language model available on OpenRouter.

The core idea is to leverage an ensemble method to achieve higher-quality, more robust, and more creative responses from LLMs. Instead of a single generation, this tool:

1.  **Fans out** multiple parallel requests to a model of your choice for initial "candidate" answers.
    
2.  **Synthesizes** these candidates using a final call to a (potentially more powerful) model, instructing it to act as an expert editor to merge strengths, correct errors, and produce a single, superior response.
    

This project now includes "Tournament Mode": if the number of generations is greater than 20, it will generate candidates, synthesize them in groups, and then run a final synthesis pass on the winners, enabling massive-scale ensembling.

## Features

-   **Unified API**: Access any model on OpenRouter (e.g., GPT-4o, Claude 3 Opus, Llama 3 70B) through a single interface.
    
-   **Configurable Models**: Use environment variables to easily set different models for candidate generation and final synthesis.
    
-   **Ensemble Generation**: Improve response quality through multi-candidate synthesis.
    
-   **Tournament Mode**: Scale the ensemble method for even higher-quality results on complex prompts.
    
-   **FastAPI Endpoint**: Exposes the functionality as a clean, efficient API service.
    

## Setup

1.  **Clone the repository:**
    
    ```
    git clone <repository_url>
    cd <repository_name>
    
    ```
    
2.  **Install dependencies:**
    
    ```
    pip install -r requirements.txt
    
    ```
    
3.  Configure your environment:
    
    Create a .env file in the root directory by copying the example:
    
    ```
    cp .env.example .env
    
    ```
    
    Now, edit the `.env` file and add your OpenRouter API key and any other custom settings.
    
    ```
    # Required: Your OpenRouter API Key
    OPENROUTER_API_KEY="sk-or-..."
    
    # Optional: Customize the models used
    # Model for generating initial candidates
    GENERATION_MODEL="anthropic/claude-3-haiku-20240307"
    # Model for synthesizing the final answer
    SYNTHESIS_MODEL="openai/gpt-4o"
    
    # Optional: App identity for OpenRouter leaderboards
    # See: https://openrouter.ai/docs/features/app-attribution
    YOUR_SITE_URL="http://localhost:8000"
    YOUR_SITE_NAME="OpenRouter Pro Mode"
    
    ```
    

## Usage

### Run the API Server

```
uvicorn main:app --host 0.0.0.0 --port 8000

```

### Example Request

You can now send requests to the `/pro-mode` endpoint.

```
curl -X POST http://localhost:8000/pro-mode \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain self-play in RL with a concrete example.","num_gens":5}'

```

The response will be a JSON object containing the `final` synthesized answer and a list of all the original `candidates`.

```
{
  "final": "Self-play is a technique in reinforcement learning where an agent learns by playing against itself... For instance, in the game of Go, DeepMind's AlphaGo Zero started with no human data...",
  "candidates": [
    "Candidate answer 1...",
    "Candidate answer 2...",
    "Candidate answer 3...",
    "Candidate answer 4...",
    "Candidate answer 5..."
  ]
}

```

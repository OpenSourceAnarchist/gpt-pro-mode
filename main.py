import os
import time
from typing import List, Tuple, Dict, Any
import concurrent.futures as cf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

# Load from environment variables with defaults
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "deepseek/deepseek-chat-v3-0324:free")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "deepseek/deepseek-r1-0528:free")
SITE_URL = os.getenv("YOUR_SITE_URL")
SITE_NAME = os.getenv("YOUR_SITE_NAME")

# --- Constants ---
# Load numeric constants from environment, with defaults, and cast to int
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", 8192))
TOURNAMENT_THRESHOLD = int(os.getenv("TOURNAMENT_THRESHOLD", 20))
GROUP_SIZE = int(os.getenv("GROUP_SIZE", 10))

# Hardcoded constants
MAX_WORKERS = 100
MAX_GENS = 100

# --- FastAPI App Initialization ---
app = FastAPI(
    title="OpenRouter Pro Mode",
    description="Uses ensemble methods via OpenRouter for superior model responses."
)

# --- OpenAI Client for OpenRouter ---
# We initialize it once and reuse it.
# The API key is checked at startup.
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

# --- FIX: Build headers dictionary defensively ---
# This prevents errors if SITE_URL or SITE_NAME are None or empty.
default_headers = {}
if SITE_URL:
    default_headers["HTTP-Referer"] = SITE_URL
if SITE_NAME:
    default_headers["X-Title"] = SITE_NAME

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers=default_headers,
)

# --- Pydantic Schemas ---
class ProModeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user prompt to generate responses for.")
    num_gens: int = Field(..., ge=1, le=MAX_GENS, description="Number of candidate generations.")

class ProModeResponse(BaseModel):
    final: str
    candidates: List[str]

# --- Core Logic ---
def _one_completion(prompt: str, temperature: float, model: str) -> str:
    """
    Executes a single chat completion call to OpenRouter with retry logic.
    """
    delay = 0.5
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
                top_p=1,
                stream=False,
            )
            # Add defensive check for the response structure
            if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                return resp.choices[0].message.content
            else:
                # The response was successful but empty, return empty string.
                return ""
        except OpenAIError as e:
            if attempt == 2:
                # On the last attempt, re-raise the exception
                raise e
            time.sleep(delay)
            delay *= 2
    return "" # Should not be reached, but linters appreciate it

def _build_synthesis_messages(candidates: List[str]) -> List[Dict[str, str]]:
    """
    Constructs the system and user messages for the synthesis step.
    """
    numbered_candidates = "\n\n".join(
        f"<candidate number={i+1}>\n{text}\n</candidate>"
        for i, text in enumerate(candidates)
    )
    system_prompt = (
        "You are a world-class editor and synthesizer of information. Your task is to analyze "
        "multiple AI-generated candidate answers to a prompt. Your goal is to produce a single, "
        "definitive, and superior response. Merge the strengths of all candidates, correct any "
        "errors or inconsistencies, eliminate repetition, and present the final answer clearly "
        "and decisively. Do not mention the candidate process in your final output."
    )
    user_prompt = (
        f"You are given {len(candidates)} candidate answers below, delimited by XML tags.\n\n"
        f"{numbered_candidates}\n\n"
        "Synthesize these into the single best final answer."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def _synthesize(candidates: List[str]) -> str:
    """
    Performs the synthesis step, calling the synthesis model.
    """
    messages = _build_synthesis_messages(candidates)
    resp = client.chat.completions.create(
        model=SYNTHESIS_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=MAX_OUTPUT_TOKENS,
        top_p=1,
        stream=False,
    )
    # Add defensive check for the synthesis response
    if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
        return resp.choices[0].message.content
    return "Synthesis failed to generate content." # Return a specific error message

def _fanout_candidates(prompt: str, n_runs: int) -> List[str]:
    """
    Generates `n_runs` candidates in parallel using a thread pool.
    """
    num_workers = min(n_runs, MAX_WORKERS)
    results: List[str] = [""] * n_runs
    with cf.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map each future to its index to preserve order
        future_to_index = {
            executor.submit(_one_completion, prompt, 0.9, GENERATION_MODEL): i
            for i in range(n_runs)
        }
        for future in cf.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                # Log the error but continue, allowing other threads to finish
                print(f"Candidate generation for index {index} failed: {e}")
                results[index] = f"Error: {e}" # Mark failure
    return results

def _pro_mode_simple(prompt: str, n_runs: int) -> ProModeResponse:
    """
    Handles the standard pro mode: generate all candidates, then synthesize once.
    """
    candidates = _fanout_candidates(prompt, n_runs)
    filtered_candidates = [c for c in candidates if c and not c.startswith("Error:")]
    if not filtered_candidates:
        raise HTTPException(status_code=503, detail="All candidate generations failed.")
    
    final_text = _synthesize(filtered_candidates)
    return ProModeResponse(final=final_text, candidates=candidates)

def _pro_mode_tournament(prompt: str, n_runs: int) -> ProModeResponse:
    """
    Handles tournament mode: generate, synthesize in groups, then a final synthesis.
    """
    # Round 1: Generate all candidates
    all_candidates = _fanout_candidates(prompt, n_runs)
    filtered_candidates = [c for c in all_candidates if c and not c.startswith("Error:")]
    if not filtered_candidates:
        raise HTTPException(status_code=503, detail="All candidate generations failed.")

    # Group into chunks for the first synthesis round
    groups = [filtered_candidates[i:i+GROUP_SIZE] for i in range(0, len(filtered_candidates), GROUP_SIZE)]
    
    # Synthesize each group in parallel
    group_winners: List[str] = []
    with cf.ThreadPoolExecutor(max_workers=min(len(groups), MAX_WORKERS)) as executor:
        future_to_group = {executor.submit(_synthesize, group): group for group in groups}
        for future in cf.as_completed(future_to_group):
            try:
                group_winners.append(future.result())
            except Exception as e:
                print(f"A synthesis group failed: {e}")

    if not group_winners:
        raise HTTPException(status_code=503, detail="All synthesis groups failed.")

    # Final Round: Synthesize the winners of each group
    final_text = _synthesize(group_winners) if len(group_winners) > 1 else group_winners[0]
    return ProModeResponse(final=final_text, candidates=all_candidates)

# --- API Endpoint ---
@app.post("/pro-mode", response_model=ProModeResponse)
def pro_mode_endpoint(body: ProModeRequest):
    """
    The main API endpoint for running Pro Mode.
    It decides whether to run a simple or tournament-style process.
    """
    try:
        if body.num_gens > TOURNAMENT_THRESHOLD:
            return _pro_mode_tournament(prompt=body.prompt, n_runs=body.num_gens)
        else:
            return _pro_mode_simple(prompt=body.prompt, n_runs=body.num_gens)
    except OpenAIError as e:
        raise HTTPException(status_code=e.status_code or 502, detail=f"An API error occurred: {e.body.get('message', str(e))}")
    except HTTPException:
        # Re-raise known HTTP exceptions
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected internal error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Use 'main:app' string for uvicorn to enable clean reloading
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True # Recommended for development
    )

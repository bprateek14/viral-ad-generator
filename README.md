# viral-ad-generator

## Introduction
**viral-ad-generator** is an automated pipeline for creating viral video ads from trending tweets. It leverages state-of-the-art Large Language Models (LLMs) and Veo-3 video generation to transform social media trends into engaging, multi-scene video content. The system is designed for marketers, content creators, and researchers interested in rapid prototyping of viral video ads.

## Features
- **Viral Tweet Collection:** Fetches trending tweets using advanced search and filters for high engagement.
- **Tweet Classification:** Uses LLMs to identify tweets with meaningful, ad-worthy content.
- **Script Generation:** Converts tweets into multi-scene, cinematic video scripts.
- **Video Production:** Generates videos for each scene using Veo-3 and stitches them into a final ad.
- **Frame Extraction:** Captures key frames for thumbnails or further analysis.
- **End-to-End Automation:** Runs as a notebook or Python scripts for full workflow automation.

## Architecture
1. **Data Collection:** Fetches viral tweets via Twitter API.
2. **Classification:** Filters tweets using LLM-based prompt classification.
3. **Script Generation:** Produces multi-scene scripts with prompt engineering and LLMs.
4. **Video Generation:** Uses Google GenAI (Veo-3) to create scene videos.
5. **Stitching & Export:** Combines scenes, extracts frames, and saves outputs.

```
+-------------------+      +-------------------+      +-------------------+      +-------------------+
|  Viral Tweet API  | ---> |   LLM Classifier  | ---> |  Script Generator | ---> |   Video Generator |
+-------------------+      +-------------------+      +-------------------+      +-------------------+
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/viral-ad-generator.git
cd viral-ad-generator
```

### 2. Install dependencies

You can use [uv](https://github.com/astral-sh/uv) for fast dependency management with `pyproject.toml`:

```bash
uv venv
uv pip install -r requirements.txt
# or, to use pyproject.toml directly:
uv pip install -e .
```

Alternatively, with standard pip:

```bash
pip install -r requirements.txt
# or
pip install -e .
```

### 3. Set up environment variables
Create a `.env` file in the project root:
```
X_API_KEY=your_twitter_api_key
GOOGLE_CLOUD_REGION=us-central1
```

### 4. Configure TOML
Edit `config/config.toml` with your model, API, and prompt settings.

### 5. Prepare data folders
Ensure the following folders exist:
- `data/`
- `videos/`
- `frames/`
- `stitched_videos/`

## Usage

### Notebook Workflow
Run the notebook step-by-step:
```
notebooks/viral-ad-gen.ipynb
```
- Fetch and classify tweets
- Generate video scripts
- Produce and stitch videos


## Configuration

- **config/config.toml:** Central config for API keys, model names, and prompt templates.
- **.env:** Secrets and environment variables.
- **requirements.txt:** Python dependencies.

## Troubleshooting

- **API Errors:** Check your API keys and network connectivity.
- **Model Access:** Ensure you have access to Veo-3 and Gemini APIs.
- **GPU Support:** For best performance, use a GPU-enabled environment.
- **File Paths:** Verify all input/output directories exist and are writable.

## Data

- Classified tweets: `data/viral_tweets_classified.csv`
- Scripts: `data/viral_tweets_with_scripts.csv`
- Videos: `videos/`
- Frames: `frames/`
- Stitched videos: `stitched_videos/`


## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI](https://openai.com/)
- [Google GenAI](https://cloud.google.com/genai)
- [MoviePy](https://zulko.github.io/moviepy/)
- [OpenCV](https://opencv.org/)

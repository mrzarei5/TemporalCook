# TemporalCook: Visual Question Answering for Cooking Activities

**This is the official repository for TemporalCook: Benchmarking Temporal and Procedural Reasoning in Multimodal Large Language Models**

> **Paper**: Accepted to [MUCG @ ACM MM 2025](https://mucg-workshop.github.io/) (The 1st International Workshop on MLLM for Unified Comprehension and Generation) 

## 📋 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Experiment Types](#experiment-types)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)

## Overview

TemporalCook is a research project that evaluates how well various Vision-Language Models (VLMs) can understand and reason about temporal sequences in cooking activities. The project includes:

- **Multi-Model Evaluation**: Support for 11 different VLM models
- **RAG Baseline**: Retrieval-Augmented Generation approach for enhanced reasoning
- **Automated Evaluation**: GPT-based evaluation system for answer accuracy

## Getting Started

### 1. Prepare the TemporalCook Dataset

The TemporalCook dataset consists of the following files:

- **TemporalCook.json**: Main dataset with questions and answers
- **recipe_video_ids.json**: Recipe to video ID mapping  
- **images.zip** (~475MB): Cooking activity images organized by recipe type

**Download Instructions:**
1. **JSON files**: Already included in this repository (`dataset/` folder)
2. **Images**: Download from [Google Drive](https://drive.google.com/file/d/1a4WacU9CiesBezMHhU0ODqGREkhWIl_a/view?usp=sharing)
3. **Extract images**: After downloading, extract `images.zip` to the `dataset/` directory

```bash
# Example: After downloading images.zip to dataset/
cd dataset/
unzip images.zip
# This will create the images/ folder with all cooking activity images
```

### 2. Prepare YouCook2 Data for vqa_rag Experiments

For experiments using the `vqa_rag` mode, you must first extract transcripts from the YouCook2 video knowledge source:

1. **Download YouCook2 Data**
   - Download the raw videos and the Train+Val annotation JSON file from the [YouCook2 download page](http://youcook2.eecs.umich.edu/download).
   - Extract all downloaded archive files to a single directory (e.g., `/path/to/youcook2/`).

2. **Extract Transcripts**
   - Use the provided `transcriptsExtract.py` script to extract transcripts from the YouCook2 videos and annotations into the `dataset/` folder.
   - Command:
     ```bash
     python transcriptsExtract.py --youcook_root /path/to/youcook2/
     ```
   - The `--youcook_root` argument should point to the directory where you extracted the YouCook2 files.

## Project Structure

```
TemporalCook/
├── dataset/                     # Dataset files (see above for preparation)
│   ├── TemporalCook.json       # Main dataset with questions and answers
│   ├── recipe_transcript_dic.json  # Recipe transcripts for RAG
│   ├── images/                 # Cooking activity images
│   └── README.md               # Dataset documentation
├── models/                     # VLM model implementations
│   ├── gpt41.py               # GPT-4.1 implementation
│   ├── gpt41_mini.py          # GPT-4.1 Mini implementation
│   ├── gpt4o.py               # GPT-4o implementation
│   ├── llava.py               # LLaVA implementation
│   ├── gemini25pro.py         # Gemini 2.5 Pro implementation
│   ├── llama4scout17b.py      # Llama 4 Scout 17B implementation
│   ├── llama32b90.py          # Llama 3 2B 90 implementation
│   ├── llama4maverick17b.py   # Llama 4 Maverick 17B implementation
│   ├── pixtral.py             # Pixtral Large implementation
│   ├── qwenvl25.py            # QwenVL 2.5 implementation
│   └── gemma3_27b_it.py       # Gemma 3 27B IT implementation
├── exp_result/                 # Experiment results
├── vqa.py                     # Main VQA evaluation script
├── evaluate_vqa.py            # Answer evaluation script
├── useGPT.py                  # GPT utility for RAG
├── useWhisper.py              # Whisper transcription utility
├── transcriptsExtract.py      # Transcript extraction utility
├── utils.py                   # Utility functions
└── README.md                  # This file
```

## Supported Models

The framework supports evaluation of the following 11 Vision-Language Models:

### OpenAI Models
- **GPT-4.1** (`gpt41`) - Latest GPT-4 model
- **GPT-4.1 Mini** (`gpt41_mini`) - Smaller, faster GPT-4 variant
- **GPT-4o** (`gpt4o`) - GPT-4 Omni model

### Google Models
- **Gemini 2.5 Pro** (`geminipro25`) - Google's latest multimodal model

### Meta Models
- **LLaVA** (`llava`) - Large Language and Vision Assistant
- **Llama 4 Scout 17B** (`llama4scout`) - Llama 4 variant
- **Llama 3 2B 90** (`llama3290b`) - Llama 3 variant
- **Llama 4 Maverick 17B** (`llama4maverick`) - Llama 4 variant

### Other Models
- **Pixtral Large** (`pixtral_large`) - PixArt-based model
- **QwenVL 2.5** (`qwen25`) - Alibaba's vision-language model
- **Gemma 3 27B IT** (`gemma3_27b_it`) - Google's Gemma model

## Installation

1. **Create and activate a conda environment**:
   ```bash
   conda create --name temporalcook python=3.9
   conda activate temporalcook
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with your API keys. The required keys depend on which models you want to use:

   | Model Family     | Models Supported                                     | Required API Key           |
   |------------------|------------------------------------------------------|----------------------------|
   | OpenAI           | gpt41, gpt41_mini, gpt4o                             | OPENAI_API_KEY             |
   | Google           | geminipro25                                          | GOOGLE_API_KEY             |
   | Mistral AI       | Pixtral Large                                        | MISTRALAI_API_KEY          |
   | HuggingFace      | Other models                                         | HF_API_KEY                 |

   Example `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   MISTRALAI_API_KEY=your_mistralai_api_key_here
   HF_API_KEY=your_huggingface_api_key_here
   ```

   Only the keys for the models you intend to use are required.

## Usage

### Running VQA Evaluation

The main evaluation script supports three experiment types:

1. **Standard VQA** (`vqa`): Image + question → answer
2. **Text-only QA** (`qa`): Question only → answer (no image)
3. **VQA with RAG** (`vqa_rag`): Image + question + retrieved context → answer

#### Basic Usage

```bash
python vqa.py --vqa_model gpt41 --exp_type vqa --n_answers 1
```

#### Advanced Usage

```bash
python vqa.py \
    --vqa_model gpt41 \
    --llm_model gpt41_mini \
    --exp_type vqa_rag \
    --n_answers 3 \
    --data_root ./dataset \
    --save_root ./exp_result \
    --device_id 0
```

### Parameters

- `--vqa_model`: Vision-Language Model to evaluate (see supported models above)
- `--llm_model`: Language model for RAG (choices: `gpt4o_mini`, `gpt41_mini`)
- `--exp_type`: Experiment type (`vqa`, `qa`, `vqa_rag`)
- `--n_answers`: Number of answers to generate (default: 1)
- `--data_root`: Path to dataset directory (default: `./dataset`)
- `--save_root`: Path to save results (default: `./exp_result`)
- `--device_id`: GPU device ID (default: 0)
- `--seed`: Random seed (default: 3)

### Evaluating Results

After running VQA evaluation, evaluate the accuracy of generated answers:

```bash
python evaluate_vqa.py \
    --save_root ./exp_result \
    --exp_type vqa \
    --vqa_model gpt41 \
    --evaluator gpt41_mini \
    --api_type_key OPENAI_API_KEY
```

## Dataset

The TemporalCook dataset contains cooking-related VQA questions with the following structure:

- **Questions**: Temporal and procedural reasoning questions about cooking activities
- **Images**: Cooking activity images from various recipe types (organized by recipe folders)
- **Ground Truth**: Expert-annotated answers
- **Supporting Videos**: Instructional videos with relevant procedure steps

Each dataset entry includes:
- Question text and ground truth answer
- Image path and recipe category
- Supporting video information for RAG experiments

**Dataset Organization:**
- Images are organized in recipe-specific folders (e.g., `pizza marghetta/`, `pad thai/`, `sushi/`)
- Each recipe folder contains multiple cooking activity images
- Total dataset size: ~479MB (3.5MB JSON + ~475MB images)

For detailed dataset information, see the [dataset README](dataset/README.md).

## Experiment Types

### 1. Standard VQA
Evaluates how well VLMs can answer questions about cooking images without additional context.

### 2. Text-only QA
Tests language-only reasoning by removing visual input, focusing on the model's cooking knowledge.

### 3. VQA with RAG
Enhances VQA performance by retrieving relevant cooking procedure steps from a knowledge base of instructional videos. **Requires transcript extraction as described above.**

## Results

Results are saved in the `exp_result/` directory with the following structure:

```
exp_result/
└── {vqa_model}_{llm_model}_{exp_type}/
    └── n_answers_{n}/
        ├── results.json              # Raw model outputs
        ├── results_a_{evaluator}.json # Evaluation results
        ├── metrics_{evaluator}.json   # Accuracy metrics
        └── tokens_{evaluator}.json    # Token usage statistics
```

## Evaluation Metrics

The evaluation system uses GPT-based semantic matching to determine answer correctness:

- **Accuracy**: Percentage of semantically correct answers
- **Token Usage**: Input/output token counts for cost analysis
- **Detailed Analysis**: Per-question evaluation with justification

## 📋 Dataset Usage Terms

The TemporalCook dataset is derived from the [YouCookII dataset](https://youcook2.eecs.umich.edu/) and is provided for **non-commercial, research purposes only**. All images originate from YouCookII and are subject to the same usage restrictions.

**Key Terms:**
- ✅ **Permitted**: Academic research, non-commercial use
- ❌ **Not Permitted**: Commercial use, redistribution for profit
- 📚 **Required**: Citation of both our paper and YouCookII dataset

For complete licensing details and citation information, see the [dataset README](dataset/README.md).
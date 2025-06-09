# Visual-Storyteller
Text-to-Visual Storytelling Using LLMs


# Directory Structure

```
Visual-Storyteller/
├── data/                  # Raw datasets
│   ├── personas/
│   ├── booksummaries/
│   ├── tinystories/
│   ├── reddit_stories/
│   ├── poe_stories/
│   ├── guttenberg_stories/
│   ├── sentiment_stories/
│   ├── hippocorpus/
│   ├── scifi_stories/
│   ├── fairytaleqa/
│   ├── writingprompts/
│   └── writingpromptsx/
├── processed_data/        # Cleaned and preprocessed data
│   ├── combined_data.txt  # Combined corpus
│   ├── train.txt          # Training data
│   ├── val.txt            # Validation data
│   ├── test.txt           # Test data
│   ├── instructions.json  # Instruction data for fine-tuning
├── models/                # Trained models
│   ├── llm/               # Text generation LLM
│   │   ├── pre-trained/
│   │   └── fine-tuned/
│   └── vision_llm/        # Vision LLM (pretrained, or fine-tuned)
├── scripts/               # Scripts for data processing, training, etc.
│   ├── data_processing/
│   │   ├── clean_data.py
│   │   ├── combine_data.py
│   │   ├── tokenize.py
│   │   └── split_data.py
│   ├── llm_training/
│   │   ├── pretrain.py
│   │   ├── finetune.py
│   │   └── instruction_finetune.py
│   ├── vision_generation/
│   │   └── generate_images.py
│   ├── utils/             # utility scripts
│   │   └── tts.py         # for Text-to-Speech conversion
├── visual_storyteller/    # Main application code
│   ├── app.py             # Flask/Streamlit app
│   ├── components/        # UI components
│   └── assets/            # Images, CSS, etc.
├── notebooks/             # Jupyter notebooks for experimentation
│   ├── data_exploration.ipynb
│   ├── model_evaluation.ipynb
├── README.md
└── requirements.txt
```


# Datasets

* [personas](http://www.cs.cmu.edu/~ark/personas/)
* [CMU Booksummarues](http://www.cs.cmu.edu/~dbamman/booksummaries.html)
* [Tinystories Narrative Classification](https://www.kaggle.com/api/v1/datasets/download/thedevastator/tinystories-narrative-classification)
* [reddit-short-stories](https://www.kaggle.com/api/v1/datasets/download/trevordu/reddit-short-stories)
* [poe short stories](https://www.kaggle.com/api/v1/datasets/download/leangab/poe-short-stories-corpuscsv)
* [1002 Short Stories (Project Guttenberg)](https://www.kaggle.com/api/v1/datasets/download/shubchat/1002-short-stories-from-project-guttenberg)
* [4000 Stories with Sentiment Analuysis](https://brunel.figshare.com/articles/dataset/4000_stories_with_sentiment_analysis_dataset/7712540?file=14357549)
* [Hippocorpus](https://www.kaggle.com/api/v1/datasets/download/saurabhshahane/hippocorpus)
* [sci-fi short stories](https://www.kaggle.com/api/v1/datasets/download/stealthtechnologies/sci-fi-short-stories)
* [FairyTale QA](https://huggingface.co/datasets/WorkInTheDark/FairytaleQA)
* [Writingprompts](https://huggingface.co/datasets/euclaise/writingprompts)
* [WritingPromptsX](https://huggingface.co/datasets/euclaise/WritingPromptsX)
* [sci-fi TV Shows](https://huggingface.co/datasets/lara-martin/Scifi_TV_Shows)


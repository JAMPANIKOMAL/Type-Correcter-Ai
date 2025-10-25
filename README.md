# Type Correcter Ai

**Course:** Artificial Intelligence (G5AD21AI)  
**Project:** Final AI Project Submission

A deep-learning application that corrects typing errors. This repository includes the source code, a trained model, and a Flask-based web GUI to demo the model.

## Project overview

Build an intelligent system that identifies and corrects common typographical errors using a Sequence-to-Sequence (Seq2Seq) model trained on paired noisy (mistyped) and clean text. Deliverable: a working web GUI where users input text and receive corrected output.

## Project evolution (literature review)

- Prototype 1: [invisible-autocorrect-extension](https://github.com/JAMPANIKOMAL/invisible-autocorrect-extension)  
    Concept: Frequency-based dictionary browser extension. Limitation: Non-contextual; could not fix internal-word errors or partial words.

- Prototype 2: [ghost-type-corrector](https://github.com/JAMPANIKOMAL/Ghost-Type-Corrector)  
    Concept: First Seq2Seq AI attempt; produced `autocorrect_model.h5` and tokenizer config. Limitation: Model existed but GUI integration was incomplete.

This project integrates the trained AI from Prototype 2 into a clean, functional Flask GUI.

## Methodology & techniques

- AI technique: Natural Language Processing (NLP)  
- Model architecture: Seq2Seq with Bidirectional GRU layers  
- Frameworks & libraries:
    - TensorFlow (Keras)
    - Flask
    - pandas, scikit-learn (preprocessing)
- Dataset: Custom corpus (`data/raw_corpus.txt`) used to generate training pairs

## Project structure

```
Type-Correcter-Ai/
├── .gitignore
├── app.py
├── environment.yml
├── environment-gpu.yml
├── requirements.txt
├── README.md
├── LICENSE
│
├── data/
│   ├── .gitkeep            # add raw_corpus.txt, train_*.txt, tokenizer_config.json
│
├── model/
│   ├── .gitkeep            # add autocorrect_model.h5
│
└── src/
        ├── __init__.py
        ├── autocorrect/
        │   ├── __init__.py
        │   └── predictor.py
        │
        ├── training_scripts/
        │   ├── __init__.py
        │   ├── 01_data_preprocessing.py
        │   └── 02_model_training.py
        │
        └── webapp/
                ├── __init__.py
                ├── routes.py
                ├── static/
                │   ├── script.js
                │   └── style.css
                └── templates/
                        └── index.html
```

## Setup and execution

1. Clone the repository
```
git clone https://github.com/JAMPANIKOMAL/Type-Correcter-Ai
cd Type-Correcter-Ai
```

2. Add model and data files
- Place `autocorrect_model.h5` in `model/`
- Place `tokenizer_config.json` in `data/`
- Optionally add `raw_corpus.txt`, `train_clean.txt`, `train_noisy.txt` to `data/`

3. Create the environment (choose one)

Option A — CPU (recommended)
```
conda env create -f environment.yml
conda activate type-correcter-ai-cpu
```

Option B — GPU
```
conda env create -f environment-gpu.yml
conda activate type-correcter-ai-gpu
# Post-install (required)
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
```

Option C — Python venv + pip
```
python -m venv venv
# Activate:
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```
Note: GPU support requires manual NVIDIA driver/CUDA setup.

4. Run the web app
```
python app.py
```
Open the URL shown in the console (default: http://127.0.0.1:5000).

## Optional: Re-train the model

If you add new raw data to `data/raw_corpus.txt`, run:
```
python src/training_scripts/01_data_preprocessing.py
python src/training_scripts/02_model_training.py
```

## Acknowledgements

- Dataset basis: [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/English)  
- Prototyping and debugging assistance: Google Gemini

## License

MIT License — see [LICENSE](./LICENSE) for details.

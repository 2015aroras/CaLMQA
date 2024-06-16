## CaLMQA: Exploring culturally specific long-form question answering across 23 languages

This is the repo for *CaLMQA: Exploring culturally specific long-form question answering across 23 languages*.
CaLMQA is a long-form question answering (LFQA) dataset spanning 23 high- to low-resource languages.

We recommend using the [CaLMQA HF dataset](https://huggingface.co/datasets/shanearora/CaLMQA) to view or use
CaLMQA. This repo focuses on reproducibility of our experiments.

If you find CaLMQA useful, please cite:

```
TBA
```

### Installation

Make a new Python 3.10+ environment using `virtualenv` or `conda`. Then install the `calmqa` package locally using:
```
git clone https://github.com/2015aroras/CaLMQA.git
cd CaLMQA
pip install -e .
```
To run automatic evaluations, you will also need to install optional extra dependencies
using:
```
pip install -e ".[autoeval]"
```

### Authentication for API calls

Some models are prompted via API calls. These API calls require access credentials. Our code can read relevant
credentials from environment variables or from a `.env` file (using [python-dotenv](https://pypi.org/project/python-dotenv/)).
The credentials depend on the type of model:

- [OpenAI](https://platform.openai.com/docs/overview) models (GPT-4 Turbo, GPT-4o): `OPENAI_ORG_ID` and `OPENAI_API_KEY`
- [Claude](https://docs.anthropic.com/en/docs/intro-to-claude) models (Claude Opus): `ANTHROPIC_API_KEY`
- [HF Transformers](https://huggingface.co/docs/transformers/en/index) models (Aya 13B): `HF_USER_ACCESS_TOKEN`
- [Together AI](https://www.together.ai/) models (Mixtral 8x22B, Llama 3 70B): `TOGETHER_API_KEY`
- [Vertex AI](https://cloud.google.com/vertex-ai?hl=en) models (Gemini 1.5 Pro): `GOOGLE_API_KEY`, `PROJECT_ID` and  `REGION`

More information about API authentication can be determined from the documentation of the API's producer.

### Understanding dataset files

For reproducibility purposes, we spread our dataset across multiple 'dataset' files
(in the [data/datasets](https://github.com/2015aroras/CaLMQA/tree/main/data/datasets) directory) and store extra state
information (e.g. model version, temperature) in these files along with the QA data. Each dataset file
contains the entries consisting of:

- A question object. This object contains the question text and any of its translations
along with the state when these translations were produced.
- A list of answer objects. Each answer contains the model that generated the answer (including human) and
the state (`prompting_state`) when the answer was generated.

### Reproducing Experiments

#### Generating answers

We prompt models to generate answers to questions using
[scripts/generate.py](https://github.com/2015aroras/CaLMQA/blob/main/scripts/generate.py).
For each question in a data file
(e.g. [data/datasets/dataset-specific-german.json](https://github.com/2015aroras/CaLMQA/blob/main/data/datasets/dataset-specific-german.json)),
this script fills in the prompt template
[data/prompts/generation-prompt.txt](https://github.com/2015aroras/CaLMQA/blob/main/data/prompts/generation-prompt.txt)
with the question and then prompts a model with the question.
The base form of the command is
```
python scripts/generate.py <model> --dataset_load_path <dataset path> --dataset_save_path <save path> --temperature <temperature>
```
Supported models and more options can be found by running `python scripts/generate.py --help`.
For culturally agnostic questions (e.g. those in
[data/datasets/dataset-agnostic-german.json](https://github.com/2015aroras/CaLMQA/blob/main/data/datasets/dataset-agnostic-german.json)),
the extra argument `--q_translation_langs <language>` should be passed to tell the script to prompt
using the non-English version of the question.

We generated all answers with temperature set to 0.

#### Translating culturally agnostic questions

We translate culturally agnostic questions to other languages using
[scripts/translate.py](https://github.com/2015aroras/CaLMQA/blob/main/scripts/translate.py).
For each question the English culturally agnostic data file
[data/datasets/dataset-agnostic-english.json](https://github.com/2015aroras/CaLMQA/blob/main/data/datasets/dataset-agnostic-english.json),
this script fills in the prompt template
[data/prompts/question-translation-prompt.txt](https://github.com/2015aroras/CaLMQA/blob/main/data/prompts/question-translation-prompt.txt)
with the English question, English answer and target language name,
and then prompts a model to perform the translation.
The base form of the command is
```
python scripts/translate.py questions <model> --source_langs English --target_langs <target language> --dataset_save_path <save path> --temperature <temperature>
```
Supported models and more options can be found by running `python scripts/translate.py --help`.

We ran all translations of culturally agnostic questions with GPT-4 Turbo and temperature set to 0.

#### Categorizing questions

We prompt models to generate answers to questions using
[scripts/categorize.py](https://github.com/2015aroras/CaLMQA/blob/main/scripts/categorize.py).
For each question in a data file
(e.g. [data/datasets/dataset-specific-german.json](https://github.com/2015aroras/CaLMQA/blob/main/data/datasets/dataset-specific-german.json)),
this script fills in the categorization prompt
(e.g. [data/prompts/categorization-english-prompt.txt](https://github.com/2015aroras/CaLMQA/blob/main/data/prompts/categorization-english-prompt.txt))
with the question, category names, category descriptions and category examples.
The script uses the prompt to make a model categorize the question.
The base form of the command is
```
python scripts/categorize.py <model> --all_categories -p <prompt file> --dataset_load_path <dataset path> --dataset_save_path <save path> --temperature <temperature>
```
Supported models and more options can be found by running `python scripts/categorize.py --help`. We ran all our categorization with
temperature set to 0.

#### Language Detection

We detect language using
[scripts/detect_lang.py](https://github.com/2015aroras/CaLMQA/blob/main/scripts/detect_lang.py).
For each question in a data file
(e.g. [data/datasets/dataset-specific-german.json](https://github.com/2015aroras/CaLMQA/blob/main/data/datasets/dataset-specific-german.json)),
this script uses the [`polyglot`](https://pypi.org/project/polyglot/) or
[`langid`](https://pypi.org/project/py3langid/) package to detect the language
of a question or answer.
The command to detect the language of a model's answers is:
```
python scripts/detect_lang.py --dataset_load_path <dataset path> --model_name <model>
```

Similarly, language detection for questions can be done using
```
python scripts/detect_lang.py --dataset_load_path <dataset path> --check_questions
```
For culturally agnostic questions (e.g. those in
[data/datasets/dataset-agnostic-german.json](https://github.com/2015aroras/CaLMQA/blob/main/data/datasets/dataset-agnostic-german.json)),
the extra argument `--q_translation_lang <language>` should be passed to tell the script to prompt
using the non-English version of the question.

#### Repetition Detection

We detect repetitions using
[scripts/detect_repetitions.py](https://github.com/2015aroras/CaLMQA/blob/main/scripts/detect_repetitions.py).
For each question in a data file
(e.g. [data/datasets/dataset-specific-german.json](https://github.com/2015aroras/CaLMQA/blob/main/data/datasets/dataset-specific-german.json)),
the script tokenizes answers using [`tiktoken`](https://github.com/openai/tiktoken) with the
`o200_base` encoding and then looks for repeated n-grams.
The command to detect the percentage of a model's answers with repetitions is:
```
python scripts/detect_repetitions.py --dataset_load_path <dataset path> --model_name <model>
```
More options can be found by running `python scripts/detect_repetitions.py --help`. 
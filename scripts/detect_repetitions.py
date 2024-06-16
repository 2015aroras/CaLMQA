from __future__ import annotations

import argparse
from collections import Counter

import tiktoken
from calmqa.dataset import Dataset, QuestionType
from models.model import ModelName


def get_token_ids(string: str, encoding_name: str = "o200k_base") -> list[int]:
    """Returns the token IDs in a text string using the specified encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(string)
    return token_ids


def extract_ngrams(token_ids: list[int], n: int):
    """Extract n-grams from a list of token IDs."""
    return [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]


def has_repeated_ngrams(text: str, n: int, threshold: int):
    token_ids = get_token_ids(text)

    ngrams = extract_ngrams(token_ids, n)
    ngram_counts = Counter(ngrams)

    # check if any n-gram has a count greater than or equal to threshold
    return any(v >= threshold for v in ngram_counts.values())


def detect_text_repetitions(texts: list[str], n: int, threshold: int) -> None:
    num_repetitions = sum(int(has_repeated_ngrams(text, n, threshold)) for text in texts)
    print(f"Repetitions: {100 * num_repetitions / (len(texts) or 1):.2f}%")


def detect_repetitions(
    dataset_load_path: str,
    n: int,
    threshold: int,
    model_name_str: str,
    *,
    max_answers: int | None = None,
) -> None:
    dataset = Dataset.from_file(dataset_load_path)

    questions = dataset.get_questions(QuestionType.NONE)

    answers = [
        answer
        for question in questions
        for answer in dataset.get_answers(question, model_name=ModelName[model_name_str])
    ]

    if max_answers is not None:
        answers = answers[:max_answers]

    answer_texts = [answer.untranslated.text for answer in answers]
    detect_text_repetitions(answer_texts, n, threshold)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        choices=[model_name.name for model_name in ModelName],
        required=True,
        help="Name of model the answers of which to detect language",
    )
    parser.add_argument(
        "--max_answers",
        type=int,
        default=None,
        help="If set, limits the number of answers to the number provided. Used for testing.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="n-gram",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="Number of n-gram repetitions to be considered a repetitive answer",
    )
    parser.add_argument(
        "--lp",
        "--dataset_load_path",
        type=str,
        dest="dataset_load_path",
        default="data/dataset.json",
        help="Path of json file containing the dataset",
    )

    args = parser.parse_args()

    detect_repetitions(
        dataset_load_path=args.dataset_load_path,
        n=args.n,
        threshold=args.threshold,
        model_name_str=args.model_name,
        max_answers=args.max_answers,
    )


if __name__ == "__main__":
    main()

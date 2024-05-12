from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlfqa.dataset import Dataset, QuestionType
from mlfqa.language import Language


def simplify_cultural_dataset(
    dataset: Dataset,
) -> list[dict]:
    simplified_dataset = []
    for question in dataset.get_questions(QuestionType.CULTURAL):
        answers = dataset.get_answers(question)
        for answer in answers:
            if answer.prompting_state.other_state.get("is_multiple_choice") is True:
                continue

            simple_entry = {
                "lang": question.language.value,
                "type": "culture-specific",
                "question": question.untranslated.get_text(),
                "answer": answer.untranslated.text,
                "model": answer.prompting_state.model_name.value,
                "id": answer.name.replace(str(answer.language).lower(), ""),
            }
            simplified_dataset.append(simple_entry)

    return simplified_dataset


def simplify_non_cultural_dataset(
    dataset: Dataset,
) -> list[dict]:
    simplified_dataset = []
    for question in dataset.get_questions(QuestionType.NON_CULTURAL):
        answers = dataset.get_answers(question)
        for answer in answers:
            if answer.prompting_state.other_state.get("is_multiple_choice") is True:
                continue

            simple_entry = {
                "lang": answer.language.value,
                "type": "culture-agnostic",
                "question": question.translations[answer.language].get_text(),
                "answer": answer.untranslated.text,
                "model": answer.prompting_state.model_name.value,
                "id": answer.name.replace(str(answer.language).lower(), ""),
            }
            simplified_dataset.append(simple_entry)

    return simplified_dataset


def simplify_dataset(
    dataset_paths: list[str],
    output_path: str,
    *,
    language: Language | None,
) -> None:
    simplified_dataset = []
    for dataset_path in dataset_paths:
        dataset = Dataset.from_file(dataset_path)

        simplified_dataset += simplify_cultural_dataset(dataset)
        simplified_dataset += simplify_non_cultural_dataset(dataset)

        del dataset

    if language is not None:
        simplified_dataset = list(
            filter(lambda entry: entry["lang"] == language.value, simplified_dataset),
        )

    with Path(output_path).open("w", encoding="utf-8") as out_file:
        json.dump(simplified_dataset, out_file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        type=Language,
        default=None,
        help="If set, only return entries of this language",
    )
    parser.add_argument(
        "-i",
        "--datasets",
        dest="dataset_paths",
        nargs="+",
        required=True,
        help="Paths of json files containing datasets",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Path of save simplified dataset",
    )

    args = parser.parse_args()

    simplify_dataset(
        args.dataset_paths,
        args.output_path,
        language=args.language,
    )


if __name__ == "__main__":
    main()

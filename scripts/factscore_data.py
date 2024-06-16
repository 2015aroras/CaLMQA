from __future__ import annotations

import argparse
import json
from pathlib import Path

from calmqa.dataset import Dataset
from calmqa.language import Language
from models.model import ModelName


def create_factscore_input(
    model_names: list[ModelName],
    answer_languages: list[Language],
    non_cultural_datasets: list[Dataset],
) -> list[dict]:
    factscore_input = []
    entries = [entry for dataset in non_cultural_datasets for entry in dataset.entries]
    for entry in entries:
        for answer in entry.answers:
            language = answer.language
            if language not in answer_languages:
                continue

            if answer.prompting_state.model_name not in model_names:
                continue

            if Language.English not in answer.translations:
                # TODO: Make this an error eventually
                continue

            topic = entry.question.translations[Language.English].get_text()
            output = answer.translations[Language.English].text
            factscore_input.append(
                {
                    "topic": topic,
                    "output": output,
                },
            )

    return factscore_input


def create_and_save_factscore_input(
    model_names: list[ModelName],
    answer_languages: list[Language],
    non_cultural_dataset_paths: list[str],
    factscore_save_path: str,
) -> None:
    non_cultural_datasets = [
        Dataset.from_file(non_cultural_dataset_path.lower())
        for non_cultural_dataset_path in non_cultural_dataset_paths
    ]

    factscore_input = create_factscore_input(
        model_names,
        answer_languages,
        non_cultural_datasets,
    )

    with Path(factscore_save_path).open("w", encoding="utf-8") as save_file:
        for line in factscore_input:
            save_file.write(json.dumps(line, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert non-cultural data into FActScore input form",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        choices=[model_name.name for model_name in ModelName],
        help="Models to use",
    )
    parser.add_argument(
        "--answer_languages",
        type=Language,
        nargs="+",
        default=None,
        help="If set, only use answers whose original language is in this list",
    )
    parser.add_argument(
        "--ncdps",
        "--non_cultural_dataset_paths",
        type=str,
        nargs="*",
        default=[],
        dest="non_cultural_dataset_paths",
        help="Paths of jsons file containing non-cultural datasets",
    )
    parser.add_argument(
        "-o",
        "--factscore_save_path",
        type=str,
        dest="factscore_save_path",
        default="data/factscore/factscore_input.json",
        help="Path of file to save factscore input json to.",
    )

    args = parser.parse_args()

    create_and_save_factscore_input(
        model_names=[ModelName[name] for name in args.model_names],
        answer_languages=args.answer_languages,
        non_cultural_dataset_paths=args.non_cultural_dataset_paths,
        factscore_save_path=args.factscore_save_path,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
from collections import defaultdict

from calmqa.dataset import Dataset, QuestionType
from calmqa.language import Language


def split_non_cultural_dataset(
    dataset: Dataset,
) -> dict[Language, Dataset]:
    entries_by_lang: dict[Language, list[Dataset.Entry]] = defaultdict(list)
    for entry in dataset.entries:
        assert entry.question.language == Language.English
        assert entry.question.type & QuestionType.NON_CULTURAL == QuestionType.NON_CULTURAL

        entry_by_lang: dict[Language, Dataset.Entry] = {}
        for answer in entry.answers:
            answer_lang = answer.language

            if answer_lang not in entry_by_lang:
                # Make a copy of the question with only the relevant translation
                question_copy = copy.deepcopy(entry.question)
                for lang in list(question_copy.translations.keys()):
                    if lang not in (Language.English, answer_lang):
                        del question_copy.translations[lang]

                entry_by_lang[answer_lang] = Dataset.Entry(question_copy, [])

            entry_by_lang[answer_lang].answers.append(answer)

        for lang, lang_entry in entry_by_lang.items():
            entries_by_lang[lang].append(lang_entry)

    return {
        lang: Dataset(entries)
        for lang, entries in entries_by_lang.items()
    }


def split_and_save_non_cultural_dataset(
    dataset_path: str,
    output_path_pattern: str,
) -> None:
    if "[language]" not in output_path_pattern:
        msg = "Output path must contain '[language]' placeholder"
        raise ValueError(msg)

    dataset = Dataset.from_file(dataset_path)

    split_datasets = split_non_cultural_dataset(dataset)

    for language, lang_dataset in split_datasets.items():
        output_path = output_path_pattern.replace("[language]", language.name.lower())
        lang_dataset.to_file(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--dataset",
        dest="dataset_path",
        required=True,
        help="Path of json file containing the non cultural dataset",
    )
    parser.add_argument(
        "-o",
        "--output_path_pattern",
        required=True,
        help="Pattern of output non-cultural dataset paths. Use '[language]' as a placeholder.",
    )

    args = parser.parse_args()

    split_and_save_non_cultural_dataset(
        args.dataset_path,
        args.output_path_pattern,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import random

from datasets import Dataset as HfDataset
from datasets import ClassLabel, Features, Value
from dotenv import load_dotenv
from calmqa.dataset import Dataset, QuestionType
from calmqa.language import Language
from models.model import ModelName

MAX_WORKER_ID_LENGTH: int = 6


def simplify_cultural_dataset(
    dataset: Dataset,
    *,
    human_eval_only: bool = False,
) -> list[dict]:
    simplified_dataset = []
    for question in dataset.get_questions(QuestionType.CULTURAL):
        if human_eval_only and not question.human_evaluated:
            continue

        answers = dataset.get_answers(question, model_name=ModelName.HUMAN)
        assert len(answers) <= 1
        answer_text = answers[0].untranslated.text if len(answers) > 0 else None

        simple_entry = {
            # "source": question.source,
            "language": question.language.value,
            "question_type": "culturally specific",
            "question": question.untranslated.get_text(),
            "question_english": question.translations[Language.English].get_text(),
            "answer": answer_text,
            # "model": answer.prompting_state.model_name.value,
            # "id": answer.name.replace(str(answer.language).lower(), ""),
            # "worker_id": question.collector[:MAX_WORKER_ID_LENGTH],
            # "url": question.url,
        }
        simplified_dataset.append(simple_entry)

    return simplified_dataset


def simplify_non_cultural_dataset(
    dataset: Dataset,
    *,
    human_eval_only: bool = False,
) -> list[dict]:
    simplified_dataset = []
    for question in dataset.get_questions(QuestionType.NON_CULTURAL):
        if human_eval_only and not question.human_evaluated:
            continue

        answers = dataset.get_answers(question, model_name=ModelName.HUMAN)
        assert len(answers) <= 1
        answer_text = answers[0].untranslated.text if len(answers) > 0 else None

        assert len(question.translations) <= 2
        non_english_q_languages = [
            lang for lang in question.translations if lang != Language.English
        ]
        assert len(non_english_q_languages) <= 1
        language = (
            non_english_q_languages[0] if len(non_english_q_languages) > 0 else Language.English
        )

        simple_entry = {
            # "source": question.source,
            "language": language.value,
            "question_type": "culturally agnostic",
            "question": question.translations[language].get_text(),
            "question_english": question.translations[Language.English].get_text(),
            "answer": answer_text,
            # "model": answer.prompting_state.model_name.value,
            # "id": answer.name.replace(str(answer.language).lower(), ""),
            # "worker_id": question.collector[:MAX_WORKER_ID_LENGTH],
            # "url": question.url,
        }
        simplified_dataset.append(simple_entry)

    return simplified_dataset


def convert_to_hf(
    dataset_paths: list[str],
    output_path: str | None = None,
    repo: str | None = None,
    *,
    held_out_percent: float = 0.0,
    seed: int | None = None,
) -> None:
    if held_out_percent < 0.0 or held_out_percent > 1.0:
        raise ValueError("Held out data percentage must be between 0 and 1 inclusive.")

    rng = random.Random(seed) if seed is not None else None

    simplified_dataset: list[dict] = []
    for dataset_path in dataset_paths:
        dataset = Dataset.from_file(dataset_path)

        if rng is not None:
            rng.shuffle(dataset.entries)

        kept_entries_count = int((1 - held_out_percent) * len(dataset.entries))
        dataset.entries = dataset.entries[:kept_entries_count]

        dataset.entries = sorted(dataset.entries, key=lambda entry: int(entry.question.name.split(":")[1]))
        dataset.to_file(dataset_path)

        simplified_dataset += simplify_cultural_dataset(dataset)
        simplified_dataset += simplify_non_cultural_dataset(dataset)

        del dataset

    def generator():
        for item in simplified_dataset:
            yield item

    # sources = sorted(set(entry["source"] for entry in simplified_dataset))
    languages = sorted(set(entry["language"] for entry in simplified_dataset))
    question_types = sorted(set(entry["question_type"] for entry in simplified_dataset))

    hf_dataset = HfDataset.from_generator(
        generator,
        features=Features(
            {
                # "source": ClassLabel(names=sources),
                "language": ClassLabel(names=languages),
                "question_type": ClassLabel(names=question_types),
                "question": Value(dtype="string"),
                "question_english": Value(dtype="string"),
                "answer": Value(dtype="string"),
                # "model": answer.prompting_state.model_name.value,
                # "id": answer.name.replace(str(answer.language).lower(), ""),
                # "worker_id": Value(dtype="string"),
                # "url": Value(dtype="string"),
            }
        ),
    )

    if output_path is not None:
        hf_dataset.to_json(
            output_path,
            force_ascii=False,
            # lines=False,
            # orient="table",
        )
    if repo is not None:
        load_dotenv()

        res = hf_dataset.push_to_hub(repo)
        print(res)


def main() -> None:
    parser = argparse.ArgumentParser()
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
        help="Local path of jsonl to store Hugging Face dataset in",
    )
    parser.add_argument(
        "--repo",
        help="Name of repo to which result should be loaded",
    )

    parser.add_argument(
        "--held_out_percent",
        type=float,
        default=0.25,
        help="Percentage of the data from each file to skip. Entries at the end of file are skipped",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for shuffling data before withholding questions. If not set, no shuffling is performed.",
    )

    args = parser.parse_args()

    convert_to_hf(
        args.dataset_paths,
        args.output_path,
        args.repo,
        held_out_percent=args.held_out_percent,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

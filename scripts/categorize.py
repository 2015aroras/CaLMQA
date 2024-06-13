from __future__ import annotations

import argparse
import csv
import dataclasses
import re
import time
from pathlib import Path
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass

from calmqa.dataset import Category, Dataset, Question, QuestionType
from calmqa.language import Language
from models.model import Model, ModelName
from tqdm import tqdm


@dataclass
class CategoryMetadata:
    category: Category
    description: str
    example: str


@dataclass
class CategoriesMetadata:
    metadata: list[CategoryMetadata]

    @classmethod
    def from_file(cls, load_file_path: str) -> CategoriesMetadata:
        if load_file_path.endswith(".json"):
            categories_metadata_json = Path(load_file_path).read_text()
            return TypeAdapter(CategoriesMetadata).validate_json(categories_metadata_json)
        if load_file_path.endswith(".csv"):
            with Path(load_file_path).open("r") as f:
                reader = csv.DictReader(f)
                categories_metadata_dicts = [
                    {
                        "category": line_dict["Category"],
                        "description": line_dict["Description"],
                        "example": line_dict["Example"],
                    }
                    for line_dict in reader
                ]
                return TypeAdapter(CategoriesMetadata).validate_python(
                    {"metadata": categories_metadata_dicts}
                )

        raise NotImplementedError


def _insert_categories_in_prompt(
    prompt: str, categories: list[Category], categories_metadata: CategoriesMetadata
) -> str:
    categories_strings = []
    for category in categories:
        metadatas = [
            metadata for metadata in categories_metadata.metadata if metadata.category == category
        ]

        if len(metadatas) > 1:
            raise RuntimeError(f"Multiple instances of metadata found for category {category}")
        if len(metadatas) == 0:
            raise RuntimeError(f"No instances of metadata found for category {category}")
        metadata = metadatas[0]

        category_str = f"{category.value} – {metadata.description}."
        if metadata.example != "":
            category_str += f"  Example: {metadata.example}"
        categories_strings.append(category_str)

    return prompt.replace("[categories]", "\n".join(categories_strings))


def _extract_category(response: str, allowed_categories: list[Category]) -> Category | None:
    groups = re.findall(r"<category>(.*?)</category>", response, flags=re.IGNORECASE)

    # print(response, groups)
    if len(groups) > 1:
        raise RuntimeError(
            f"Multiple categories in response. Response: {response} . Categories: {groups}"
        )
    if len(groups) == 0:
        raise RuntimeError(f"No categories in response. Response: {response}")

    return Category(groups[0].lower())


def _categorize_and_store(
    model: Model,
    question: Question,
    prompt_template: str,
    categories: list[Category],
    categories_metadata: CategoriesMetadata,
    dataset: Dataset,
    *,
    overwrite: bool = False,
) -> bool:
    if not overwrite and question.category is not None:
        return False

    cleaned_question = question.untranslated.get_text()
    cleaned_question = re.sub("\n+", "\n", cleaned_question)

    cleaned_english_question = question.translations[Language.English].get_text()
    cleaned_english_question = re.sub("\n+", "\n", cleaned_english_question)

    prompt = prompt_template.replace("[question]", cleaned_question)
    prompt = prompt.replace("[translation]", cleaned_english_question)
    prompt = _insert_categories_in_prompt(prompt, categories, categories_metadata)

    response, _ = model.prompt(prompt)
    category = _extract_category(response, categories)

    question_dict = dataclasses.asdict(question)
    question_dict["category"] = category

    dataset.add_or_update_question(Question(**question_dict))

    return True


def categorize_and_store(
    model: Model,
    questions: list[Question],
    prompt_template: str,
    categories: list[Category],
    categories_metadata: CategoriesMetadata,
    dataset: Dataset,
    *,
    overwrite: bool = False,
    save_progress: bool = True,
    max_prompts_per_minute: int | None = None,
) -> None:
    minute_start = time.time() if max_prompts_per_minute else None
    prompts_in_minute = 0 if max_prompts_per_minute else None

    for question in tqdm(
        questions,
        desc=f"Categorizing {len(questions)} questions using {model.name.value}",
    ):
        # Limit max number of prompts within a minutes
        if max_prompts_per_minute is not None:
            assert minute_start is not None
            assert prompts_in_minute is not None

            if prompts_in_minute >= max_prompts_per_minute:
                time_to_minute_end = max(minute_start + 60 - time.time(), 0)
                time.sleep(time_to_minute_end)

                prompts_in_minute = 0

            if prompts_in_minute == 0:
                minute_start = time.time()

        model_prompted = _categorize_and_store(
            model,
            question,
            prompt_template,
            categories,
            categories_metadata,
            dataset,
            overwrite=overwrite,
        )

        if model_prompted and prompts_in_minute is not None:
            prompts_in_minute += 1

        if save_progress:
            dataset.to_file()


def categorize_questions(  # noqa: PLR0913
    categories: list[Category],
    model_name: ModelName,
    prompt_file_path: str,
    categories_metadata_path: str,
    dataset_load_path: str,
    dataset_save_path: str,
    max_output_tokens: int,
    *,
    question_names: list[str] | None = None,
    max_questions: int | None = None,
    max_prompts_per_minute: int | None = None,
    overwrite: bool = False,
    save_progress: bool = True,
    **kwargs,
) -> None:
    dataset = Dataset.from_file(dataset_load_path)
    dataset.default_save_path = dataset_save_path

    categories_metadata = CategoriesMetadata.from_file(categories_metadata_path)

    model = Model.make(model_name, max_output_tokens, **kwargs)
    prompt_template = Path(prompt_file_path).read_text()

    questions = dataset.get_questions(QuestionType.NONE)
    if question_names is not None:
        questions = [question for question in questions if question.name in question_names]
    if max_questions is not None:
        questions = questions[:max_questions]

    categorize_and_store(
        model,
        questions,
        prompt_template,
        categories,
        categories_metadata,
        dataset,
        overwrite=overwrite,
        save_progress=save_progress,
        max_prompts_per_minute=max_prompts_per_minute,
    )

    dataset.to_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        choices=[model_name.name for model_name in ModelName],
        help="Name of model to use to perform categorization",
    )
    parser.add_argument(
        "-p",
        "--prompt_file",
        type=str,
        default="data/prompts/categorization-non-english-prompt.txt",
        help="Path of file containing the prompt for categorization",
    )
    parser.add_argument(
        "--metadata",
        "--categories_metadata",
        dest="categories_metadata_path",
        type=str,
        default="data/categorization/categories-metadata.csv",
        help="Path of file containing the metadata of the categories",
    )

    categories_group = parser.add_mutually_exclusive_group(required=True)
    categories_group.add_argument(
        "--categories",
        type=lambda category: Category(category.title()),
        nargs="+",
        default=None,
        help="Categories list.",
    )
    categories_group.add_argument(
        "--all_categories",
        action="store_true",
        help="If set, use all categories.",
    )
    categories_group.add_argument(
        "--exclude_categories",
        type=lambda category: Category(category.title()),
        nargs="+",
        default=None,
        help="If set, use all categories except those in this list.",
    )

    parser.add_argument(
        "--question_names",
        nargs="+",
        default=None,
        help="If set, only use questions whose names are in this list.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="If set, limits the number of questions to the number provided. Used for testing.",
    )
    parser.add_argument(
        "--max_prompts_per_minute",
        type=int,
        default=None,
        help="If set, limits the number of times the model is prompted per minute.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrites existing translations with newly-generated translation.",
    )
    parser.add_argument(
        "--lp",
        "--dataset_load_path",
        type=str,
        dest="dataset_load_path",
        default="data/dataset.json",
        help="Path of json file containing the dataset",
    )
    parser.add_argument(
        "--sp",
        "--dataset_save_path",
        type=str,
        dest="dataset_save_path",
        default=None,
        help="Path of file to save the dataset to. Defaults to the load path",
    )

    # Local models args
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[],
        help="Ids of gpus available for use.",
    )
    parser.add_argument(
        "--max_gpu_mem",
        type=float,
        default=None,
        help="Max memory to use on each GPU in bytes.",
    )

    # Model parameter args
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max tokens in output.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Max tokens in output.",
    )

    args = parser.parse_args()

    if args.all_categories:
        categories = list(Category)
    elif args.exclude_categories:
        categories = [category for category in Category if category not in args.exclude_categories]
    else:
        categories = args.categories

    categorize_questions(
        categories=categories,
        model_name=ModelName[args.model_name],
        prompt_file_path=args.prompt_file,
        categories_metadata_path=args.categories_metadata_path,
        dataset_load_path=args.dataset_load_path,
        dataset_save_path=args.dataset_save_path or args.dataset_load_path,
        max_output_tokens=args.max_tokens,
        question_names=args.question_names,
        max_questions=args.max_questions,
        max_prompts_per_minute=args.max_prompts_per_minute,
        overwrite=args.overwrite,
        gpus=args.gpus,
        max_gpu_mem=int(args.max_gpu_mem) if args.max_gpu_mem is not None else None,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

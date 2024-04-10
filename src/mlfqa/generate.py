from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from models.model import Model, ModelName
from tqdm import tqdm

from mlfqa.dataset import Answer, Dataset, Question, QuestionType
from mlfqa.language import Language


def _prompt_model_and_store(  # noqa: PLR0913
    model: Model,
    question: Question,
    prompt_template: str,
    q_translation_language: Language,
    a_language: Language,
    dataset: Dataset,
    *,
    overwrite_existing_answers: bool = False,
) -> None:
    q_translation = question.translations[q_translation_language]

    prompt = prompt_template.replace("[question]", q_translation.get_text())
    prompt = prompt.replace("[question_language]", q_translation_language.name)
    prompt = prompt.replace("[answer_language]", a_language.name)

    existing_answers = dataset.get_answers(question, model.name.name, prompt, a_language)
    assert len(existing_answers) <= 1
    answer_id = existing_answers[0].a_id if len(existing_answers) == 1 else None

    if not overwrite_existing_answers and answer_id is not None:
        return

    response = model.prompt(prompt)

    answer = Answer.make(model.name.name, prompt, a_language, response)
    dataset.add_or_update_answer(question, answer)


def prompt_model_and_store(  # noqa: PLR0913
    model: Model,
    questions: list[Question],
    prompt_template: str,
    answer_langs: list[Language] | None,
    dataset: Dataset,
    q_translation_langs: list[Language] | None = None,
    *,
    overwrite_existing_answers: bool = False,
    save_progress: bool = True,
) -> None:
    for question in tqdm(questions, desc=f"Prompting {model.name}"):
        languages_to_prompt = q_translation_langs or [question.language]

        for q_language in languages_to_prompt:
            answer_languages = answer_langs or [q_language]

            for a_language in answer_languages:
                _prompt_model_and_store(
                    model,
                    question,
                    prompt_template,
                    q_language,
                    a_language,
                    dataset,
                    overwrite_existing_answers=overwrite_existing_answers,
                )

        if save_progress:
            dataset.to_file()


def generate(  # noqa: PLR0913
    model_name: ModelName,
    prompt_file_path: str,
    question_langs: list[Language] | None,
    q_translation_langs: list[Language] | None,
    answer_langs: list[Language] | None,
    question_type: QuestionType,
    dataset_load_path: str,
    dataset_save_path: str,
    max_output_tokens: int,
    *,
    max_questions: int | None = None,
    overwrite_answers: bool = False,
    save_progress: bool = True,
    **kwargs: Any,
) -> None:
    dataset = Dataset.from_file(dataset_load_path)
    dataset.default_save_path = dataset_save_path

    model = Model.make(model_name, max_output_tokens, **kwargs)
    prompt_template = Path(prompt_file_path).read_text()

    questions = dataset.get_questions(question_type, question_langs)
    if max_questions is not None:
        questions = questions[:max_questions]

    prompt_model_and_store(
        model,
        questions,
        prompt_template,
        answer_langs,
        dataset,
        q_translation_langs,
        overwrite_existing_answers=overwrite_answers,
        save_progress=save_progress,
    )

    dataset.to_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        choices=[model_name.name for model_name in ModelName],
        help="Name of model to use",
    )
    parser.add_argument(
        "-p",
        "--prompt_file",
        type=str,
        default="data/prompts/generation-prompt.txt",
        help="Path of file containing the prompt",
    )
    parser.add_argument(
        "--answer_langs",
        type=Language,
        nargs="+",
        default=None,
        help="Languages in which an answer should be generated. If not set, the answer will be generated in the same language as the question",  # noqa: E501
    )
    parser.add_argument(
        "--question_langs",
        type=Language,
        nargs="+",
        default=None,
        help="If set, only prompts with questions originally of the given languages",
    )
    parser.add_argument(
        "--q_translation_langs",
        type=Language,
        nargs="+",
        default=None,
        help="Only prompts using translations of questions in the given languages. If not set, untranslated questions are used.",  # noqa: E501
    )
    parser.add_argument(
        "--question_types",
        type=lambda inp: QuestionType[inp],
        nargs="+",
        default=[QuestionType.NONE],
        help="Filters the type of questions for which the model is prompted",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max tokens in output.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="If set, limits the number of questions to the number provided. Used for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
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
        default="data/dataset.json",
        help="Path of file to save the dataset to",
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

    args = parser.parse_args()

    question_type = QuestionType.NONE
    for q_type in args.question_types:
        question_type |= q_type

    generate(
        model_name=ModelName[args.model_name],
        prompt_file_path=args.prompt_file,
        answer_langs=args.answer_langs,
        question_langs=args.question_langs,
        q_translation_langs=args.q_translation_langs,
        question_type=question_type,
        dataset_load_path=args.dataset_load_path,
        dataset_save_path=args.dataset_save_path,
        max_output_tokens=args.max_tokens,
        max_questions=args.max_questions,
        overwrite_answers=args.overwrite,
        gpus=args.gpus,
        max_gpu_mem=int(args.max_gpu_mem) if args.max_gpu_mem is not None else None,
    )


if __name__ == "__main__":
    main()

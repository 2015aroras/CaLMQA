from __future__ import annotations

import argparse
import dataclasses
import itertools
import re
import time
from pathlib import Path

from calmqa.dataset import Answer, Dataset, Question, QuestionType
from calmqa.language import Language
from models.model import Model, ModelName
from tqdm import tqdm


def _prompt_model_and_store(  # noqa: PLR0913
    model: Model,
    question: Question,
    prompt_template: str,
    q_translation_language: Language,
    a_language: Language,
    dataset: Dataset,
    *,
    overwrite_existing_answers: bool = False,
) -> bool:
    q_translation = question.translations[q_translation_language]

    cleaned_question = q_translation.get_text()
    cleaned_question = re.sub("\n+", "\n", cleaned_question)

    prompt = prompt_template.replace("[question]", cleaned_question)
    prompt = prompt.replace("[question_language]", q_translation_language.name)
    prompt = prompt.replace("[answer_language]", a_language.name)

    prompting_state = model.get_prompting_state(prompt)
    other_state = {}

    prompting_state.other_state.update(other_state)
    existing_answers = dataset.get_answers(
        question,
        a_language,
        **dataclasses.asdict(prompting_state),
    )
    assert len(existing_answers) <= 1

    if not overwrite_existing_answers and len(existing_answers) == 1:
        return False

    response, prompting_state = model.prompt(prompt)
    prompting_state.other_state.update(other_state)

    answer_name = f"{question.name}:{q_translation_language}:{prompting_state.model_name.value}"
    answer_name = answer_name.lower()

    answer = Answer.make(answer_name, prompting_state, a_language, response)
    dataset.add_or_update_answer(question, answer)

    return True


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
    max_prompts_per_minute: int | None = None,
) -> None:
    desc = f"Prompting {model.name} with {len(questions)} questions"
    if q_translation_langs is not None:
        desc = f"{desc} translated into {len(q_translation_langs)} languages"
    if answer_langs is not None:
        desc = f"{desc}, with answers in {len(answer_langs)} languages"

    q_trans_langs = q_translation_langs or [None]
    answer_target_langs = answer_langs or [None]

    minute_start = time.time() if max_prompts_per_minute else None
    prompts_in_minute = 0 if max_prompts_per_minute else None

    for question, q_lang, a_lang in tqdm(
        itertools.product(questions, q_trans_langs, answer_target_langs),
        desc=desc,
        total=len(questions) * len(q_trans_langs) * len(answer_target_langs),
    ):
        q_language = q_lang or question.language
        if q_language not in question.translations:
            continue

        a_language = a_lang or q_language

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

        model_prompted = _prompt_model_and_store(
            model,
            question,
            prompt_template,
            q_language,
            a_language,
            dataset,
            overwrite_existing_answers=overwrite_existing_answers,
        )

        if model_prompted and prompts_in_minute is not None:
            prompts_in_minute += 1

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
    max_prompts_per_minute: int | None = None,
    overwrite_answers: bool = False,
    save_progress: bool = True,
    **kwargs,
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
        max_prompts_per_minute=max_prompts_per_minute,
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
        type=lambda lang: Language(lang.title()),
        nargs="+",
        default=None,
        help="Languages in which an answer should be generated. If not set, the answer will be generated in the same language as the question",  # noqa: E501
    )
    parser.add_argument(
        "--question_langs",
        type=lambda lang: Language(lang.title()),
        nargs="+",
        default=None,
        help="If set, only prompts with questions originally of the given languages",
    )

    q_translation_langs_group = parser.add_mutually_exclusive_group()
    q_translation_langs_group.add_argument(
        "--q_translation_langs",
        type=lambda lang: Language(lang.title()),
        nargs="+",
        default=None,
        help="Only prompts using translations of questions in the given languages. If not set, untranslated questions are used.",  # noqa: E501
    )
    q_translation_langs_group.add_argument(
        "--all_q_translation_langs",
        action="store_true",
        help="If set, prompts using all translations of questions.",
    )

    parser.add_argument(
        "--question_types",
        type=lambda inp: QuestionType[inp],
        nargs="+",
        default=[QuestionType.NONE],
        help="Filters the type of questions for which the model is prompted",
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

    question_type = QuestionType.NONE
    for q_type in args.question_types:
        question_type |= q_type

    if args.all_q_translation_langs:
        args.q_translation_langs = list(Language)

    generate(
        model_name=ModelName[args.model_name],
        prompt_file_path=args.prompt_file,
        answer_langs=args.answer_langs,
        question_langs=args.question_langs,
        q_translation_langs=args.q_translation_langs,
        question_type=question_type,
        dataset_load_path=args.dataset_load_path,
        dataset_save_path=args.dataset_save_path or args.dataset_load_path,
        max_output_tokens=args.max_tokens,
        max_questions=args.max_questions,
        max_prompts_per_minute=args.max_prompts_per_minute,
        overwrite_answers=args.overwrite,
        gpus=args.gpus,
        max_gpu_mem=int(args.max_gpu_mem) if args.max_gpu_mem is not None else None,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

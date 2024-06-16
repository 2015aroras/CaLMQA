from __future__ import annotations

import argparse
import itertools
import random
import re
from pathlib import Path
from typing import Any

from models.model import Model, ModelName
from tqdm import tqdm

from calmqa.dataset import Answer, Dataset, Question, QuestionType
from calmqa.language import Language


def _prompt_model_and_store(  # noqa: PLR0913
    model: Model,
    question: Question,
    prompt_template: str,
    q_translation_language: Language,
    answer_options: dict[str, Answer],
    correct_option_key: str,
    dataset: Dataset,
    *,
    rng_seed: int = 42,
    overwrite_existing_answers: bool = False,
) -> None:
    q_translation = question.translations[q_translation_language]

    cleaned_question = q_translation.get_text()
    cleaned_question = re.sub("\n+", "\n", cleaned_question)

    prompt = prompt_template
    prompt = prompt.replace("[question]", q_translation.get_text())
    prompt = prompt.replace("[language]", q_translation_language.name)

    other_state = {}
    other_state["rng_seed"] = rng_seed
    other_state["is_multiple_choice"] = True
    other_state["mc_options"] = {}
    other_state["mc_correct_answer"] = correct_option_key

    for key, answer in answer_options.items():
        if f"[option {key}]" not in prompt:
            msg = f"Invalid answer option {key}"
            raise ValueError(msg)
        option = answer.translations[q_translation_language].text
        option = re.sub("\n+", "\n", option)

        other_state["mc_options"][key] = option
        prompt = prompt.replace(f"[option {key}]", option)

    if len(matches := re.findall(r"\[option ([^\]]+)\]", prompt)) > 0:
        msg = f"No answer provided for MC options {matches}"
        raise ValueError(msg)

    answer_name = f"{question.name}:{q_translation_language}:{model.name.value}:mc".lower()
    existing_answer = dataset.get_answer(question, answer_name)

    if not overwrite_existing_answers and existing_answer is not None:
        return

    response, probabilities, prompting_state = model.prompt_and_next_token_probs(prompt)
    prompting_state.other_state.update(other_state)

    answer = Answer.make(
        answer_name,
        prompting_state,
        q_translation_language,
        response,
        option_probs=probabilities,
    )
    dataset.add_or_update_answer(question, answer)


def _create_mc_options(
    question: Question,
    option_keys: list[str],
    q_translation_language: Language,
    dataset: Dataset,
    rng: random.Random,
) -> tuple[dict[str, Answer], str]:
    human_answers = dataset.get_answers(
        question,
        model_name=ModelName.HUMAN,
    )
    if len(human_answers) == 0:
        msg = "No gold answers for this question, cannot convert to MC"
        raise RuntimeError(msg)
    assert len(human_answers) == 1
    human_answer = human_answers[0]

    random_answers = dataset.get_random_answers(
        len(option_keys) - 1,
        [human_answer],
        rng,
        model_name=ModelName.HUMAN,
        language=q_translation_language,
    )
    answers = [human_answer, *random_answers]
    # No MC answers allowed in random options
    assert all(
        not answer.prompting_state.other_state.get("is_multiple_choice", False)
        for answer in answers
    )
    rng.shuffle(answers)

    mc_options = dict(zip(option_keys, answers))

    correct_options = [
        key for key, answer in mc_options.items() if answer.name == human_answer.name
    ]
    assert len(correct_options) == 1

    return mc_options, correct_options[0]


def do_mc_and_store(  # noqa: PLR0913
    model: Model,
    questions: list[Question],
    prompt_template: str,
    dataset: Dataset,
    q_translation_langs: list[Language] | None = None,
    *,
    overwrite_existing_answers: bool = False,
    save_progress: bool = True,
    rng_seed: int = 42,
) -> None:
    mc_option_keys = re.findall(r"\[option ([^\]]+)\]", prompt_template)
    if len(mc_option_keys) == 0:
        msg = "No MC option placeholders in prompt template"
        raise ValueError(msg)

    desc = f"Prompting {model.name} with {len(questions)} MC questions"
    if q_translation_langs is not None:
        desc = f"{desc} translated into {len(q_translation_langs)} languages"

    q_trans_langs = q_translation_langs or [None]
    rng = random.Random(rng_seed)

    for question, q_lang in tqdm(
        itertools.product(questions, q_trans_langs),
        desc=desc,
        total=len(questions) * len(q_trans_langs),
    ):
        q_language = q_lang or question.language
        if q_language not in question.translations:
            continue

        answer_options, correct_option = _create_mc_options(
            question,
            mc_option_keys,
            q_language,
            dataset,
            rng,
        )

        _prompt_model_and_store(
            model,
            question,
            prompt_template,
            q_language,
            answer_options,
            correct_option,
            dataset,
            rng_seed=rng_seed,
            overwrite_existing_answers=overwrite_existing_answers,
        )

        if save_progress:
            dataset.to_file()


def do_multiple_choice(  # noqa: PLR0913
    model_name: ModelName,
    prompt_file_path: str,
    question_langs: list[Language] | None,
    q_translation_langs: list[Language] | None,
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

    do_mc_and_store(
        model,
        questions,
        prompt_template,
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
        default="data/prompts/multiple-choice-prompt.txt",
        help="Path of file containing the prompt",
    )
    parser.add_argument(
        "--question_langs",
        type=Language,
        nargs="+",
        default=None,
        help="If set, only prompts with questions originally of the given languages",
    )

    q_translation_langs_group = parser.add_mutually_exclusive_group()
    q_translation_langs_group.add_argument(
        "--q_translation_langs",
        type=Language,
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

    args = parser.parse_args()

    question_type = QuestionType.NONE
    for q_type in args.question_types:
        question_type |= q_type

    if args.all_q_translation_langs:
        args.q_translation_langs = list(Language)

    do_multiple_choice(
        model_name=ModelName[args.model_name],
        prompt_file_path=args.prompt_file,
        question_langs=args.question_langs,
        q_translation_langs=args.q_translation_langs,
        question_type=question_type,
        dataset_load_path=args.dataset_load_path,
        dataset_save_path=args.dataset_save_path or args.dataset_load_path,
        max_output_tokens=args.max_tokens,
        max_questions=args.max_questions,
        overwrite_answers=args.overwrite,
        gpus=args.gpus,
        max_gpu_mem=int(args.max_gpu_mem) if args.max_gpu_mem is not None else None,
    )


if __name__ == "__main__":
    main()

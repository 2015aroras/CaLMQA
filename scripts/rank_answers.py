from __future__ import annotations

import argparse
import copy
import dataclasses
import random
import re
from pathlib import Path

from calmqa.dataset import Answer, Dataset, Question, QuestionType
from models.model import Model, ModelName
from tqdm import tqdm


def _get_pairwise_ranking(response: str) -> int:
    is_first_best = "[[A]]" in response
    is_second_best = "[[B]]" in response
    is_tie = "[[C]]" in response

    if is_first_best + is_second_best + is_tie != 1:
        raise ValueError(f"Response does not give clear result: {response}")

    if is_first_best:
        return -1
    if is_second_best:
        return 1
    if is_tie:
        return 0
    raise NotImplementedError


def _get_criteria_key(criteria: str, rater_model_name: ModelName) -> str:
    return f"{criteria}|{rater_model_name.name}"


def _get_pair_key(answer1: Answer, answer2: Answer) -> str:
    return f"{answer1.name}|{answer2.name}"


def rank_answer_pair(
    answer1: Answer,
    answer2: Answer,
    ranker_model: Model,
    question: Question,
    prompt_template: str,
    criteria_name: str,
    criteria_desc: str,
) -> int:
    cleaned_question = question.untranslated.get_text()
    cleaned_question = re.sub("\n+", "\n", cleaned_question)

    prompt = prompt_template.replace("[question]", cleaned_question)
    prompt = prompt.replace("[answerA]", answer1.untranslated.text)
    prompt = prompt.replace("[answerB]", answer2.untranslated.text)
    prompt = prompt.replace("[source_language]", question.language.value)
    prompt = prompt.replace("[criteria_name]", criteria_name)
    prompt = prompt.replace("[criteria_desc]", criteria_desc)

    response, _ = ranker_model.prompt(prompt)

    return _get_pairwise_ranking(response)


def pairwise_rank_answers(
    answers: list[Answer],
    rater_model: Model,
    question: Question,
    prompt_template: str,
    criteria_name: str,
    criteria_desc: str,
    dataset: Dataset,
    *,
    both_orders: bool = False,
    overwrite: bool = False,
    save_progress: bool = True,
) -> bool:
    criteria_key = _get_criteria_key(criteria_name, rater_model.name)
    pairwise_best_answers = copy.deepcopy(question.pairwise_best_answers) or {}

    if overwrite or criteria_key not in pairwise_best_answers:
        pairwise_best_answers[criteria_key] = {}
    criteria_pairwise_best_answers = pairwise_best_answers[criteria_key]

    for i, answer1 in enumerate(answers):
        for answer2 in answers[i + 1 :]:
            # Skip if only 1 order is needed and is present
            if (
                not both_orders
                and (
                    _get_pair_key(answer1, answer2) in criteria_pairwise_best_answers
                    or _get_pair_key(answer2, answer1) in criteria_pairwise_best_answers
                )
            ):
                continue

            answer_pairs = [(answer1, answer2), (answer2, answer1)]
            if not both_orders:
                answer_pairs = [random.choice(answer_pairs)]

            for a1, a2 in answer_pairs:
                if _get_pair_key(a1, a2) not in criteria_pairwise_best_answers:
                    rank = rank_answer_pair(
                        a1,
                        a2,
                        rater_model,
                        question,
                        prompt_template,
                        criteria_name,
                        criteria_desc,
                    )
                    criteria_pairwise_best_answers[_get_pair_key(a1, a2)] = rank

    question_dict = dataclasses.asdict(question)
    question_dict["pairwise_best_answers"] = pairwise_best_answers

    updated_question = Question(**question_dict)

    dataset.add_or_update_question(updated_question)

    if save_progress:
        dataset.to_file()

    return True


def rank_answers(  # noqa: PLR0913
    ranker_model_name: ModelName,
    answer_model_names: list[ModelName],
    prompt_file_path: str,
    criteria_name: str,
    criteria_desc_file_path: str,
    dataset_load_path: str,
    dataset_save_path: str,
    max_output_tokens: int,
    *,
    pairwise: bool = True,
    both_orders: bool = False,
    max_questions: int | None = None,
    overwrite: bool = False,
    save_progress: bool = True,
    **kwargs,
) -> None:
    dataset = Dataset.from_file(dataset_load_path)
    dataset.default_save_path = dataset_save_path

    ranker_model = Model.make(ranker_model_name, max_output_tokens, **kwargs)
    criteria_desc = Path(criteria_desc_file_path).read_text()
    prompt_template = Path(prompt_file_path).read_text()

    questions = dataset.get_questions(QuestionType.NONE)

    if max_questions is not None:
        questions = questions[:max_questions]

    for question in tqdm(
        questions,
        desc=f"Ranking {len(questions)} questions using {ranker_model.name.value}",
    ):
        answers = []
        for model_name in answer_model_names:
            model_answers = dataset.get_answers(question, model_name=model_name)
            assert len(model_answers) <= 1
            if len(model_answers) == 1:
                answers.append(model_answers[0])

        if len(answers) == 0:
            continue

        if len(answers) != len(answer_model_names):
            raise RuntimeError(
                f"Expected answers from {len(answer_model_names)} models, received answers from {len(answers)} models"
            )

        if pairwise:
            pairwise_rank_answers(
                answers,
                ranker_model,
                question,
                prompt_template,
                criteria_name,
                criteria_desc,
                dataset,
                both_orders=both_orders,
                overwrite=overwrite,
                save_progress=save_progress,
            )
        else:
            raise NotImplementedError

    dataset.to_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ranker_model",
        type=str,
        choices=[model_name.name for model_name in ModelName],
        help="Name of model to use to rank answers",
    )
    parser.add_argument(
        "answer_models",
        type=str,
        nargs="+",
        choices=[model_name.name for model_name in ModelName],
        help="Name of model whose answers are being ranked",
    )

    parser.add_argument(
        "--criteria_name",
        required=True,
        help="Name of criteria being ranked",
    )
    parser.add_argument(
        "-d",
        "--criteria_desc_file",
        type=str,
        required=True,
        help="Path of file containing the description of the ranking criteria",
    )
    parser.add_argument(
        "-p",
        "--prompt_file",
        type=str,
        default="data/prompts/pairwise-ranking-prompt.txt",
        help="Path of file containing the prompt for ranking",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="If set, rank answers in pairwise fashion instead of all at once.",
    )
    parser.add_argument(
        "--both_orders",
        action="store_true",
        help="If this and `pairwise` are both set, rank both orders of answers.",
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

    rank_answers(
        ranker_model_name=ModelName[args.ranker_model.upper()],
        answer_model_names=[ModelName[name] for name in args.answer_models],
        pairwise=args.pairwise,
        both_orders=args.both_orders,
        prompt_file_path=args.prompt_file,
        criteria_name=args.criteria_name,
        criteria_desc_file_path=args.criteria_desc_file,
        dataset_load_path=args.dataset_load_path,
        dataset_save_path=args.dataset_save_path or args.dataset_load_path,
        max_output_tokens=args.max_tokens,
        max_questions=args.max_questions,
        overwrite=args.overwrite,
        gpus=args.gpus,
        max_gpu_mem=int(args.max_gpu_mem) if args.max_gpu_mem is not None else None,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

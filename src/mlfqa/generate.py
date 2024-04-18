from __future__ import annotations

import argparse
import dataclasses
import itertools
import random
import re
from pathlib import Path

from models.model import Model, ModelName
from tqdm import tqdm

from mlfqa.dataset import Answer, Dataset, Question, QuestionType
from mlfqa.language import Language

RAG_RNG_SEED = 1


def _prompt_model_and_store(  # noqa: PLR0913
    model: Model,
    question: Question,
    prompt_template: str,
    q_translation_language: Language,
    a_language: Language,
    dataset: Dataset,
    rag_num_documents: int,
    *,
    rng: random.Random | None = None,
    overwrite_existing_answers: bool = False,
) -> None:
    q_translation = question.translations[q_translation_language]

    cleaned_question = q_translation.get_text()
    cleaned_question = re.sub("\n+", "\n", cleaned_question)

    prompt = prompt_template.replace("[question]", cleaned_question)
    prompt = prompt.replace("[question_language]", q_translation_language.name)
    prompt = prompt.replace("[answer_language]", a_language.name)

    prompting_state = model.get_prompting_state(prompt)
    other_state = {}

    if rag_num_documents > 0:
        assert rng is not None

        if "[documents]" not in prompt_template:
            msg = f"Retrieval augmented requested but no [documents] placeholder in template {prompt_template}"
            raise ValueError(msg)
        assert "[documents]" not in prompt

        human_answers = dataset.get_answers(
            question,
            model_name=ModelName.HUMAN,
        )
        assert len(human_answers) == 1
        human_answer = human_answers[0]

        random_answers = dataset.get_random_answers(
            rag_num_documents - 1,
            [human_answer],
            rng,
            language=q_translation_language,
        )
        answers = [human_answer, *random_answers]
        rng.shuffle(answers)
        answers: list[Answer]

        documents = [
            f"Document {i+1}: {re.sub("\n+", "\n", answer.translations[q_translation_language].text)}"
            for i, answer in enumerate(answers)
        ]

        prompt = prompt.replace("[documents]", "\n".join(documents))

        other_state["rag_answer_names"] = [answer.name for answer in answers]

    existing_answers = dataset.get_answers(
        question,
        a_language,
        **dataclasses.asdict(prompting_state),
        **other_state,
    )
    assert len(existing_answers) <= 1

    if not overwrite_existing_answers and len(existing_answers) == 1:
        return

    response, prompting_state = model.prompt(prompt)
    prompting_state.other_state.update(other_state)

    answer_name = f"{question.name}:{q_translation_language}:{prompting_state.model_name.value}"
    if rag_num_documents > 0:
        answer_name += f":rag{rag_num_documents}"

    answer = Answer.make(answer_name, prompting_state, a_language, response)
    dataset.add_or_update_answer(question, answer)


def prompt_model_and_store(  # noqa: PLR0913
    model: Model,
    questions: list[Question],
    prompt_template: str,
    answer_langs: list[Language] | None,
    dataset: Dataset,
    rag_num_documents: int,
    q_translation_langs: list[Language] | None = None,
    *,
    overwrite_existing_answers: bool = False,
    save_progress: bool = True,
) -> None:
    desc = f"Prompting {model.name} with {len(questions)} questions"
    if q_translation_langs is not None:
        desc = f"{desc} translated into {len(q_translation_langs)} languages"
    if answer_langs is not None:
        desc = f"{desc}, with answers in {len(answer_langs)} languages"

    q_trans_langs = q_translation_langs or [None]
    answer_target_langs = answer_langs or [None]

    rng = random.Random(RAG_RNG_SEED) if rag_num_documents > 0 else None

    for question, q_lang, a_lang in tqdm(
        itertools.product(questions, q_trans_langs, answer_target_langs),
        desc=desc,
        total=len(questions) * len(q_trans_langs) * len(answer_target_langs),
    ):
        q_language = q_lang or question.language
        if q_language not in question.translations:
            continue

        a_language = a_lang or q_language

        _prompt_model_and_store(
            model,
            question,
            prompt_template,
            q_language,
            a_language,
            dataset,
            rag_num_documents,
            rng=rng,
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
    rag_num_documents: int,
    *,
    max_questions: int | None = None,
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
        rag_num_documents,
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

    # Retrieval augmented generation args
    parser.add_argument(
        "--rag_num_documents",
        type=int,
        default=0,
        help="Number of sample answers ('documents') to use for retrieval augmented generation. If set to 0, then retrieval augmentation will not be used.",
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
        dataset_save_path=args.dataset_save_path,
        max_output_tokens=args.max_tokens,
        max_questions=args.max_questions,
        overwrite_answers=args.overwrite,
        gpus=args.gpus,
        rag_num_documents=args.rag_num_documents,
        max_gpu_mem=int(args.max_gpu_mem) if args.max_gpu_mem is not None else None,
    )


if __name__ == "__main__":
    main()

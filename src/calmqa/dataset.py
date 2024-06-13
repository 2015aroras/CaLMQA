from __future__ import annotations

import argparse
import copy
import csv
import enum
import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

from models.claude_model import ClaudePromptParameters
from models.google_model import GooglePromptingState
from models.mistral_model import MistralPromptingState
from models.model import ModelName, PromptingState
from models.openai_model import OpenAIPromptParameters
from models.transformers_model import TransformersPromptParameters
from pydantic import Discriminator, RootModel, Tag, TypeAdapter
from pydantic import Field as PyField
from pydantic.dataclasses import dataclass

from calmqa.language import Language

if TYPE_CHECKING:
    import random

Source = str
PromptingStateAnnotation = Annotated[
    Annotated[OpenAIPromptParameters, Tag(OpenAIPromptParameters.__name__)]
    | Annotated[ClaudePromptParameters, Tag(ClaudePromptParameters.__name__)]
    | Annotated[GooglePromptingState, Tag(GooglePromptingState.__name__)]
    | Annotated[MistralPromptingState, Tag(MistralPromptingState.__name__)]
    | Annotated[TransformersPromptParameters, Tag(TransformersPromptParameters.__name__)]
    | Annotated[PromptingState, Tag(PromptingState.__name__)],
    Discriminator(PromptingState.get_discriminator_value),
]

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "[question]"


class Category(enum.Enum):
    EDUCATION_AND_CAREER = "education and career"
    GOVERNANCE_AND_SOCIETY = "governance and society"
    GEOGRAPHY_TOURISM_AND_CLIMATE = "geography, tourism, and climate"
    TECHNOLOGY = "technology"
    ECONOMY_AND_INDUSTRY = "economy and industry"
    MEDIA_AND_ENTERTAINMENT = "media and entertainment"
    FOOD_AND_DRINK = "food and drinks"
    HISTORY = "history"
    LANGUAGE_ART_AND_LITERATURE = "language, art and literature"
    RELIGION_BELIEFS_CUSTOMS_TRADITIONS = "religion, beliefs, customs, and traditions"
    HEALTH_AND_WELLNESS = "health and wellness"
    OTHER = "other"


class QuestionType(enum.IntFlag):
    NONE = 0
    CULTURAL = enum.auto()
    NON_CULTURAL = enum.auto()
    LONG_FORM = enum.auto()
    ALL = NON_CULTURAL | CULTURAL | LONG_FORM


@dataclass(frozen=True)
class QuestionTranslation:
    language: Language
    text: str
    prompting_state: PromptingStateAnnotation | None = PyField(default=None)

    def get_text(self) -> str:
        return self.text


@dataclass(frozen=True)
class Question:
    name: str
    type: QuestionType
    source: Source
    collector: str
    language: Language
    translations: dict[Language, QuestionTranslation]
    url: str | None = PyField(default=None)
    human_evaluated: bool = PyField(default=False)
    category: Category | None = PyField(default=None)

    @property
    def untranslated(self) -> QuestionTranslation:
        return self.translations[self.language]


@dataclass(frozen=True)
class AnswerTranslation:
    language: Language
    text: str
    prompting_state: PromptingStateAnnotation | None = PyField(default=None)


@dataclass(frozen=True)
class Answer:
    name: str
    language: Language
    translations: dict[Language, AnswerTranslation]
    prompting_state: PromptingStateAnnotation
    option_probs: dict[str, float] | None = PyField(default=None)

    @classmethod
    def make(
        cls,
        name: str,
        prompting_state: PromptingStateAnnotation,
        language: Language,
        text: str,
        *,
        option_probs: dict[str, float] | None = None,
    ) -> Answer:
        """`Create an answer without any translations."""
        return cls(
            name,
            language,
            {language: AnswerTranslation(language, text)},
            prompting_state,
            option_probs=option_probs,
        )

    @classmethod
    def make_human_answer(
        cls,
        name: str,
        prompt: str,
        language: Language,
        text: str,
        *,
        option_probs: dict[str, float] | None = None,
    ) -> Answer:
        """Create a human's answer without any translations."""
        prompting_state = PromptingState(prompt, ModelName.HUMAN, -1)

        return cls(
            name,
            language,
            {language: AnswerTranslation(language, text)},
            prompting_state,
            option_probs=option_probs,
        )

    @property
    def untranslated(self) -> AnswerTranslation:
        return self.translations[self.language]


@dataclass
class Dataset:
    entries: list[Dataset.Entry]
    default_save_path: str | None = PyField(default=None, exclude=True)

    def __post_init__(self) -> None:
        answer_names = [answer.name for entry in self.entries for answer in entry.answers]

        names_counter = Counter(answer_names)
        duplicate_answer_names = {name for name in names_counter if names_counter[name] > 1}
        if len(duplicate_answer_names) > 0:
            msg = f"Duplicate answer names found in dataset: {duplicate_answer_names}"
            raise ValueError(msg)

    @dataclass
    class Entry:
        question: Question
        answers: list[Answer]

    def get_questions(
        self,
        q_type: QuestionType,
        languages: list[Language] | None = None,
    ) -> list[Question]:
        return [
            copy.deepcopy(entry.question)
            for entry in self.entries
            if entry.question.type & q_type == q_type
            and (languages is None or entry.question.language in languages)
        ]

    def _get_entry(self, question: Question) -> Dataset.Entry:
        matching_entries = [entry for entry in self.entries if entry.question.name == question.name]
        assert len(matching_entries) <= 1, f"More than 1 entry for question {question.name}"
        assert len(matching_entries) >= 1, f"No entry for a question {question.name}"
        return matching_entries[0]

    def get_answer(
        self,
        question: Question,
        answer_name: str,
    ) -> Answer | None:
        entry = self._get_entry(question)

        matching_answers: list[Answer] = [
            answer for answer in entry.answers if answer.name == answer_name
        ]
        assert len(matching_answers) <= 1

        return copy.deepcopy(matching_answers[0]) if len(matching_answers) > 0 else None

    def _answer_matches_state(
        self,
        answer: Answer,
        **prompting_state_filter_kwargs,
    ) -> bool:
        # Check other_state
        matches = all(
            answer.prompting_state.other_state.get(key) == val
            for key, val in prompting_state_filter_kwargs.get("other_state", {}).items()
        )

        # Check base prompting state
        return matches and all(
            key == "other_state" or getattr(answer.prompting_state, key, None) == val
            for key, val in prompting_state_filter_kwargs.items()
        )

    def get_answers(
        self,
        question: Question,
        language: Language | None = None,
        **prompting_state_filter_kwargs,
    ) -> list[Answer]:
        entry = self._get_entry(question)

        matching_answers: list[Answer] = []
        for answer in entry.answers:
            if language is not None and answer.language != language:
                continue

            if self._answer_matches_state(answer, **prompting_state_filter_kwargs):
                matching_answers.append(copy.deepcopy(answer))

        return matching_answers

    def get_random_answers(
        self,
        num_answers: int,
        exclude_answers: list[Answer],
        rng: random.Random,
        language: Language | None = None,
        **prompting_state_filter_kwargs,
    ) -> list[Answer]:
        excluded_answer_names = {answer.name for answer in exclude_answers}

        candidate_answers = [
            answer
            for entry in self.entries
            for answer in entry.answers
            if answer.name not in excluded_answer_names
            and (language is None or answer.language == language)
            and self._answer_matches_state(answer, **prompting_state_filter_kwargs)
        ]

        return rng.sample(candidate_answers, num_answers)

    def add_or_update_question(
        self,
        question: Question,
    ) -> None:
        matching_entries = [entry for entry in self.entries if entry.question.name == question.name]
        assert len(matching_entries) <= 1, f"More than 1 entry for question {question.name}"

        if len(matching_entries) == 1:
            entry = matching_entries[0]
            entry.question = copy.deepcopy(question)
        else:
            entry = Dataset.Entry(copy.deepcopy(question), [])
            self.entries.append(entry)

    def add_or_update_question_translation(
        self,
        question: Question,
        translation: QuestionTranslation,
    ) -> None:
        entry = self._get_entry(question)
        entry.question.translations[translation.language] = copy.deepcopy(translation)

    def add_or_update_answer(
        self,
        question: Question,
        answer: Answer,
    ) -> None:
        entry = self._get_entry(question)

        existing_answers = [
            entry_answer for entry_answer in entry.answers if entry_answer.name == answer.name
        ]
        assert len(existing_answers) <= 1, f"Too many answers match answer with name: {answer.name}"

        if len(existing_answers) == 1:
            existing_answer = existing_answers[0]
            entry.answers = [
                entry_answer
                for entry_answer in entry.answers
                if entry_answer.name != existing_answer.name
            ]

        entry.answers.append(copy.deepcopy(answer))

    def to_file(self, save_file_path: str | None = None) -> None:
        save_file_path = save_file_path or self.default_save_path
        if not save_file_path:
            msg = "No dataset save file path provided and no default is set"
            raise ValueError(msg)

        dataset_json = RootModel[Dataset](self).model_dump_json(
            indent=2,
            exclude_none=True,
            round_trip=True,
            serialize_as_any=True,
        )
        Path(save_file_path).write_text(dataset_json)

    @classmethod
    def from_file(cls, load_file_path: str) -> Dataset:
        dataset_json = Path(load_file_path).read_text()
        return TypeAdapter(Dataset).validate_json(dataset_json)

    def _merge_entries(self, entry: Dataset.Entry, other_entry: Dataset.Entry) -> None:
        assert entry.question.language == other_entry.question.language
        assert entry.question.name == other_entry.question.name
        assert entry.question.source == other_entry.question.source
        assert entry.question.type == other_entry.question.type
        assert entry.question.url == other_entry.question.url
        entry.question.translations.update(other_entry.question.translations)

        msg = "Answer merging not yet implemented"
        raise NotImplementedError(msg)

    def merge(self, other: Dataset) -> None:
        self_entries_by_q_name = {entry.question.name: entry for entry in self.entries}

        for other_entry in other.entries:
            if other_entry.question.name not in self_entries_by_q_name:
                self.entries.append(copy.deepcopy(other_entry))
            else:
                self_entry = self_entries_by_q_name[other_entry.question.name]
                self._merge_entries(self_entry, other_entry)


class Field(enum.Enum):
    IS_CULTURAL = enum.auto()
    LANGUAGE = enum.auto()
    QUESTION_WITH_ELABORATION = enum.auto()
    QUESTION_TITLE = enum.auto()
    QUESTION_ELABORATION = enum.auto()
    QUESTION_ENGLISH_TRANSLATION = enum.auto()
    QUESTION_SOURCE = enum.auto()
    QUESTION_COLLECTOR = enum.auto()
    ANSWER = enum.auto()
    ANSWER_SOURCE = enum.auto()
    URL = enum.auto()


def _construct_dataset_from_dict_entries(
    dataset_name: str,
    dict_entries: list[dict[Field, Any]],
) -> Dataset:
    entries: list[Dataset.Entry] = []

    for i, dict_entry in enumerate(dict_entries):
        question_type = QuestionType.LONG_FORM

        if not isinstance(dict_entry[Field.IS_CULTURAL], bool):
            msg = f"Is cultural is not a boolean: {dict_entry[Field.IS_CULTURAL]}"
            raise TypeError(msg)
        is_cultural_question = dict_entry[Field.IS_CULTURAL]

        question_type |= (
            QuestionType.CULTURAL if is_cultural_question else QuestionType.NON_CULTURAL
        )

        language = Language[dict_entry[Field.LANGUAGE]]

        question_translations: dict[Language, QuestionTranslation] = {}
        if Field.QUESTION_WITH_ELABORATION in dict_entry:
            question_untranslated = QuestionTranslation(
                language,
                text=dict_entry[Field.QUESTION_WITH_ELABORATION],
            )
        elif Field.QUESTION_TITLE in dict_entry:
            question_untranslated = QuestionTranslation(
                language,
                text=dict_entry[Field.QUESTION_TITLE]
                + "\n"
                + dict_entry.get(Field.QUESTION_ELABORATION),
            )
        else:
            msg = "No question text fields found"
            raise ValueError(msg)
        question_translations[language] = question_untranslated

        if Field.QUESTION_ENGLISH_TRANSLATION in dict_entry:
            question_translations[Language.English] = QuestionTranslation(
                Language.English,
                text=dict_entry[Field.QUESTION_ENGLISH_TRANSLATION],
            )

        question = Question(
            f"{dataset_name}:{i}",
            question_type,
            dict_entry[Field.QUESTION_SOURCE],
            dict_entry[Field.QUESTION_COLLECTOR],
            language,
            question_translations,
            dict_entry.get(Field.URL),
        )

        answers = []
        if Field.ANSWER in dict_entry:
            answer = Answer.make_human_answer(
                f"{question.name}:human",
                question.untranslated.get_text(),
                language,
                dict_entry[Field.ANSWER],
            )
            answers.append(answer)

        entry = Dataset.Entry(question, answers)
        entries.append(entry)

    return Dataset(entries)


def _construct_dataset_from_csv_and_metadata(
    csv_path: str,
    columns_metadata_path: str,
    dataset_name: str,
) -> Dataset:
    with Path(csv_path).open("r") as prolific_data_file:
        csv_data = csv.DictReader(prolific_data_file)

        with Path(columns_metadata_path).open("r") as columns_metadata_file:
            columns_metadata = json.load(columns_metadata_file)
            if not isinstance(columns_metadata, dict):
                msg = "Provided columns metadata is not a dictionary"
                raise TypeError(msg)
            columns_metadata = cast(dict[str, Any], columns_metadata)

        dict_entries: list[dict[Field, Any]] = []

        for entry in csv_data:
            dict_entry: dict[Field, Any] = {}

            for column_name, column_metadata in columns_metadata.items():
                if not isinstance(column_metadata, dict):
                    msg = (
                        f"Column metadata {column_metadata} for column {column_name} is not a dict"
                    )
                    raise TypeError(msg)
                column_metadata = cast(dict[str, Any], column_metadata)

                value = (
                    column_metadata["value"] if "value" in column_metadata else entry[column_name]
                )
                if value == "" and column_metadata.get("required", False):
                    msg = f"Column {column_name} is empty in {entry} but is required"
                    raise ValueError(msg)

                field_str = column_metadata["field"]
                assert isinstance(field_str, str)
                field = Field[field_str.upper()]

                if field in dict_entry:
                    msg = f"Field {field} found more than once in {entry}"
                    raise ValueError(msg)
                dict_entry[field] = value

            dict_entries.append(dict_entry)

    return _construct_dataset_from_dict_entries(dataset_name, dict_entries)


def _construct_dataset_from_csvs_and_metadata(
    csv_paths: str,
    columns_metadata_path: str,
    dataset_name: str,
) -> Dataset:
    return Dataset(
        [
            entry
            for csv_path in csv_paths
            for entry in _construct_dataset_from_csv_and_metadata(
                csv_path,
                columns_metadata_path,
                dataset_name,
            ).entries
        ],
    )


def _construct_and_save_dataset(args: argparse.Namespace) -> None:
    dataset: Dataset
    if args.data_source == "csv":
        dataset = _construct_dataset_from_csvs_and_metadata(
            args.csv_paths,
            args.metadata_path,
            args.dataset_name,
        )
    else:
        msg = f"Unimplemented data source: {args.data_source}"
        raise NotImplementedError(msg)

    dataset.to_file(args.out_file)


def _merge_datasets(args: argparse.Namespace) -> None:
    datasets = [Dataset.from_file(dataset_path) for dataset_path in args.dataset_paths]
    merged_dataset = datasets[0]
    for dataset in datasets[1:]:
        merged_dataset.merge(dataset)

    merged_dataset.to_file(args.out_file)


def _perform_action(args: argparse.Namespace) -> None:
    if args.action == "construct":
        _construct_and_save_dataset(args)
        return
    if args.action == "merge":
        _merge_datasets(args)
        return

    msg = f"Unimplemented action: {args.action}"
    raise NotImplementedError(msg)


def _prepare_construct_parser(construct_parser: argparse.ArgumentParser) -> None:
    construct_parser.set_defaults(action="construct")

    construct_parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name, used to create question ids.",
    )
    construct_parser.add_argument(
        "-o",
        "--out_file",
        default="data/dataset.json",
        type=str,
        help="File to output constructed dataset to",
    )

    subparsers = construct_parser.add_subparsers(title="Data source", required=True)

    csv_subparser = subparsers.add_parser("from_csv")
    csv_subparser.set_defaults(data_source="csv")
    csv_subparser.add_argument(
        "--csvs",
        dest="csv_paths",
        type=str,
        nargs="+",
        required=True,
        help="Csv file from which to construct the dataset",
    )
    csv_subparser.add_argument(
        "--metadata",
        "--metadata_path",
        dest="metadata_path",
        type=str,
        required=True,
        help="Json file with metadata about the columns of the csv",
    )


def _prepare_merge_parser(merge_parser: argparse.ArgumentParser) -> None:
    merge_parser.set_defaults(action="merge")

    merge_parser.add_argument(
        "dataset_paths",
        type=str,
        nargs="+",
        help="Paths of datasets to merges",
    )

    merge_parser.add_argument(
        "-o",
        "--out_file",
        default="data/dataset.json",
        type=str,
        help="File to output merged dataset to",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(title="Action", required=True)
    construct_parser = subparsers.add_parser(
        "construct",
        help="Construct a dataset from some source",
    )
    _prepare_construct_parser(construct_parser)
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge several datasets into one",
    )
    _prepare_merge_parser(merge_parser)

    args = parser.parse_args()
    _perform_action(args)


if __name__ == "__main__":
    main()

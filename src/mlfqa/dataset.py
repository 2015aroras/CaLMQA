from __future__ import annotations

import argparse
import copy
import csv
import enum
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

from models.model import ModelName, PromptParameters
from pydantic import Field as PyField
from pydantic import RootModel, TypeAdapter
from pydantic.dataclasses import dataclass

from mlfqa.language import Language

if TYPE_CHECKING:
    from dataclasses import KW_ONLY

Source = str


logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "[question]"


class QuestionType(enum.IntFlag):
    NONE = 0
    CULTURAL = enum.auto()
    NON_CULTURAL = enum.auto()
    LONG_FORM = enum.auto()
    ALL = NON_CULTURAL | CULTURAL | LONG_FORM


@dataclass
class QuestionID:
    id: int

    @classmethod
    def make(
        cls: type[Self],
        source: Source,
        original_translation: QuestionTranslation,
    ) -> QuestionID:
        original_translation_json = RootModel[QuestionTranslation](
            original_translation,
        ).model_dump_json()
        string_to_hash = "|".join([str(source), original_translation_json])
        return QuestionID(int(hashlib.sha256(string_to_hash.encode()).hexdigest(), 16))


@dataclass
class AnswerID:
    id: int

    @classmethod
    def make(
        cls: type[Self],
        source: Source,
        prompt: str,
        language: Language,
        option_probs: dict[str, float] | None,
    ) -> AnswerID:
        string_to_hash = "|".join(
            [str(source), prompt, language.name, f"option_probs={option_probs}"],
        )
        return AnswerID(int(hashlib.sha256(string_to_hash.encode()).hexdigest(), 16))


@dataclass(frozen=True)
class QuestionTranslation:
    language: Language
    title: str | None = None
    elaboration: str | None = None
    text: str | None = None

    def get_text(self) -> str:
        if self.text is not None:
            return self.text

        if self.title is not None:
            if self.elaboration is not None:
                return f"{self.title}\n{self.elaboration}"

            return self.title

        msg = "Question translation has neither title not text"
        raise RuntimeError(msg)


@dataclass(frozen=True)
class Question:
    type: QuestionType
    source: Source
    language: Language
    translations: dict[Language, QuestionTranslation]
    url: str | None = PyField(default=None)
    q_id: QuestionID = PyField(default_factory=lambda: QuestionID(-1))

    def __post_init__(self) -> None:
        generated_hash = QuestionID.make(self.source, self.untranslated)

        if self.q_id.id == -1:
            object.__setattr__(
                self,
                "q_id",
                generated_hash,
            )
        elif self.q_id != generated_hash:
            msg = f"Question id {self.q_id} does not match expected id {generated_hash}"
            raise ValueError(msg)

    @property
    def untranslated(self) -> QuestionTranslation:
        return self.translations[self.language]


@dataclass(frozen=True)
class AnswerTranslation:
    language: Language
    text: str


@dataclass(frozen=True)
class Answer:
    prompt_parameters: PromptParameters
    language: Language
    translations: list[AnswerTranslation]
    _: KW_ONLY
    source: Source | None = PyField(default=None)
    """Deprecated in favor of `prompt_parameters`"""
    prompt: str | None = PyField(default=None)
    """Deprecated in  favor of `prompt_parameters`. What the source was prompted with to get this answer"""  # noqa: E501
    option_probs: dict[str, float] | None = PyField(default=None)

    @classmethod
    def make(
        cls: type[Self],
        prompt_parameters: PromptParameters,
        language: Language,
        text: str,
        *,
        option_probs: dict[str, float] | None = None,
    ) -> Answer:
        """`Create an answer without any translations."""
        return cls(
            prompt_parameters,
            language,
            [AnswerTranslation(language, text)],
            option_probs=option_probs,
        )

    @classmethod
    def make_human_answer(
        cls: type[Self],
        prompt: str,
        language: Language,
        text: str,
        *,
        option_probs: dict[str, float] | None = None,
    ) -> Answer:
        """Create a human's answer without any translations."""
        prompt_parameters = PromptParameters(prompt, ModelName.HUMAN, -1)

        return cls(
            prompt_parameters,
            language,
            [AnswerTranslation(language, text)],
            option_probs=option_probs,
        )

    @property
    def untranslated(self) -> AnswerTranslation:
        untranslated_answers = [
            translation
            for translation in self.translations
            if translation.language == self.language
        ]
        assert len(untranslated_answers) == 1
        return untranslated_answers[0]


@dataclass
class Dataset:
    entries: list[Dataset.Entry]
    default_save_path: str | None = PyField(default=None, exclude=True)

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
        matching_entries = [entry for entry in self.entries if entry.question.q_id == question.q_id]
        assert len(matching_entries) <= 1, f"More than 1 entry for question {question.q_id}"
        assert len(matching_entries) >= 1, f"No entry for a question {question.q_id}"
        return matching_entries[0]

    def get_answers(
        self,
        question: Question,
        source: Source | None = None,
        prompt: str | None = None,
        language: Language | None = None,
    ) -> list[Answer]:
        entry = self._get_entry(question)

        return [
            copy.deepcopy(answer)
            for answer in entry.answers
            if (source is None or source == answer.source)
            and (prompt is None or prompt == answer.prompt)
            and (language is None or language == answer.language)
        ]

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
            entry_answer
            for entry_answer in entry.answers
            if entry_answer.prompt_parameters == answer.prompt_parameters
        ]
        assert (
            len(existing_answers) <= 1
        ), f"Too many answers match answer with prompt parameters: {answer.prompt_parameters}"

        if len(existing_answers) == 1:
            existing_answer = existing_answers[0]
            entry.answers = [
                entry_answer
                for entry_answer in entry.answers
                if entry_answer.prompt_parameters != existing_answer.prompt_parameters
            ]

        entry.answers.append(copy.deepcopy(answer))

    def to_file(self, save_file_path: str | None = None) -> None:
        save_file_path = save_file_path or self.default_save_path
        if not save_file_path:
            msg = "No dataset save file path provided and no default is set"
            raise ValueError(msg)

        dataset_json = RootModel[Dataset](self).model_dump_json(indent=2)
        Path(save_file_path).write_text(dataset_json)

    @classmethod
    def from_file(cls: type[Self], load_file_path: str) -> Dataset:
        dataset_json = Path(load_file_path).read_text()
        return TypeAdapter(Dataset).validate_json(dataset_json)

    def _merge_entries(self, entry: Dataset.Entry, other_entry: Dataset.Entry) -> None:
        assert entry.question.language == other_entry.question.language
        assert entry.question.q_id == other_entry.question.q_id
        assert entry.question.source == other_entry.question.source
        assert entry.question.type == other_entry.question.type
        assert entry.question.url == other_entry.question.url
        entry.question.translations.update(other_entry.question.translations)

        msg = "Answer merging not yet implemented"
        raise NotImplementedError(msg)

    def merge(self, other: Dataset) -> None:
        self_entries_by_q_id = {entry.question.q_id.id: entry for entry in self.entries}

        for other_entry in other.entries:
            if other_entry.question.q_id.id not in self_entries_by_q_id:
                self.entries.append(copy.deepcopy(other_entry))
            else:
                self_entry = self_entries_by_q_id[other_entry.question.q_id.id]
                self._merge_entries(self_entry, other_entry)


class Field(enum.Enum):
    IS_CULTURAL = enum.auto()
    LANGUAGE = enum.auto()
    QUESTION_WITH_ELABORATION = enum.auto()
    QUESTION_TITLE = enum.auto()
    QUESTION_ELABORATION = enum.auto()
    QUESTION_SOURCE = enum.auto()
    ANSWER = enum.auto()
    ANSWER_SOURCE = enum.auto()
    URL = enum.auto()


def _construct_dataset_from_dict_entries(
    dict_entries: list[dict[Field, Any]],
) -> Dataset:
    entries: list[Dataset.Entry] = []

    for dict_entry in dict_entries:
        question_type = QuestionType.LONG_FORM

        if not isinstance(dict_entry[Field.IS_CULTURAL], bool):
            msg = f"Is cultural is not a boolean: {dict_entry[Field.IS_CULTURAL]}"
            raise TypeError(msg)
        is_cultural_question = dict_entry[Field.IS_CULTURAL]

        question_type |= (
            QuestionType.CULTURAL if is_cultural_question else QuestionType.NON_CULTURAL
        )

        language = Language[dict_entry[Field.LANGUAGE]]

        if Field.QUESTION_WITH_ELABORATION in dict_entry:
            question_translation = QuestionTranslation(
                language,
                text=dict_entry[Field.QUESTION_WITH_ELABORATION],
            )
        elif Field.QUESTION_TITLE in dict_entry:
            question_translation = QuestionTranslation(
                language,
                title=dict_entry[Field.QUESTION_TITLE],
                elaboration=dict_entry.get(Field.QUESTION_ELABORATION),
            )
        else:
            msg = "No question text fields found"
            raise ValueError(msg)

        question = Question(
            question_type,
            dict_entry[Field.QUESTION_SOURCE],
            language,
            {language: question_translation},
            dict_entry[Field.URL],
        )
        answer = Answer.make_human_answer(
            question.untranslated.get_text(),
            language,
            dict_entry[Field.ANSWER],
        )
        entry = Dataset.Entry(question, [answer])
        entries.append(entry)

    return Dataset(entries)


def _construct_dataset_from_csv_and_metadata(csv_path: str, columns_metadata_path: str) -> Dataset:
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

    return _construct_dataset_from_dict_entries(dict_entries)


def _construct_dataset_from_csvs_and_metadata(
    csv_paths: str,
    columns_metadata_path: str,
) -> Dataset:
    return Dataset(
        [
            entry
            for csv_path in csv_paths
            for entry in _construct_dataset_from_csv_and_metadata(
                csv_path,
                columns_metadata_path,
            ).entries
        ],
    )


def _construct_and_save_dataset(args: argparse.Namespace) -> None:
    dataset: Dataset
    if args.data_source == "csv":
        dataset = _construct_dataset_from_csvs_and_metadata(args.csv_paths, args.metadata_path)
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

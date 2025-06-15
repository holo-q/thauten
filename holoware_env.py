import json
import logging
import random
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from datasets import Dataset
from openai import OpenAI
from pydantic import BaseModel
from rich import box
from rich.panel import Panel
from rich.table import Table
from verifiers import Environment, Rubric

import prompts
from log import console
from prompts import format_conversation, PromptInstance, PromptLibrary, rollout_prompt

logger = logging.getLogger(__name__)

# TODO allow multiple evaluation contexts

class CommModel(BaseModel):
    """
    A well-defined data object / schema for communication with a LLM.
    Just a Pydantic BaseModel with some additional nice things for intercommunication.
    """
    @classmethod
    def get_json_schema(cls, indent=2) -> str:
        """
        Returns the JSON schema for the model as a formatted string.
        """
        return json.dumps(cls.model_json_schema(), indent=indent)

    @classmethod
    def get_compact_schema(cls, include_descriptions: bool = True) -> str:
        """
        Generates a compact, human-readable schema from the Pydantic model fields automatically.
        Can optionally include field descriptions as comments.
        """
        from typing import get_args, get_origin, Literal as PyLiteral, List as PyList
        import json

        lines = []
        # We use cls.model_fields which is available on Pydantic models
        for i, (field_name, field_info) in enumerate(cls.model_fields.items()):
            field_type = field_info.annotation
            origin = get_origin(field_type)
            args = get_args(field_type)

            value = None
            if origin is PyLiteral:
                value = "|".join(map(str, args))
            elif origin is list or origin is PyList:
                value = [field_info.description or "list of items"]
            else:
                if field_type is not None and hasattr(field_type, '__name__'):
                    value = field_type.__name__
                elif field_type is not None:
                    value = str(field_type)
                else:
                    value = 'Any'  # Default for untyped fields

            # Use json.dumps to correctly format the value part of the key-value pair
            line = f'  "{field_name}": {json.dumps(value)}'
            if i < len(cls.model_fields) - 1:
                line += ","

            # Add description as comment if requested
            if include_descriptions and field_info.description:
                line += f"  # {field_info.description}"

            lines.append(line)

        return "{\n" + "\n".join(lines) + "\n}"

class VerificationModel(CommModel):
    @abstractmethod
    def get_verification_score(self) -> float:
        pass


class HolowareEnv(Environment):
    """
    Compression environment that generates compression/decompression pairs.

    For each prompt, generates:
    1. Compression rollout: original_content → compressed_form
    2. Decompression rollout: compressed_form → decompressed_content

    Both rollouts receive the same reward based on compression quality + fidelity.
    This works within standard GRPO framework without trainer modifications.
    """

    def __init__(
        self,
        dataset: Dataset,
        path: str,
        score_class: Type[VerificationModel],
        eval_dataset: Optional[Dataset] = None,
        eval_model: str = "Qwen/Qwen2.5-7B-Instruct",
        alpha: float = 0.01,
        beta: float = 1.0,
        base_score: float = 10.0,
        max_concurrent: int = 64,
        dry_run: bool = False,
        **kwargs
    ):
        self.prompt_lib = PromptLibrary()
        self.holoware = self.prompt_lib.load_holoware(path)
        self.score_class = score_class
        self.evaluation_schema = score_class.get_compact_schema(include_descriptions=True)
        self.dry_run = dry_run

        self.alpha = alpha
        self.beta = beta
        self.base_score = base_score

        if not self.dry_run:
            from openai import OpenAI
            self.evaluator_client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="none"
            )
            self.evaluator_model = eval_model
        else:
            self.evaluator_client = None
            self.evaluator_model = None

        if dry_run:
            console.print(self.holoware.to_rich_debug())

        def reward(prompt, completion, answer, state, **kwargs) -> float:
            return state.get("reward", 0.0)

        rubric = Rubric(funcs=[reward], weights=[1.0])
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt="",
            rubric=rubric,
            max_concurrent=max_concurrent,
            message_type='chat',
            **kwargs
        )

    def rollout(self, client: OpenAI, model: str, prompt: str | List[Dict[str, Any]], answer: str, sampling_args=None, **kwargs: Any) -> Tuple[Union[str, List[Dict[str, Any]]], Dict[str, Any]]:
        raise NotImplementedError("This environment uses a custom generate loop, not the standard rollout method.")

    def generate(self,
                 dataset: Dict[str, List[Any]] | Dataset,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 sampling_args=None,
                 max_concurrent: int | None = None,
                 score_rollouts: bool = True,
                 **kwargs: Any) -> Dict[str, List[Any]]:
        """
        Overrides the base generate method to perform a full compression/decompression cycle.
        """
        if sampling_args is None:
            sampling_args = {}

        res = {
            "original":     [],
            "compressed":   [],
            "decompressed": [],
            "evaluation":   [],
            "fidelity":     [],
            "reward":       [],
            "prompt":       [],
            "completion":   [],
            "answer":       [],
            "state":        [],
        }

        # The 'generate' function for the rollout
        def get_generation(messages):
            return self.get_model_response(messages=messages, client=client, model=model, sampling_args=sampling_args, message_type='chat')

        def get_evaluation(messages):
            return self.get_model_response(messages=messages, client=client, model=model, sampling_args=sampling_args, message_type='chat')

        for sample in dataset:
            original = sample.get('text', '')

            unrolled = rollout_prompt(self.holoware, get_generation, env={
                "input":        original,
                "original":     original,
                "score-format": self.evaluation_schema,
            })
            compressed = unrolled.extract_fence("compress") or ""
            decompressed = unrolled.extract_fence("decompress") or ""

            # The last context contains the evaluation
            raw_evaluation_str = format_conversation(unrolled.contexts[-1].messages)
            eval_json = prompts.extract_json(raw_evaluation_str)
            score = random.uniform(0, 5)
            verif_obj = None

            if eval_json:
                try:
                    verif_obj = self.score_class.model_validate_json(eval_json)
                    score = verif_obj.get_verification_score()
                except Exception as e:
                    logger.warning(f"Failed to parse or process evaluation JSON: {e}")

            if self.dry_run:
                console.print(Panel(
                    unrolled.to_rich(),
                    title="[bold yellow]Dry Run: Full Conversation Flow[/]",
                    border_style="yellow",
                    box=box.ROUNDED,
                    title_align="left"
                ))

            token_count = len(compressed.split())
            penalty = self.alpha * token_count + self.beta * (1 - score)
            reward = self.base_score - penalty
            reward = max(0.0, reward)

            # Store results
            res["original"].append(original)
            res["compressed"].append(compressed)
            res["decompressed"].append(decompressed)
            res["fidelity"].append(score)
            res["evaluation"].append(verif_obj.model_dump() if verif_obj else {})
            res["reward"].append(reward)
            res["answer"].append(original)
            res["state"].append({"reward": reward})

        return res

    def get_model_response(self,
                           messages: str | List[Dict[str, str]],
                           client: OpenAI,
                           model: str,
                           sampling_args: Dict[str, Any] = {},
                           message_type: Literal['chat', 'completion'] | None = None,
                           sanitize_sampling_args: bool = True,
                           **kwargs: Any):
        """
        Override the base method to intercept calls during a dry run.
        """
        if self.dry_run:
            sctx = ""
            if isinstance(messages, list):
                sctx = " ".join([msg.get('content', '') for msg in messages])
            elif isinstance(messages, str):
                sctx = messages

            if "compress" in sctx.lower() and "decompress" not in sctx.lower():
                return "compressed-symbols"
            elif "decompress" in sctx.lower():
                return "decompressed-text"
            else:
                # Return a default valid JSON for dry runs
                return f'```json\n{self.score_class(severity="MINOR", quality="EXCELLENT").model_dump_json()}\n```'

        return super().get_model_response(messages, client, model, sampling_args, message_type)


    def log(self, conversation: PromptInstance, fidelity_score: float, reward: float) -> None:
        if self.dry_run:
            console.print(Panel(
                conversation.to_rich(),
                title="[bold yellow]Dry Run: Full Conversation Flow[/]",
                border_style="yellow",
                box=box.ROUNDED,
                title_align="left"
            ))
            return

        # Create a table for beautiful logging
        table = Table(box=box.MINIMAL, show_header=False, expand=True)
        table.add_column(style="bold magenta")
        table.add_column(style="white")

        table.add_row("Fidelity:", f"{fidelity_score:.2f}")
        table.add_row("Reward:", f"{reward:.2f}")
        # table.add_row("Evaluation:", evaluation)

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row(Panel(
            table,
            title=f"[bold green]Compression Sample[/]",
            border_style="green",
            title_align="left"
        ))
        console.print(grid)


    def format_dataset(self,
                       dataset: Dataset,
                       system_prompt: str | None = None,
                       few_shot: List[Dict[str, Any]] | None = None,
                       question_key: str = "text",
                       answer_key: str = "answer") -> Dataset:
        return dataset

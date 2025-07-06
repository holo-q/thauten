import random
from typing import List, Literal

class FidelityCritique(CommModel):
    """
    A structured representation of a fidelity evaluation, comparing an original
    text with its decompressed counterpart.
    """

    deviations: List[str] = Field(default=[], description="List of factual changes or alterations from the original.")
    inaccuracies: List[str] = Field(default=[], description="List of incorrect information introduced in the decompressed text.")
    missing_statements: List[str] = Field(default=[], description="List of important information from the original that was lost.")
    acceptable_extensions: List[str] = Field(default=[], description="List of valid elaborations or extensions that don't contradict the original.")
    severity: Literal["MINOR", "MODERATE", "MAJOR"] = Field(description="The overall severity of the issues found.")
    quality: Literal["EXCELLENT", "GOOD", "FAIR", "POOR"] = Field(description="The overall quality of the decompression.")

    @property
    def total_issues(self) -> int:
        """Calculates the total number of issues found."""
        return len(self.deviations) + len(self.inaccuracies) + len(self.missing_statements)

class FidelityAttractor(Attractor):
    def __init__(self, a, b, alpha=0.01, beta=1.0):
        super().__init__()
        self.a = a
        self.b = b
        self.alpha= alpha
        self.beta = beta

        # tb.add_row("Alpha (compression)", f"{alpha}")
        # tb.add_row("Beta (fidelity)", f"{beta}")

        self.add_rule(self.rule_fidelity)

    def rule_fidelity(self, rollout):
        n_tokens = len(self.a.split())
        fidelity = random.uniform(0, 5)

        decel = self.alpha * n_tokens + self.beta * (1 - fidelity)
        gravity = self.base_score - decel
        gravity = max(0.0, gravity)

        return gravity


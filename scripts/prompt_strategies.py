import random
from typing import Dict, List


class PromptSelector:
    def __init__(self, strategy: str = 'concise'):
        self.strategy = strategy

        self.strategy_preferences = {
            'concise': ['p1'],
            'detailed': ['p7', 'p6', 'p5', 'p4', 'p3', 'p2', 'p1'],
        }

        if strategy not in self.strategy_preferences:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(self.strategy_preferences.keys())}"
            )

    def select_prompt(self, annotation: Dict) -> str:
        prompts = annotation.get('prompts', {})

        if not prompts:
            raise ValueError(f"No prompts found in annotation: {annotation}")

        preferred_keys = self.strategy_preferences[self.strategy]

        for key in preferred_keys:
            if key in prompts and prompts[key]:
                prompt = prompts[key]
                return prompt[0] if isinstance(prompt, list) else prompt

        raise ValueError(
            f"No prompt found for strategy '{self.strategy}'. "
            f"Available keys: {list(prompts.keys())}"
        )

    def get_available_strategies(self) -> List[str]:
        return list(self.strategy_preferences.keys())
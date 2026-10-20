import re

from balrog.agents.base import BaseAgent


class CustomAgent(BaseAgent):
    """An agent that generates actions and plans based on observations with a customizable approach."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.plan = None

    def act(self, obs, prev_action=None):
        """Generate the next action and update the plan based on the current observation.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the updated plan and selected action.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        plan_text = f"Current Plan:\n{self.plan}\n" if self.plan else "You have no plan yet.\n"

        planning_instructions = """
Review the current plan above if present.
Think step by step: evaluate what happened from your previous action, analyze the current observation, and decide whether to update the plan or keep it.
Finally, determine the best next action.

Format your answer strictly in the following way:
THOUGHT: <your step-by-step reasoning about the observation and situation>
PLAN: <write your new plan, or "No changes to the plan." ONLY if you already have a plan above>
ACTION: <your next action>
        """.strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + plan_text + "\n" + planning_instructions

        response = self.client.generate(messages)

        # Extract the plan and action from the LLM's response
        thought, plan, action = self._extract_plan_and_action(response.completion)
        
        # Update the internal plan if it has changed
        is_no_changes = "no changes" in plan.lower()
        
        if plan and not is_no_changes:
            self.plan = plan

        # Save the plan in the response.reasoning field and the action in response.completion
        # Combine thought and plan into reasoning, and set action as completion
        combined_reasoning = f"Thought: {thought}\nPlan: {plan}" if thought else plan
        response = response._replace(reasoning=combined_reasoning, completion=action)
        return response

    def _extract_plan_and_action(self, response_text):
        """Extract the plan and action from the LLM's response text.

        Args:
            response_text (str): The response text from the LLM.

        Returns:
            tuple: A tuple containing the extracted plan and action.
        """
        thought_match = re.search(
            r"THOUGHT:\s*(.*?)(?=\nPLAN:|\Z)", response_text, re.IGNORECASE | re.DOTALL
        )
        plan_match = re.search(r"PLAN:\s*(.*?)(?=\nACTION:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r"ACTION:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        plan = plan_match.group(1).strip() if plan_match else "No changes to the plan."
        action = action_match.group(1).strip() if action_match else None

        return thought, plan, action
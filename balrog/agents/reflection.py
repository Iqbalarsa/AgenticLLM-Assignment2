import atexit
import json
import re
from pathlib import Path

from balrog.agents.base import BaseAgent
from balrog.prompt_builder.history import Message   # check the import path in your version


class ReflexionAgent(BaseAgent):
    MEMORY_PATH = Path("/local/s4859049/BALROG/reflection_memory.json")
    MAX_REFLECTIONS = 5     # how many lessons to inject
    MAX_HIST_SIZE = 10      # cap on steps shown to the reflecto

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.plan = None
        self.trajectory = []    # [(obs_text, thought, plan, action)]
        self.reflections = self._load_memory()
        atexit.register(self._cleanup_at_exit)

    def _cleanup_at_exit(self):
        if hasattr(self, "trajectory") and self.trajectory:
            self.end_episode(outcome=None)

    def _load_memory(self):
        try:
            return json.loads(self.MEMORY_PATH.read_text())
        except Exception:
            return []

    def _save_memory(self):
        # Open adn read data from previous episodes
        disk_lessons = self._load_memory()
        
        # Track existing lesson
        existing_texts = {r["lesson"] for r in disk_lessons if isinstance(r, dict) and "lesson" in r}
        
        # 3. Append only new reflections generated in the current episode
        for r in self.reflections:
            if isinstance(r, dict) and r.get("lesson") and r["lesson"] not in existing_texts:
                disk_lessons.append(r)
                existing_texts.add(r["lesson"])
                
        # 4. Save the merged reflections back to disk
        self.reflections = disk_lessons
        self.MEMORY_PATH.write_text(json.dumps(self.reflections[-30:], indent=2))

    def _reflection_text(self):
        if not self.reflections:
            return ""
        lessons = self.reflections[-self.MAX_REFLECTIONS:]
        body = "\n".join(f"- {r['lesson']}" for r in lessons if isinstance(r, dict) and "lesson" in r)
        return f"Lessons learned from previous attempts (apply them!):\n{body}\n\n"

    def act(self, obs, prev_action=None):
        # prev_action is None on the first step of an episode
        if prev_action is None and self.trajectory:
            # new episode started without an explicit end_episode() call
            self.end_episode(outcome=None)

        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        plan_text = f"Current Plan:\n{self.plan}\n" if self.plan else "You have no plan yet.\n"
        instructions = """
Review the lessons and the current plan above if present.
Think step by step: evaluate what happened from your previous action, analyze the current
observation, and decide whether to update the plan or keep it. Then pick the best next action.

Format your answer strictly in the following way:
THOUGHT: <your step-by-step reasoning>
PLAN: <your new plan, or "No changes to the plan." ONLY if you already have a plan above>
ACTION: <your next action>
""".strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += (
                "\n\n" + self._reflection_text() + plan_text + "\n" + instructions
            )

        response = self.client.generate(messages)
        thought, plan, action = self._extract_plan_and_action(response.completion)

        if plan and "no changes" not in plan.lower():
            self.plan = plan

        obs_text = obs["text"]["long_term_context"] if isinstance(obs, dict) and "text" in obs else str(obs)
        self.trajectory.append((obs_text, thought, plan, action))

        reasoning = f"Thought: {thought}\nPlan: {plan}" if thought else plan
        return response._replace(reasoning=reasoning, completion=action)

    def end_episode(self, outcome=None):
        """Call when an episode ends. outcome: dict like {'reward':..., 'done':..., 'progression':...}"""
        if not self.trajectory:
            return

        steps = self.trajectory[-self.MAX_HIST_SIZE:]
        traj_str = "\n".join(
            f"[{i}] OBS: {o[:120]}\n    THOUGHT: {t[:100]}\n    ACTION: {a}"
            for i, (o, t, _, a) in enumerate(steps)
        )
        outcome_str = json.dumps(outcome) if outcome else "unknown (episode ended)"

        prompt = f"""You are reviewing a failed or incomplete attempt at a game.
Outcome: {outcome_str}
Final plan: {self.plan}

Trajectory (last {len(steps)} steps):
{traj_str}

Diagnose the main mistakes (wrong assumptions, repeated useless actions, ignored
observations, bad plans). Then write 1-3 concise, actionable, GENERAL lessons for the next
attempt. Do not describe specific coordinates or one-off events.

Format:
LESSON: <lesson 1>
LESSON: <lesson 2>"""

        resp = self.client.generate([Message(role="user", content=prompt, attachment=None)])
        lessons = re.findall(r"LESSON:\s*(.+)", resp.completion)

        for l in lessons:
            self.reflections.append({"lesson": l.strip(), "outcome": outcome})
        self._save_memory()

        # reset per-episode state, keep memory
        self.trajectory = []
        self.plan = None

    def _extract_plan_and_action(self, response_text):
        thought_match = re.search(
            r"THOUGHT:\s*(.*?)(?=\nPLAN:|\Z)", response_text, re.IGNORECASE | re.DOTALL
        )
        plan_match = re.search(r"PLAN:\s*(.*?)(?=\nACTION:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r"ACTION:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        plan = plan_match.group(1).strip() if plan_match else "No changes to the plan."
        action = action_match.group(1).strip() if action_match else None

        return thought, plan, action
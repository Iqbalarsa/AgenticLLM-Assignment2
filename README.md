# Agentic LLM: Reasoning and Self-Reflection in BALROG

This repository contains the code and experimental setup for **Assignment 2: Reasoning** in the Agentic LLM course. In this project, we study the reasoning and self-reflection abilities of Large Language Models (LLMs) on interactive puzzle and game benchmarks using the **[BALROG](https://balrogai.com)** framework.

The project is divided into three key stages:
1. **Exercise 1: Baseline Benchmark Evaluation** — Running baseline evaluations with a naive agent on local models (`Qwen2.5-7B`, `SmolLM3-3B`) and cloud models (`Gemini-3.5-Flash-Lite`) across environments such as **BabyAI** and **Crafter**.
2. **Exercise 2: Custom Reasoning Agent** — Improving reasoning capabilities by developing a custom agent using Chain-of-Thought (CoT) prompting.
3. **Exercise 3: Self-Reflection Agent** — Implementing an agent based on the Reflexion architecture that extracts lessons from failed attempts across episodes.

---

## 1. Local LLM Server Setup (vLLM)

To run experiments with locally hosted open-source models, launch an OpenAI-compatible server using `vllm` in your first terminal:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager
```

> **Model Switch:** You can replace the model path with another model if needed, such as `HuggingFaceTB/SmolLM3-3B`.

In your evaluation terminal, export the environment variables so that the client routes requests to the local server:

```bash
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="http://localhost:8000/v1"
```

---

## 2. Automated Experiments Script

To run evaluations conveniently across 3 separate runs (10 episodes each), you can execute the provided shell script:

```bash
bash exp_1.sh
```

---

## 3. Exercise 1: Baseline Benchmark Evaluation

In Exercise 1, we establish the baseline performance using the standard `naive` agent.

### Run Command Example:

```bash
python eval.py \
  envs.names=babyai \
  tasks.babyai_tasks=['BabyAI-MixedTrainLocal-v0/pickup'] \
  eval.num_workers=10 \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_text_history=5 \
  client.client_name=openai \
  client.model_id="Qwen/Qwen2.5-7B-Instruct-AWQ" \
  client.generate_kwargs.max_tokens=512
```

- **`envs.names`**: Environment name (`babyai` or `crafter`).
- **`tasks.babyai_tasks`**: Specific tasks to evaluate (e.g., `pickup`, `goto`, `open`).
- **`agent.type`**: Set to `naive`.
- **`eval.num_workers`**: Number of parallel workers used for evaluation.

---

## 4. Exercise 2: Custom Reasoning Agent (Chain-of-Thought)

For Exercise 2, we implement a custom agent that introduces structured reasoning. Instead of predicting the next action directly, the model is prompted to think step-by-step:
1. **THOUGHT**: Evaluates previous actions and current observations.
2. **PLAN**: Maintains and updates the high-level plan.
3. **ACTION**: Decides on the next concrete executable action.

- **Agent Implementation:** Located in `agents/custom/`

### Run Command:

Run the same evaluation script with `agent.type=custom`:

```bash
python eval.py \
  envs.names=babyai \
  tasks.babyai_tasks=['BabyAI-MixedTrainLocal-v0/pickup'] \
  eval.num_workers=10 \
  agent.type=custom \
  agent.max_image_history=0 \
  agent.max_text_history=5 \
  client.client_name=openai \
  client.model_id="Qwen/Qwen2.5-7B-Instruct-AWQ" \
  client.generate_kwargs.max_tokens=512
```

---

## 5. Exercise 3: Self-Reflection Agent (Reflexion)

In Exercise 3, we implement a self-reflection mechanism based on the **Reflexion** framework. When an episode fails or remains incomplete, the agent reviews the final outcome and its recent trajectory to generate concise, actionable `LESSON` statements. These lessons are stored and injected into the prompt of subsequent attempts.

- **Agent Implementation:** Located in `agents/reflection/`

### Important Setup Note:
> **Set `eval.num_workers=1`:** Because the agent saves lessons and updates shared memory sequentially after each episode, running with a single worker is strongly recommended to prevent memory collisions and race conditions.

### Run Command:

Run the evaluation script with `agent.type=ref`:

```bash
python eval.py \
  envs.names=babyai \
  tasks.babyai_tasks=['BabyAI-MixedTrainLocal-v0/pickup'] \
  eval.num_workers=1 \
  agent.type=ref \
  agent.max_image_history=0 \
  agent.max_text_history=5 \
  client.client_name=openai \
  client.model_id="Qwen/Qwen2.5-7B-Instruct-AWQ" \
  client.generate_kwargs.max_tokens=512
```

---

## 6. Repository Structure

```text
├── agents/
│   ├── custom/        # Implementation for Exercise 2 (CoT Agent)
│   └── reflection/    # Implementation for Exercise 3 (Reflexion Agent)
├── exp_1.sh           # Shell script to automate multiple experiment runs
└── README.md          # Project documentation
```

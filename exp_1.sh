#!/bin/sh

#BabyAI
for i in 1 2 3; do
  python eval.py \
    envs.names=babyai \
    tasks.babyai_tasks="['BabyAI-MixedTrainLocal-v0/goto', 'BabyAI-MixedTrainLocal-v0/open', 'BabyAI-MixedTrainLocal-v0/pickup']" \
    agent.type=custom \ 
    eval.num_workers=5 \
    agent.max_image_history=0 \
    agent.max_text_history=5 \
    client.client_name=openai \
    client.model_id="HuggingFaceTB/SmolLM3-3B" \
    client.generate_kwargs.max_tokens=512
done

#Crafter
for i in 1 2 3; do
  python eval.py \
    envs.names=crafter \
    agent.type=custom \
    eval.num_workers=5 \
    agent.max_image_history=0 \
    agent.max_text_history=5 \
    client.client_name=openai \
    client.model_id="HuggingFaceTB/SmolLM3-3B" \
    client.generate_kwargs.max_tokens=512
done
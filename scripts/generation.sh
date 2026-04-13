unset ROCR_VISIBLE_DEVICES

export model_path="0k9d0h1/7b-planner-1.5b-reranker-nq-hotpotqa-filtered-tp-reranker"
export data_path="/home/kdh0901/Reranker/data/planner/test_2000.parquet"
export base_save_path="/home/kdh0901/Reranker/validation_data/verbalR3_7b_tts/generation_test"

srun --mpi=pmi2 python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    data.shuffle=False \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$current_save_path \
    data.batch_size=1024 \
    model.path=$model_path \
    model.trust_remote_code=True \
    rollout.temperature=1 \
    rollout.top_k=-1 \
    rollout.top_p=0.95 \
    rollout.seed=0 \
    rollout.max_model_len=4096 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=3072 \
    data.max_obs_length=500 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.95 \
    max_turns=4 \
    reranker.url="http://localhost:8005/rerank/batch" \
    reranker.topk=3 \
    reranker.retriever_initial_topk=15 \
    reranker.return_full_documents=false \
    reranker.size="3b" \
    do_search=True \
    output_format="json" \
    score_tts=true \
    tts_n=5 \
    tts_strat="branch_bernoulli" \
    output_dir="/home/kdh0901/Reranker/validation_data/verbalR3_7b_tts"

echo "========================================================="
echo "All experiments complete."
echo "========================================================="

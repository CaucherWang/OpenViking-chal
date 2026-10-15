- [Ingest]
nohup python -m benchmark.aml.personamem-v2.ingest_personamem_v2_history \
  --experiment-name 255k-Max-Token-OncePerPersona \
  --persona-splits benchmark \
  --commit-mode once-per-persona \
  --timeout 900 \
  --extraction-timeout 900 \
  > benchmark/aml/data/results/ingest_255k-Max-Token-OncePerPersona.log 2>&1 &

nohup python -m benchmark.aml.personamem-v2.ingest_personamem_v2_history \
  --experiment-name 255k-Max-Token-official \
  --persona-splits benchmark \
  --max-chunk-messages 20 \
  --timeout 900 \
  --extraction-timeout 900 \
  > benchmark/aml/data/results/ingest_255k-Max-Token-official.log 2>&1 &

- [Ingest-fail-retry]
nohup python -m benchmark.aml.personamem-v2.ingest_personamem_v2_history \
  --experiment-name 255k-Max-Token-official \
  --persona-ids 209 208 136 101 213 218 \
  --existing-policy overwrite \
  --concurrency 3 \
  --timeout 900 \
  --extraction-timeout 1800 \
  > benchmark/aml/data/results/retry_ingest_255k-Max-Token-official.log 2>&1 &


- [Evaluate-expr]
nohup python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_search \
  --experiment-name official \
  --output benchmark/aml/data/results/personamem_v2_search_official_mcq_32k_full.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_search_official_mcq_32k_full_summary.json \
  > benchmark/aml/data/results/personamem_v2_search_official_mcq_32k_full.log 2>&1 &

nohup python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_search \
  --experiment-name official \
  --answer-backend gpt-4o-mini \
  --output benchmark/aml/data/results/personamem_v2_search_official_gpt4o_mini_mcq_32k_full.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_search_official_gpt4o_mini_mcq_32k_full_summary.json \
  > benchmark/aml/data/results/personamem_v2_search_official_gpt4o_mini_mcq_32k_full.log 2>&1 &

nohup python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_search \
  --experiment-name once-per-persona \
  --answer-backend gpt-4o-mini \
  --output benchmark/aml/data/results/personamem_v2_search_once-per-persona_gpt4o_mini_mcq_32k_full.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_search_once-per-persona_gpt4o_mini_mcq_32k_full_summary.json \
  > benchmark/aml/data/results/personamem_v2_search_once-per-persona_gpt4o_mini_mcq_32k_full.log 2>&1 &

nohup python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_search \
  --experiment-name 255k-Max-Token-OncePerPersona \
  --answer-backend seed \
  --output benchmark/aml/data/results/personamem_v2_search_255k_once_per_persona_mcq_32k_full.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_search_255k_once_per_persona_mcq_32k_full_summary.json \
  > benchmark/aml/data/results/personamem_v2_search_255k_once_per_persona_mcq_32k_full.log 2>&1 &

nohup .venv/bin/python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_search \
  --experiment-name 255k-Max-Token-OncePerPersona \
  --answer-backend gpt-4o-mini \
  --output benchmark/aml/data/results/personamem_v2_search_255k_once_per_persona_gpt4o_mini_mcq_32k_full.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_search_255k_once_per_persona_gpt4o_mini_mcq_32k_full_summary.json \
  > benchmark/aml/data/results/personamem_v2_search_255k_once_per_persona_gpt4o_mini_mcq_32k_full.log 2>&1 &

nohup python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_search \
  --experiment-name 255k-Max-Token-official \
  --top-k 10 \
  --output benchmark/aml/data/results/personamem_v2_search_255k_official_seed_topk10.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_search_255k_official_seed_topk10_summary.json \
  > benchmark/aml/data/results/personamem_v2_search_255k_official_seed_topk10.log 2>&1 &

nohup .venv/bin/python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_search \
  --experiment-name 255k-Max-Token-official \
  --answer-max-tokens 5000 \
  --top-k 5 \
  --parallel 35 \
  --output benchmark/aml/data/results/personamem_v2_search_255k_official_seed_topk5.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_search_255k_official_seed_topk5_summary.json \
  > benchmark/aml/data/results/personamem_v2_search_255k_official_seed_topk5.log 2>&1 &

nohup python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_search \
  --experiment-name 255k-Max-Token-official \
  --answer-backend gpt-5.5 \
  --parallel 5 \
  --output benchmark/aml/data/results/personamem_v2_search_255k_official_gpt5_5_topk100.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_search_255k_official_gpt5_5_topk100_summary.json \
  > benchmark/aml/data/results/personamem_v2_search_255k_official_gpt5_5_topk100.log 2>&1 &

nohup bash -c '
set -e
for k in 5 10 20; do
  python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_search \
    --experiment-name 255k-Max-Token-official \
    --answer-backend gpt-4o-mini \
    --top-k "$k" \
    --output "benchmark/aml/data/results/personamem_v2_search_255k_official_gpt4o_mini_topk${k}.csv" \
    --summary-output "benchmark/aml/data/results/personamem_v2_search_255k_official_gpt4o_mini_topk${k}_summary.json"
done
' > benchmark/aml/data/results/personamem_v2_search_255k_official_gpt4o_mini_topk_5_10_20.log 2>&1 &

nohup bash -c '
set -e
for k in 5 10 20; do
  python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_search \
    --experiment-name 255k-Max-Token-official \
    --answer-backend gpt-5.5 \
    --answer-max-tokens 5000 \
    --top-k "$k" \
    --retry-delay 5 \
    --output "benchmark/aml/data/results/personamem_v2_search_255k_official_gpt5_5_topk${k}.csv" \
    --summary-output "benchmark/aml/data/results/personamem_v2_search_255k_official_gpt5_5_topk${k}_summary.json"
done
' > benchmark/aml/data/results/personamem_v2_search_255k_official_gpt5_5_topk_5_10_20.log 2>&1 &

- [Evaluate-fail-retry]
nohup python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_search \
  --retry-errors \
  --experiment-name 255k-Max-Token-OncePerPersona \
  --parallel 2 \
  --output benchmark/aml/data/results/personamem_v2_search_255k_once_per_persona_mcq_32k_full.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_search_255k_once_per_persona_mcq_32k_full_summary.json \
  > benchmark/aml/data/results/retry_personamem_v2_search_255k_once_per_persona_mcq_32k_full.log 2>&1 &

- [Evaluate-ORACLE]
nohup python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_full_history \
  --answer-backend gpt-4o-mini \
  --output benchmark/aml/data/results/personamem_v2_full_history_gpt4o_mini_mcq_32k.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_full_history_gpt4o_mini_mcq_32k_summary.json \
  > benchmark/aml/data/results/personamem_v2_full_history_gpt4o_mini_mcq_32k.log 2>&1 &

nohup python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_full_history \
  --answer-backend gpt-5.5 \
  --answer-max-tokens 5000 \
  --parallel 25 \
  --output benchmark/aml/data/results/personamem_v2_full_history_gpt5_5_mcq_32k.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_full_history_gpt5_5_mcq_32k_summary.json \
  > benchmark/aml/data/results/personamem_v2_full_history_gpt5_5_mcq_32k.log 2>&1 &


- [Evaluate-ORACLE-fail-retry]
nohup python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_full_history \
  --retry-errors \
  --answer-backend gpt-4o-mini \
  --parallel 10 \
  --output benchmark/aml/data/results/personamem_v2_full_history_gpt4o_mini_mcq_32k.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_full_history_gpt4o_mini_mcq_32k_summary.json \
  > benchmark/aml/data/results/retry_personamem_v2_full_history_gpt4o_mini_mcq_32k.log 2>&1 &

nohup python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_full_history \
  --retry-errors \
  --parallel 25 \
  --output benchmark/aml/data/results/personamem_v2_full_history_mcq_32k.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_full_history_mcq_32k_summary.json \
  > retry_full_history_mcq_32k.log 2>&1 &

nohup python -m benchmark.aml.personamem-v2.evaluate_personamem_v2_full_history \
  --retry-errors \
  --answer-backend gpt-5.5 \
  --answer-max-tokens 5000 \
  --parallel 15 \
  --retry-delay 5 \
  --output benchmark/aml/data/results/personamem_v2_full_history_gpt5_5_mcq_32k.csv \
  --summary-output benchmark/aml/data/results/personamem_v2_full_history_gpt5_5_mcq_32k_summary.json \
  > benchmark/aml/data/results/retry_personamem_v2_full_history_gpt5_5_mcq_32k.log 2>&1 &

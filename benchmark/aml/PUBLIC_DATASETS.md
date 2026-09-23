# Public dataset coverage

This inventory records the public upstream sources used by
`benchmark.aml.public_datasets`. The official Agent Memory Leaderboard scored
bundle is not publicly downloadable.

| Adapter group | Public source | Local coverage | Fidelity |
| --- | --- | --- | --- |
| LoCoMo | https://github.com/snap-research/locomo | Original release | Public upstream |
| LoCoMo-Refined | https://github.com/mem-eval-suite/LoCoMo_refined | Conversations, questions, media metadata | Public upstream |
| LongMemEval | https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned | S, M, and oracle variants | Public upstream |
| PersonaMem | https://huggingface.co/datasets/bowen-upenn/PersonaMem | 32K, 128K, and 1M contexts/questions | Public upstream |
| PersonaMem-v2 | https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2 | Text benchmark queries grouped by persona; 32k and 128k history-size modes; MCQ and generative eval modes | Public upstream |
| BEAM | https://huggingface.co/datasets/Mohammadta/BEAM and https://huggingface.co/datasets/Mohammadta/BEAM-10M | 100K, 500K, 1M, and 10M | Public upstream |
| ScriptMem | https://github.com/memorax-ai/ScriptMem | Questions, golds, and public synthetic placeholder | Partial public release |
| CL-bench | https://huggingface.co/datasets/tencent/CL-bench | General release | AML preprocessing approximation |
| CL-bench Life | https://huggingface.co/datasets/tencent/CL-bench-Life | Life release | AML preprocessing approximation |
| Mem-Gallery | https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery | Full released media/dialog data | Text-caption proxy |
| ATM-Bench | https://huggingface.co/datasets/Jingbiao/ATM-Bench | Full released memory bank and questions | Text-caption/OCR proxy |
| SWEContextBench | https://huggingface.co/datasets/jiayuanz3/SWEContextBench | Full five-table release | Public coding-memory proxy |

Run `.venv/bin/python -m benchmark.aml.public_datasets inventory` for the local
file counts, sizes, and readiness state.

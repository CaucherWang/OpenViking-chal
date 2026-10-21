# Top20 开源合规规则召回实验

本实验只回答一个问题：同一份 40 条开源合规规则走普通 OpenViking 用户记忆链路，
是否会比专用 Rule 编译与检索链路产生更多漏入库、漏召回或漏提交。

## 固定输入

- 规则集：`.openviking/experiments/rules-compliance-20261016/sources/open_source_compliance_rules.md`
- 数据集：同目录 `repositories/manifest.json` 固定的 volcengine Top20 checkout
- 人工标签：`benchmark/rules_compliance/ground_truth.json`
- 实验 Skill：`examples/skills/openviking-rule-search/SKILL.md`

不重新抓取仓库、排名或规则来源。Ground Truth 只标注 Rule 的触发条件是否由 checkout
中的证据建立，不标注最终合规结论。

## 双臂

| 项目 | main 对照组 | rules 实验组 |
|---|---|---|
| 离线输入 | 整份 Markdown 作为 `peer_id=user` 的用户消息写入 Session，再调用 `session.commit` | 整份 Markdown 直接作为 `rule_compile` source |
| 在线入口 | VikingBot 正常 `_process_message`，触发自动用户记忆召回 | VikingBot 正常 `_process_message` |
| 在线工具 | VikingBot 默认只读工具，禁用 `openviking_rule_search` | VikingBot 默认只读工具 |
| 在线 Skill | 无 | 通过 VikingBot sandbox 原生加载 `openviking-rule-search` |
| 模型 / 本地仓库路径 / 审计与提交要求 | 相同 | 相同 |

脚本不注册工具，也不实现仓库扫描逻辑，只把冻结 checkout 的本地路径交给 VikingBot。
Agent 如何检查仓库完全由 VikingBot 默认能力决定。两组收到完全相同的审计请求，
差异来自离线存储链路、`openviking_rule_search` 是否可用，以及实验组原生加载的通用
Rule 工具使用 Skill；Skill 不拼接到用户 Prompt。

主指标是 Agent 最终输出 Rule ID 的 macro recall 和 micro recall。Precision、F1、每仓
差值、耗时和 Token 为辅助指标。诊断按三阶段拆分：

1. 离线不可用：Session commit 或 Rule compile 后，对应存储中不存在该 Rxx。
2. 在线未暴露：规则已入库，但自动召回与工具结果均未向 Agent 暴露该 Rxx。
3. 暴露未提交：Agent 已看到该 Rxx，但最终 JSON 没有提交。

`runtime/main_memory_inventory.json` 和 `runtime/rules_inventory.json` 保存离线 Rxx
保留情况；每次运行的 `result.json` 保存自动注入的 `relevant_memories` 和完整
`tools_used`，评估脚本直接使用这些原始字段，不生成二次工具事件。Session commit
即使没有生成长期 memory，也会保留原始 archive；评估另行统计 Agent 是否通过 OV
工具发现该 archive，避免把手工兜底误记为自动记忆召回。

## 运行

```bash
.venv/bin/python benchmark/rules_compliance/setup.py start
.venv/bin/python benchmark/rules_compliance/setup.py offline
.venv/bin/python benchmark/rules_compliance/run.py --concurrency 4
.venv/bin/python benchmark/rules_compliance/evaluate.py
```

先对单仓 smoke：

```bash
.venv/bin/python benchmark/rules_compliance/run.py \
  --phase smoke --repos mediakit-cli --concurrency 1
.venv/bin/python benchmark/rules_compliance/evaluate.py \
  --phase smoke --allow-partial
```

新结果写入 `.openviking/experiments/rules-memory-vs-compiled-20261019/`，不会覆盖
上一轮 resource-vs-rules 实验。已存在的 `result.json` 不会被覆盖；需要重跑时只删除
对应仓库和 arm 的运行目录。

# 规则校验（Beta）

把协议、代码规范等材料中的强约束批量编译为规则，再让智能体根据具体任务召回规则、核对证据并调整行动。本教程以用户提供的开源许可证要求为例。

OpenViking 负责规则编译和适用性召回。是否满足或违反规则由智能体判断，搜索接口本身不返回合规结论。概念和生命周期见[规则（Beta）](../concepts/17-rule-scenarios.md)。

## 前置条件

- 运行包含规则功能的 OpenViking 服务端，配置可用的编译模型和向量模型。编译器默认复用服务端的视觉语言模型，也可以通过 `rules.compiler` 指定。
- 使用包含 `rule_compile()` 和 `rule_search()` 的 Python 远程客户端，并配置当前用户的认证身份。编译和搜索应使用同一用户。
- 准备用户确认的强约束材料，保留原文的条件、例外和阈值。不要把普通建议或偏好当作规则。

模型配置见[服务端规则配置](../configuration/01-server.md#rules)，连接约定见[接口总览](../api/01-overview.md)。以下代码按顺序运行，服务地址和接口密钥替换为自己的配置。

## 第一步：提交规则材料

```python
from openviking_sdk import SyncHTTPClient

client = SyncHTTPClient(url="http://localhost:1933", api_key="your-key")
task = client.rule_compile(
    instruction=(
        "仓库声明自己是开源项目时，"
        "必须提供明确、完整且可定位的项目许可证文本。"
    ),
)
print(task["task_id"])
```

批量导入文件或目录时，将上面的调用替换为：

```python
task = client.rule_compile(options={"sources": ["./compliance-rules/"]})
```

客户端会上传本地来源。已导入的资源可以直接使用 `{"sources": ["viking://resources/compliance-rules"]}`，避免重复导入。网络接口的来源格式见[规则接口](../api/26-rules.md)。

## 第二步：等待完成并检查产物

```python
import time


def wait_for_task(task_id, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = client.get_task(task_id)
        if current is None:
            raise RuntimeError(f"任务不存在：{task_id}")
        if current["status"] == "completed":
            return current
        if current["status"] in {"failed", "cancelled"}:
            raise RuntimeError(current)
        time.sleep(1)
    raise TimeoutError(f"任务仍在运行：{task_id}")


completed = wait_for_task(task["task_id"])
result = completed["result"]
print(result["operations"])
print(result["written_uris"])

# 编译完成与写入后的索引处理可能属于不同任务。
write_task_id = result.get("write_result", {}).get("task_id")
if write_task_id:
    wait_for_task(write_task_id)

for uri in result["written_uris"]:
    print(client.read(uri))
```

检查生成规则是否保留了主体、义务、条件和例外，以及原文证据。`UNCERTAIN` 和 `REJECT` 只记录原因；`NOOP` 表示已有规则覆盖输入。超时只停止本地等待，不取消服务端任务。

## 第三步：按具体任务搜索

先读取审查对象，再把目标、对象、计划动作和已有证据组织为 `task_context`。以下内容是示例事实，使用时替换为实际读取的内容：

```python
matches = client.rule_search(
    task_context=(
        "只读检查一个仓库的开源许可证。"
        "已读取 README.md，其中声明项目开源；"
        "根目录存在 LICENSE，尚未检查其正文。"
        "接下来计划读取许可证文本并核对完整性。"
    ),
)
print(matches["rendered"])
print(matches.get("warning"))
```

把完整的 `rendered` 交给智能体；直接对接接口的应用还应检查 `judge`、`warning` 和 `scenario_selection`。判断模型不可用时可能放行候选，不能把放行分数当作已确认适用的概率。

## 第四步：智能体逐条核验

智能体对每条返回规则执行以下步骤：

1. 用当前任务证据检查 `applies_when`，不推断缺失的事实；空条件表示除例外外均适用。
2. 检查明确例外，再对照 `content` 核验拟执行动作或回答。
3. 对于审查任务，记录规则标识、来源与对象证据，区分已满足、违反和证据不足。规则已满足也应纳入适用规则。
4. 新事实实质改变适用场景时，再次搜索。

本例中，说明文件 `README.md` 的声明触发规则；许可证文件 `LICENSE` 存在还不足以确认文本完整，应继续读取。空结果不代表没有相关规则，也不构成“校验通过”。

## 接入已有智能体

[VikingBot](../guides/17-vikingbot.md) 已提供 `openviking_rule_search` 工具（Beta），连接 OpenViking 后可按需调用。

其他智能体可以使用 Python 远程客户端或网络接口，并安装配套技能。将完整目录放入智能体配置的技能目录，再让智能体加载：

| 任务 | 配套技能 |
| --- | --- |
| 准备来源、编译并检查结果 | [规则编译技能](https://github.com/volcengine/OpenViking/tree/main/examples/skills/add-rules) |
| 组织任务上下文、召回并核验规则 | [规则搜索技能](https://github.com/volcengine/OpenViking/tree/main/examples/skills/openviking-rule-search) |

技能提供操作与行为要求；智能体仍需具备可调用的接口。当前 Beta 通过用户或智能体主动调用使用。

## 相关文档

- [规则（Beta）](../concepts/17-rule-scenarios.md)
- [规则接口（Beta）](../api/26-rules.md)
- [后台任务](../api/17-tasks.md)
- [应用开发路径](./01-overview.md)

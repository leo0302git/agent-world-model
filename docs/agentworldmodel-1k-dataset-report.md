# AgentWorldModel-1K 数据集报告

本报告基于本地数据集：

```text
/data1/jczhong/datasets/AgentWorldModel-1K
```

AgentWorldModel-1K 是 Agent World Model（AWM）发布的 1000 个合成
agent tool-use 环境。它不是普通的 `prompt -> answer` 数据集，而是一组
**可执行环境材料**：每个 scenario 都有自然语言任务、SQLite 数据库 schema、
初始数据、API/tool 规格、FastAPI/MCP 环境代码，以及每个任务对应的 verifier。

## 初步印象

这个数据集可以理解为：

- 1000 个 synthetic scenario / environment。
- 每个 scenario 10 个用户任务，总计 10000 个任务。
- 每个环境有 SQLite schema 和 sample data，可以还原初始数据库状态。
- 每个环境有生成好的 API spec 和完整 Python 环境源码。
- 每个任务有 verifier，可用于判断 agent 运行后的数据库状态是否完成任务。

使用时要注意：**不要用行号跨文件关联**。这些 JSONL 文件的顺序不保证一致。
应该使用 `scenario` 字段 join；`gen_scenario.jsonl` 里对应字段叫 `name`。

## 文件清单与体量

| 文件 | 大小 | 行数 | 粒度 | 作用 |
|---|---:|---:|---|---|
| `README.md` | 512B | - | metadata | 数据集卡片和资源链接 |
| `gen_scenario.jsonl` | 1.3M | 1000 | scenario | 场景名和自然语言场景描述 |
| `gen_tasks.jsonl` | 2.7M | 1000 | scenario | 每个 scenario 的 10 个用户任务 |
| `gen_db.jsonl` | 12M | 1000 | scenario | 每个 scenario 的 SQLite schema |
| `gen_sample.jsonl` | 36M | 1000 | scenario | 初始化数据库用的 sample data / insert SQL |
| `gen_spec.jsonl` | 66M | 1000 | scenario | API/tool 规格 |
| `gen_envs.jsonl` | 100M | 1000 | scenario | 完整 FastAPI/MCP 环境源码 |
| `gen_verifier.jsonl` | 237M | 10476 | task | SQL verifier，用于 code-augmented LLM-as-judge |
| `gen_verifier.pure_code.jsonl` | 44M | 10010 | task | 纯代码 verifier |

本地总大小约 `495M`。

## 汇总统计

| 指标 | 数值 |
|---|---:|
| `gen_tasks.jsonl` 中的 unique scenario 数 | 1000 |
| 每个 scenario 的任务数 | 10 |
| 理论任务总数 | 10000 |
| SQL verifier 中 unique `(scenario, task_idx)` 数 | 10000 |
| SQL verifier 原始行数 | 10476 |
| pure-code verifier 中 unique `(scenario, task_idx)` 数 | 10000 |
| pure-code verifier 原始行数 | 10010 |
| 平均 DB 表数量 / scenario | 18.46 |
| 最少 / 最多 DB 表数量 | 6 / 42 |
| 平均 sample data 表数量 / scenario | 18.41 |
| 最少 / 最多 sample data 表数量 | 12 / 42 |
| 平均 insert statement 数量 / scenario | 129.33 |
| 最少 / 最多 insert statement 数量 | 103 / 429 |
| 平均 API group 数量 / scenario | 10.69 |
| 平均 API endpoint 数量 / scenario | 35.06 |
| 最少 / 最多 API endpoint 数量 | 16 / 87 |
| 平均环境源码长度 | 100063 chars |
| 最短 / 最长环境源码长度 | 54301 / 184109 chars |
| 平均 SQL verifier 代码长度 | 9497 chars |
| 平均 pure-code verifier 代码长度 | 3988 chars |

verifier 文件存在少量重复行。建议消费时按 `(scenario, task_idx)` 去重，或者和
AWM 官方 helper 一样使用第一个匹配项。

## 文件之间如何关联

一个完整环境的概念结构是：

```text
scenario
  -> tasks[0..9]
  -> db_schema
  -> sample_data
  -> api_spec
  -> full_code
  -> verifier for each task_idx
```

使用 `scenario` 关联这些文件：

- `gen_tasks.jsonl`
- `gen_db.jsonl`
- `gen_sample.jsonl`
- `gen_spec.jsonl`
- `gen_envs.jsonl`

使用 `(scenario, task_idx)` 关联 verifier：

- `gen_verifier.jsonl`
- `gen_verifier.pure_code.jsonl`

`gen_scenario.jsonl` 的场景字段是 `name`，不是 `scenario`。

## 每个文件的字段与示例 item

下面的示例尽量使用同一个代表性场景：`e_commerce_33`。为了便于阅读，较大的
字段会做截断，但保留真实结构。

### `gen_scenario.jsonl`

一个 item 描述一个合成业务场景。

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `name` | string | scenario 标识符 |
| `description` | string | 自然语言场景描述 |

示例 item：

```json
{
  "name": "e_commerce_33",
  "description": "Amazon is the world's largest e-commerce platform offering millions of products across various categories. Users can browse and search for products, read customer reviews, compare prices, and add items to their shopping cart. The platform provides features like Prime membership for fast shipping, wish lists for saving items, order tracking, and easy returns. Customers can also manage their account settings, payment methods, addresses, and view their purchase history."
}
```

这个文件适合用来浏览环境领域，也可以作为理解任务和工具设计的上层语境。

### `gen_tasks.jsonl`

一个 item 包含某个 scenario 下的 10 条用户任务。

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `scenario` | string | scenario 标识符 |
| `tasks` | array[string] | 该 scenario 下的 10 个用户任务 |

示例 item：

```json
{
  "scenario": "e_commerce_33",
  "tasks": [
    "Search for 'wireless noise cancelling headphones', sort results by average customer rating, and add the top-rated item under $200 to my cart in quantity 1.",
    "From my past orders, find the most recent purchase of 'paper towels' and reorder the exact same item and quantity.",
    "Create a new wish list named 'Kitchen Upgrades 2025' and add the three best-selling air fryers under $150 to that list.",
    "Update my default shipping address to '1234 Elm Street, Apt 5B, Springfield, IL 62704' and set it as the primary address for all future orders.",
    "Find a Kindle eBook version of 'Atomic Habits' by James Clear, ensure the price is under $20, and purchase it to be delivered to my default Kindle device.",
    "Search for 'USB-C charging cable 6ft', filter results to only show items with Prime eligible shipping and at least a 4-star rating, then add the cheapest option to my cart with quantity 2.",
    "Locate my order for 'Instant Pot Duo 7-in-1' placed within the last 6 months and initiate a return request selecting 'Item defective or doesn't work' as the reason and requesting a refund to my original payment method.",
    "Add a new payment method using the credit card number '4111 1111 1111 1111' with expiration '12/28' and CVV '123', set it as my default payment option, and remove my previous default credit card.",
    "Subscribe to a 'household paper towels' product with at least a 4-star rating using Subscribe & Save, delivering a 12-roll pack every 2 months to my default address.",
    "Search for 'LEGO Star Wars Millennium Falcon', open the product with the highest number of customer reviews, and post a 5-star review with the title 'Amazing build quality' and body text 'Took a full weekend to build and was worth every minute.'"
  ]
}
```

这个文件是自然语言任务源。在 AWM 的构建流程里，这些 task 也作为后续 DB、
sample data、API、environment 和 verifier 的功能需求。

### `gen_db.jsonl`

一个 item 包含一个 scenario 的 SQLite schema。

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `scenario` | string | scenario 标识符 |
| `db_path` | string | AWM 生成时使用的数据库路径 |
| `db_schema` | object | schema 对象，包含 tables、DDL、indexes |

示例 item，已截断：

```json
{
  "scenario": "e_commerce_33",
  "db_path": "./outputs/databases/e_commerce_33.db",
  "db_schema": {
    "tables": [
      {
        "name": "users",
        "ddl": "CREATE TABLE users (\n    id INTEGER PRIMARY KEY,\n    username TEXT UNIQUE NOT NULL,\n    email TEXT UNIQUE NOT NULL,\n    full_name TEXT,\n    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\n    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP\n);",
        "indexes": [
          "CREATE INDEX idx_users_email ON users(email);"
        ]
      },
      {
        "name": "products",
        "ddl": "CREATE TABLE products (\n    id INTEGER PRIMARY KEY,\n    title TEXT NOT NULL,\n    description TEXT,\n    brand TEXT,\n    category_id INTEGER,\n    is_prime_eligible INTEGER NOT NULL DEFAULT 0,\n    is_subscription_eligible INTEGER NOT NULL DEFAULT 0,\n    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\n    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,\n    FOREIGN KEY (category_id) REFERENCES product_categories(id)\n);",
        "indexes": [
          "CREATE INDEX idx_products_category_id ON products(category_id);",
          "CREATE INDEX idx_products_title ON products(title);"
        ]
      }
    ]
  }
}
```

`e_commerce_33` 的完整 schema 有 19 张表。这里的 `ddl` 是可执行 SQLite DDL，
不是单纯文档。

### `gen_sample.jsonl`

一个 item 包含一个 scenario 的初始数据库数据。

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `scenario` | string | scenario 标识符 |
| `sample_data` | object | 表级 sample data，包含 reasoning 和 insert SQL |
| `tables_count` | integer | sample data 覆盖的表数量 |
| `inserts_count` | integer | insert statement 总数 |

示例 item，已截断：

```json
{
  "scenario": "e_commerce_33",
  "tables_count": 19,
  "inserts_count": 133,
  "sample_data": {
    "tables": [
      {
        "table_name": "users",
        "reasoning": "Creates the primary authenticated user (id=1) and another user to support multi-user scenarios like reviews and marketplace sellers.",
        "insert_statements": [
          "INSERT INTO users (id, username, email, full_name, created_at, updated_at) VALUES (1, 'primary_user', 'primary_user@example.com', 'Primary User', datetime('now', '-365 days'), datetime('now', '-1 days'));",
          "INSERT INTO users (id, username, email, full_name, created_at, updated_at) VALUES (2, 'secondary_user', 'secondary_user@example.com', 'Secondary User', datetime('now', '-200 days'), datetime('now', '-1 days'));"
        ]
      },
      {
        "table_name": "products",
        "reasoning": "Creates all products needed by the tasks, including headphones, USB-C cables, air fryers, paper towels, Atomic Habits, Instant Pot, and LEGO products.",
        "insert_statements": [
          "INSERT INTO products (id, title, description, brand, category_id, is_prime_eligible, is_subscription_eligible, created_at, updated_at) VALUES (1, 'Sony WH-CH710N Wireless Noise Cancelling Headphones', 'Over-ear wireless Bluetooth headphones with active noise cancellation.', 'Sony', 2, 1, 0, datetime('now', '-300 days'), datetime('now', '-2 days'));",
          "INSERT INTO products (id, title, description, brand, category_id, is_prime_eligible, is_subscription_eligible, created_at, updated_at) VALUES (17, 'LEGO Star Wars Millennium Falcon', 'Iconic LEGO Star Wars Millennium Falcon building kit.', 'LEGO', 12, 1, 0, datetime('now', '-600 days'), datetime('now', '-2 days'));"
        ]
      }
    ]
  }
}
```

这个文件对可解性很关键。任务里提到的实体应该在初始 DB 中存在，例如电商任务需要
商品、历史订单、默认地址、默认支付方式、购物车和 Kindle 设备等。

### `gen_spec.jsonl`

一个 item 包含一个 scenario 的 API/tool 规格。

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `scenario` | string | scenario 标识符 |
| `api_spec` | object | API groups、endpoints、参数、响应、依赖表字段 |

示例 item，已截断：

```json
{
  "scenario": "e_commerce_33",
  "api_spec": {
    "api_groups": [
      {
        "group_name": "Products",
        "endpoints": [
          {
            "path": "/api/products/search",
            "method": "GET",
            "summary": "Search products with text, filters, and sorting",
            "operation_id": "search_products",
            "tags": ["products", "search"],
            "request_params": {
              "query": {
                "type": "string",
                "param_type": "query",
                "required": false,
                "description": "Full-text search on product title and description.",
                "example": "wireless noise cancelling headphones"
              },
              "max_price": {
                "type": "float",
                "param_type": "query",
                "required": false,
                "description": "Maximum offer price.",
                "example": 200.0
              },
              "sort_by": {
                "type": "string",
                "param_type": "query",
                "required": false,
                "description": "Sort key: 'relevance','average_rating','review_count','sales_rank','price_asc','price_desc'.",
                "example": "average_rating"
              }
            },
            "required_tables": [
              "products",
              "product_aggregates",
              "product_offers"
            ]
          }
        ]
      }
    ]
  }
}
```

这个文件是从自然语言任务到可调用工具之间的桥梁。它适合做静态分析，例如检查
task 是否有对应的 read/write endpoint、工具参数是否覆盖题面里的关键约束。

### `gen_envs.jsonl`

一个 item 包含一个可运行环境的完整 Python 源码。

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `scenario` | string | scenario 标识符 |
| `db_path` | string | AWM 生成时使用的数据库路径 |
| `full_code` | string | 完整 FastAPI environment 源码 |

示例 item，已截断：

```json
{
  "scenario": "content_platform_1",
  "db_path": "./outputs/databases/content_platform_1.db",
  "full_code": "from fastapi import FastAPI, Query, Path, Body\nfrom pydantic import BaseModel, Field, ConfigDict\nfrom typing import Optional, List, Dict\nfrom sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey\n...\napp = FastAPI(...)\n...\n"
}
```

`full_code` 通常很大，平均约 10 万字符。里面包含 SQLAlchemy models、Pydantic
request/response models、FastAPI route handlers 和 `app` 对象。AWM 的 server
helper 会提取这段代码，注入运行时 SQLite 路径，并挂载 MCP endpoint。

### `gen_verifier.jsonl`

一个 item 对应一个 task 的 verifier。这个文件用于 SQL verifier + LLM judge 模式。

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `scenario` | string | scenario 标识符 |
| `task` | string | 任务文本 |
| `task_idx` | integer | 该任务在 scenario 内的 0-based index |
| `verification` | object | verifier 代码和元信息 |

示例 item，已截断：

```json
{
  "scenario": "e_commerce_33",
  "task_idx": 0,
  "task": "Search for 'wireless noise cancelling headphones', sort results by average customer rating, and add the top-rated item under $200 to my cart in quantity 1.",
  "verification": {
    "code": "def verify_task(initial_db_path: str, final_db_path: str) -> dict:\n    import sqlite3\n    ...\n    return {...}\n",
    "reasoning": "The verifier inspects the initial and final SQLite databases to identify the expected product offer and whether the final cart state satisfies the task.",
    "function_name": "verify_task"
  }
}
```

SQL 模式通常先用 verifier 从 DB 中提取结构化证据，再交给 LLM judge 结合
trajectory 判断是否完成。

### `gen_verifier.pure_code.jsonl`

一个 item 对应一个 task 的纯代码 verifier。

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `scenario` | string | scenario 标识符 |
| `task` | string | 任务文本 |
| `task_idx` | integer | 该任务在 scenario 内的 0-based index |
| `verification` | object | 纯代码 verifier |

示例 item，已截断：

```json
{
  "scenario": "e_commerce_33",
  "task_idx": 0,
  "task": "Search for 'wireless noise cancelling headphones', sort results by average customer rating, and add the top-rated item under $200 to my cart in quantity 1.",
  "verification": {
    "code": "def verify_task_completion(initial_db_path: str, final_db_path: str, final_answer: str | None = None) -> dict:\n    import sqlite3\n    ...\n    if final_qty == 1 and initial_qty is None:\n        return {\"result\": \"complete\"}\n    return {\"result\": \"others\"}\n",
    "raw_response": ""
  }
}
```

这个模式更适合全自动评测，因为不需要 LLM judge。它通常比较保守：只有当
initial DB 和 final DB 的差异明确证明任务完成时，才返回 `complete`。

## 典型使用方式

AWM 仓库位置：

```text
/data1/jczhong/repos/agent-world-model
```

启动一个环境：

```bash
cd /data1/jczhong/repos/agent-world-model

uv run awm env start \
  --scenario e_commerce_33 \
  --envs_load_path /data1/jczhong/datasets/AgentWorldModel-1K/gen_envs.jsonl \
  --db_schema_path /data1/jczhong/datasets/AgentWorldModel-1K/gen_db.jsonl \
  --sample_path /data1/jczhong/datasets/AgentWorldModel-1K/gen_sample.jsonl \
  --port 8001
```

检查 MCP endpoint：

```bash
uv run awm env check --url http://127.0.0.1:8001/mcp
```

用 OpenAI-compatible API 跑一个 agent：

```bash
uv run awm agent \
  --scenario e_commerce_33 \
  --task_id 0 \
  --envs_path /data1/jczhong/datasets/AgentWorldModel-1K/gen_envs.jsonl \
  --tasks_path /data1/jczhong/datasets/AgentWorldModel-1K/gen_tasks.jsonl \
  --db_path /data1/jczhong/datasets/AgentWorldModel-1K/gen_db.jsonl \
  --sample_path /data1/jczhong/datasets/AgentWorldModel-1K/gen_sample.jsonl \
  --api_url http://127.0.0.1:8000/v1 \
  --model your-model-name
```

用 pure-code verifier 验证运行结果：

```bash
uv run awm verify \
  --input outputs/agents/<timestamp> \
  --mode code \
  --verifier_code_path /data1/jczhong/datasets/AgentWorldModel-1K/gen_verifier.pure_code.jsonl
```

agent 运行目录通常包含：

- `trajectory.json`：messages、tool calls、tool responses、task metadata。
- `initial.db`：agent 操作前的数据库快照。
- `final.db`：agent 操作后的数据库快照。
- `verify.code.json` 或 `verify.sql.json`：验证结果。

## 给 ClawEnvKit 使用时的建议

如果要把这个数据集接入 ClawEnvKit，不建议把它当作已经规范化好的 episode 数据。
更合适的做法是把它当作 **raw environment source material**。

推荐内部结构：

```python
{
    "scenario": "...",
    "tasks": [...],
    "db_schema": {...},
    "sample_data": {...},
    "api_spec": {...},
    "env_code": "...",
    "verifiers": {
        0: {...},
        1: {...}
    }
}
```

进入 benchmark 或训练前，建议增加一层 validation：

1. 按 `scenario` join 所有 scenario 级文件。
2. 按 `(scenario, task_idx)` 去重 verifier。
3. 从 `gen_db.jsonl` 和 `gen_sample.jsonl` 重建 SQLite DB。
4. 启动每个 MCP 环境并检查 tools 是否能列出。
5. 对每个 task 跑 smoke trajectory 或 reference trajectory。
6. 先用 `gen_verifier.pure_code.jsonl` 做自动验证。
7. 对纯代码 verifier 无法稳定判断的任务，再使用 SQL + LLM judge。

这一步很重要：AWM 提供的是可执行环境、任务和 verifier，但数据集本身不是已经
完成求解并带有 golden trajectory 的 episode 集合。


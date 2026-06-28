# FinPilot

AI 驱动的 A股/基金智能分析助手。以自然语言对话为入口，集成股票多维度诊断、基金深度分析、持仓组合分析、市场热点追踪、财报解读等能力——输入一段话，Agent 自动调度合适的工具完成任务，并用口语化方式解读结果。

**核心能力**：LLM Agent 对话 · 股/基诊断 · 持仓分析管理 · 市场热点追踪 · 财报解读

**技术栈**：FastAPI · LangChain/LangGraph · DeepSeek · Chroma · MySQL · akshare · baostock

前端仓库：[FinPilot_front](https://github.com/eric-su/FinPilot_front)（Vue 3 + Element Plus + Vite）

---

## 功能模块

### 1. AI 对话首页
- 调用型 Agent（LangGraph ReAct），自主决定调用股票诊断、基金诊断、持仓分析、财报问答、市场热点、联网搜索等工具
- 用户只需用自然语言表达意图（如"帮我看看贵州茅台"、"我的持仓合理吗"、"今天市场有什么热点"），Agent 自动解析并调对应工具
- 多轮对话上下文保持（MySQL 持久化），会话历史可回溯
- 流式 SSE 输出，逐 token 打字机效果；工具执行期间心跳保活，不会因网络等待而超时断开
- 输出自然口语化，关键数据用对话讲清楚，不罗列原始 JSON

### 2. 股票诊断
- 7 维度卖方研报框架：业务概述 → 基本面 → 技术面 → 资金面 → 风险提示 → 综合诊断 → 实操参考
- 基本面覆盖 ROE、毛利率、净利率、PE/PB、杜邦拆解、成长能力、偿债能力、营运能力
- 技术面含 MA5/10/20/60、RSI(14)、支撑/压力位、量比、趋势判断
- 输入支持 6 位代码或股票名称（如"贵州茅台"），名称自动解析
- 实操参考分持仓者/观望者/短线博弈者三类，给出具体止损位和仓位阈值
- 数据源：baostock（主力）→ akshare 新浪源（兜底，移动网络友好）

### 3. 基金诊断
- 蚂蚁财富 5 段结构：核心诊断 → 业绩与波动 → 持仓分析 → 费率与交易 → 投资建议
- 业绩含今年来/近 1 周/1 月/3 月/6 月/1 年/3 年/成立来收益率、年化波动率、最大回撤
- 前十大持仓实时估值估算（加权涨跌幅）
- 费率分档展示（申购/赎回/管理/托管/销售服务费），标注持有期限建议
- 输入支持 6 位代码或基金名称（如"易方达蓝筹精选"），名称自动解析
- 数据源：akshare eastmoney

### 4. 持仓组合分析
- 触发方式：首页对话输入"分析我的持仓"、"我的组合怎么样"、"持仓合理吗"等
- Agent 调用内部工具，流程分两步：
  1. **并发诊断**：对所有持仓标的用 `ThreadPoolExecutor`（最多 5 线程）并发跑完整诊断图（每只股票走 7 维度分析，每只基金走 5 段分析），墙钟时间≈最慢一只
  2. **组合聚合**：按行业/类型分桶计算集中度，加权风险评分，输出组合整体报告
- 报告包含：组合一句话总结、集中度分析（前 N 重仓行业占比）、整体风险评级（低/中/高）、各持仓简要诊断摘要
- 若用户无持仓，提示先去持仓管理页添加

### 5. 市场热点分析
- 触发方式：首页对话输入"今天市场怎么样"、"有什么热点"、"哪些板块涨得好"等
- Agent 自动拉取 6 个维度的数据：
  - **主要大盘指数**：上证/深证/创业板/科创50/北证50/沪深300/中证500 的实时点位和涨跌幅
  - **行业板块涨幅榜**：涨幅前 5 的行业板块及涨跌幅
  - **行业板块跌幅榜**：跌幅前 5 的行业板块及涨跌幅
  - **概念板块涨幅榜**：涨幅前 5 的概念板块（如 AI、芯片、新能源等主题）
  - **涨停池**：当日涨停股数量、前 5 只涨停股及涨停原因
  - **大盘资金流向**：主力净流入/流出、超大单资金动向
- LLM 拿到结构化数据后，可自主调用联网搜索补充相关新闻，综合解读成口语化市场简报
- 非交易时段（周末/节假日/收盘后）返回上一个交易日数据，并明确告知用户数据日期
- 数据源：akshare eastmoney 各独立接口，单个接口失败不影响其他维度

### 6. 财报 RAG
- 上传 A 股年报/季报 PDF → 自动分块入库 → Chroma 向量检索
- 混合检索（BM25 关键词 + 向量语义），带来源页码标注
- 支持跨会话复用：同一股票的财报只需上传一次，任何会话都能查
- 财报时效性自动检测（报告期距今超过 18 个月标记陈旧），陈旧数据提示用户上传最新版本
- 嵌入模型：阿里百炼 DashScope text-embedding

### 7. 持仓管理
- 股票/基金持仓 CRUD（代码 + 名称 + 数量 + 成本价）
- 实时盈亏计算（最新行情价 × 数量 - 成本总额）
- JWT 登录 + 用户数据隔离

---

## 快速开始

### 环境要求

- Python 3.10+
- MySQL 8.0+（auth / chat / portfolio 需要；diagnosis / RAG 可独立运行）
- [DeepSeek API Key](https://platform.deepseek.com/)（LLM）

### 1. 克隆并安装依赖

```bash
git clone https://github.com/EricSu-Dev/FinPilot.git
cd FinPilot
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入必要配置：

| 变量 | 说明 | 必需 |
|------|------|:--:|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | ✅ |
| `TAVILY_API_KEY` | Tavily 搜索 API（免费额度） | ✅ |
| `DASHSCOPE_API_KEY` | 阿里百炼 API（财报 RAG 的文本嵌入） | RAG 时需要 |
| `SECRET_KEY` | JWT 签名密钥，生产环境务必更换 | ✅ |
| `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE` | MySQL 连接信息 | auth/chat/portfolio 时需要 |

### 3. 启动

```bash
python main.py
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8094
```

API 运行在 `http://localhost:8094`，健康检查：`GET /health`

---

## API 概览

| 端点 | 方法 | 鉴权 | 说明 |
|------|------|:--:|------|
| `/api/auth/register` | POST | — | 注册 |
| `/api/auth/login` | POST | — | 登录，返回 JWT |
| `/api/diagnosis` | POST | — | 股/基诊断（SSE 流式） |
| `/api/diagnosis/validate` | GET | — | 校验证券代码 |
| `/api/chat` | POST | ✅ | AI 对话（SSE 流式，工具执行期间心跳保活） |
| `/api/chat/upload` | POST | ✅ | 上传财报 PDF |
| `/api/chat/conversations` | GET | ✅ | 会话列表 |
| `/api/portfolio` | GET/POST | ✅ | 持仓列表 / 新增 |
| `/api/portfolio/{id}` | PUT/DELETE | ✅ | 修改 / 删除持仓 |
| `/api/portfolio/validate` | GET | ✅ | 校验新增持仓代码 |

---

## 项目结构

```
FinPilot/
├── main.py                    # 入口，兼容 uvicorn main:app
├── app/
│   ├── main.py                # FastAPI 应用实例 & 路由注册
│   ├── config.py              # 环境变量配置（pydantic-settings）
│   ├── llm.py                 # DeepSeek ChatOpenAI 懒加载代理
│   ├── security.py            # bcrypt 密码哈希
│   ├── api/
│   │   ├── common.py          # api_ok / api_error / sse_event
│   │   ├── deps.py            # JWT get_current_user 依赖
│   │   ├── auth.py            # 注册/登录
│   │   ├── chat.py            # 对话 SSE 流（心跳保活）
│   │   ├── diagnosis.py       # 诊断 SSE 流
│   │   ├── portfolio.py       # 持仓 CRUD
│   │   └── rag.py             # 财报上传 & 语料构建（后台线程）
│   ├── agents/
│   │   ├── chat_agent.py      # 对话 ReAct Agent（工具定义 + System Prompt）
│   │   ├── diagnosis_agent.py # 诊断 LangGraph pipeline
│   │   ├── portfolio_agent.py # 持仓组合分析（并发诊断 → 聚合）
│   │   └── chat_memory.py     # 会话 & 消息持久化
│   ├── data_source/
│   │   ├── stock.py           # 股票行情（baostock → akshare 新浪源降级）
│   │   ├── fund.py            # 基金数据（akshare eastmoney）
│   │   ├── common.py          # 数值转换 & DataFrame 校验
│   │   └── code_validation.py # 证券代码存在性校验
│   ├── tools/
│   │   ├── stock_tools.py     # 股票诊断工具
│   │   ├── fund_tools.py      # 基金诊断 & 估值工具
│   │   ├── portfolio_tools.py # 持仓汇总 & 盈亏计算
│   │   ├── market_hotspots_tool.py  # A 股市场热点
│   │   └── web_search_tool.py       # Tavily 联网搜索
│   ├── rag/
│   │   ├── loader.py          # PDF 加载 & 分块
│   │   ├── chunker.py         # 语义分块策略
│   │   ├── retriever.py       # BM25 + 向量混合检索
│   │   └── qa_chain.py        # RAG 问答链
│   └── models/
│       ├── database.py        # SQLAlchemy engine & session
│       ├── user.py            # 用户模型
│       ├── portfolio.py       # 持仓模型
│       ├── portfolio_crud.py  # 持仓 CRUD（用户隔离）
│       ├── chat.py            # 会话 & 消息模型
│       └── chat_crud.py       # 会话 CRUD
├── data/
│   └── chroma_db/             # Chroma 持久化目录
├── md/                        # 设计文档 & 故障记录
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 数据源说明

| 数据 | 主力源 | 兜底源 | 备注 |
|------|--------|--------|------|
| 股票实时行情 | baostock | akshare 新浪 `stock_zh_a_spot` | 移动网络下 baostock 不可用自动回退；新浪源无 PE/PB |
| 股票历史 K 线 | baostock | akshare 新浪 `stock_zh_a_daily` | 均前复权 |
| 股票详细财务 | baostock | — | ROE/Dupont/成长/偿债/运营，baostock 独有 |
| 股票行业分类 | baostock | — | 证监会行业分类 |
| 基金净值 / 业绩 | akshare eastmoney | — | `fund_open_fund_info_em` |
| 基金基本信息 | akshare eastmoney | — | `fund_name_em`（全市场列表，5 分钟缓存） |
| 基金持仓 / 费率 / 行业 | akshare eastmoney | — | 各独立接口 |
| 名称 → 代码解析 | akshare 新浪 spot（股票）/ eastmoney（基金） | — | 全市场扫描匹配 |
| 市场热点 | akshare eastmoney | — | 指数/板块涨幅榜/涨停股/资金流向 |
| 联网搜索 | Tavily | — | 补充最新消息、机构观点 |

---

## 架构决策

- **SSE 心跳保活**：对话 Agent 在执行同步工具（baostock/akshare）时会阻塞 20~40s，期间无 SSE 产出。通过将 agent 流放入独立线程 + `queue.Queue` 传递 chunk + 主线程 5s 超时轮询发心跳，防止前端 / Nginx 静默超时。详见 `md/chat-agent-sse-blocking.md`。
- **baostock 惰性登录**：baostock login 在移动网络下 TCP 超时约 20s。改为后台探测线程，首次请求不等 login 直接走 akshare 兜底，不阻塞。
- **akshare 缓存降级**：`stock_zh_a_spot` 全市场行情带 60s TTL 缓存；调用失败时降级返回过期缓存，避免网络偶发抖动导致整页不可用。
- **诊断公开、持仓私有**：诊断 / RAG 为无状态 AI 分析，不鉴权；portfolio / chat 需要登录做用户数据隔离。
- **同步 I/O 工具**：baostock 和 akshare 均为同步库，通过 LangGraph ToolNode 的同步调用路径执行。基金持仓估值时对前十大重仓股逐个获取实时行情，用线程池并发控制（最多 5 worker）。

---

## 许可证

MIT

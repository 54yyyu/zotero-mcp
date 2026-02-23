# 计划：在保持原版向量化行为不变的前提下，新增可选 MinerU 分块向量化（PR 友好，含元数据双索引聚合）

## 摘要
目标是把当前分支已有的 MinerU/分块能力，重构成“默认完全沿用原版行为、仅在用户明确选择时启用”的实现形态，确保后续向原版仓库提 PR 时改动集中、兼容性清晰。  
已锁定决策：
1. 新增 `extraction_mode` 配置（推荐形态）。
2. MinerU 模式下优先读取 `~/.config/zotero-mcp/md_store` 已解析内容。
3. MinerU token 默认写入配置文件，同时支持环境变量覆盖。
4. 元数据也参与检索：作为独立 pseudo-chunk。
5. 命中融合规则：正文优先 `max` 融合。

## 公共接口/配置变更
1. `~/.config/zotero-mcp/config.json` 的 `semantic_search` 下新增字段：
   - `extraction_mode`: `"local"` | `"mineru"`（默认 `"local"`）。
2. 保留并兼容旧字段：
   - `mineru.enabled`、`mineru.tokens`、`mineru.*` 参数继续支持（向后兼容）。
   - 兼容规则：若无 `extraction_mode`，按旧逻辑推断（`mineru.enabled=true` 则视为 `"mineru"`，否则 `"local"`）。
3. CLI 原有向量化入口不破坏：
   - `update-db --fulltext` 语义不变。
   - 是否走 MinerU 由配置中的 `extraction_mode` 决定。

## 实施方案（决策完整）
1. 配置层归一化
   - 在 `semantic_search` 初始化时新增统一解析函数，将 `extraction_mode` 与旧 `mineru.enabled` 折叠成一个内部运行态。
   - 运行态输出字段：`extraction_mode`、`mineru.tokens`、`mineru.client_options`。

2. Setup 交互改造（保持原版流程骨架）
   - 在原有 embedding/update/extraction/db_path 交互之后，新增“全文抽取模式”选择：
     - `local`（默认，原版行为）
     - `mineru`（新增）
   - 仅当用户选择 `mineru` 时引导 token 配置（保留当前多 token 输入与掩码显示）。
   - `setup --semantic-config-only` 同步支持该选择，不改原有参数语义。

3. 本地读取链路改造（按原实现读取 Zotero）
   - 继续使用原有 SQLite/附件解析流程扫描 Zotero（不改数据入口）。
   - 在 PDF 抽取时改为固定优先级：
     1. 先查 `md_store/{item_key}/{attachment_key}/*.(md|md.zst)`，命中即读并标记 `fulltext_source="mineru_md"`。
     2. 未命中时：
        - 若有 token：调用 MinerU API 解析，成功后写回 `md_store`，再进入分块。
        - 否则回退本地 PDF 提取（原逻辑）。
   - HTML/非 PDF 保持原逻辑。

4. 分块与向量写入（含元数据双索引）
   - 正文 chunk：保持现有 `chunk_id={item_key}:{attachment_key}:{idx}`。
   - 元数据 pseudo-chunk：新增 `chunk_id={item_key}:meta:0`，文本由标题+摘要+作者+标签等拼接生成。
   - 两类 chunk 统一入同一 collection，并在 metadata 标注 `chunk_kind`（`content`/`meta`）。
   - `locator_store` 对 `meta` chunk 可写空定位（或专用最小定位记录），避免回填报错。

5. 检索结果去重与融合排序（核心新增）
   - 检索返回后先按 `item_key` 聚合，不直接逐 chunk 输出。
   - 分数融合规则：
     - `item_score = max(best_content_score, 0.85 * meta_score)`。
   - 输出策略：
     - 每个 `item_key` 最终只返回一条主结果；
     - 证据默认返回最多 2 条正文 chunk；
     - 元数据命中仅作辅助证据（最多 1 条）。

6. 状态与可观测性
   - `db-status`/`setup-info` 增加 `extraction_mode`、`meta_chunk_enabled` 展示。
   - 保留 `mineru_enabled/token_count`。
   - `doctor` 输出补充模式与 `md_store` 命中/回退信息。

7. PR 对齐策略（面向原版仓库）
   - 变更集中在：
     - `setup_helper.py`（配置交互与写入）
     - `semantic_search.py`（配置归一化、双索引写入、聚合融合）
     - `local_db.py`（正文来源优先级）
   - 不改 MCP 工具协议，不改 `update-db` 现有必需参数。
   - 默认配置下行为与原版一致（`local` 模式）。

## 测试用例与场景
1. 配置兼容
   - 仅旧配置（无 `extraction_mode`）时，行为与旧逻辑一致。
   - 新配置 `extraction_mode=local/mineru` 时优先级正确。

2. Setup 行为
   - 选择 `local`：不强制输入 MinerU token。
   - 选择 `mineru`：触发 token 引导并正确保存。
   - token 为空时回退策略正确（不阻塞向量化）。

3. 抽取优先级
   - `md_store` 已有文件时，不调用 MinerU API。
   - `md_store` 无文件但 token 可用时，调用 MinerU 并写回 `md_store`。
   - MinerU 失败时回退本地 PDF 提取。

4. 双索引入库
   - 每个 item 至少有 `meta` chunk（文本可用时）。
   - 有全文时有多个 `content` chunk。
   - `chunk_kind`、`chunk_id` 命名符合规范。

5. 聚合去重与排序
   - 同一 item 同时命中 `meta` + `content` 时，结果列表仅 1 条。
   - 排序满足 `max(content, 0.85*meta)` 规则。
   - 仅 `meta` 命中时仍可召回 item。

6. 回归
   - 默认 `local` 模式下，`update-db --fulltext` 与原版统计口径一致。
   - metadata-only（不加 `--fulltext`）路径不受影响。

## 验收标准
1. 默认安装并运行 `setup` 后，不选 MinerU 时行为与原版一致。
2. 选 MinerU 后，能从 `md_store` 命中并完成块状向量化。
3. 支持元数据 pseudo-chunk 检索且结果 item 级去重正确。
4. `db-status` 清晰显示模式、token 与双索引状态。
5. 新增功能为增量改动，不破坏现有 CLI/MCP 对外接口。

## 假设与默认值
1. 默认 `extraction_mode="local"`。
2. MinerU token 存储在 `config.json`，运行时可被 `MINERU_TOKEN`/`MINERU_TOKENS` 覆盖。
3. `md_store` 默认路径为 `~/.config/zotero-mcp/md_store`。
4. 当 `extraction_mode="mineru"` 但 token 缺失时，不报致命错误，自动回退本地 PDF 提取。
5. 元数据融合权重固定为 `0.85`（后续可配置化，但首版先写死以控制复杂度）。

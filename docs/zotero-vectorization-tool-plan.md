# Zotero 文献库向量化工具建设计划

## 1. 目标与范围
1. 构建一个独立工具，将 Zotero 元数据与 PDF 全文向量化，形成可增量维护的本地知识库。
2. 支持外部问答系统按“用户问题”检索正文证据片段，并返回可追溯的参考文献信息。
3. 与现有 `zotero-mcp` 解耦，最终以“外挂向量后端”方式被 `zotero-mcp` 直接调用。
4. 采用“低冗余索引”原则：Zotero PDF 作为唯一正文真源，向量库仅保存向量与定位信息。

## 2. 交付目标
1. `ingest`：从本地 Zotero（优先 Local API，回退 SQLite）和附件目录抽取元数据与全文。
2. `index`：对 chunk 级文本做向量化并写入专业向量库（默认 Qdrant），不冗余持久化 chunk 原文。
3. `search`：提供 dense/hybrid 检索 + rerank + 引文聚合返回，并在查询时按定位信息回填正文片段。
4. `ops`：支持增量更新、删除同步、快照备份、质量评测与监控。
5. `adapter`：提供给 `zotero-mcp` 的统一查询接口（Python SDK 或 HTTP API）。

## 3. 架构设计
1. 数据源层
   - Zotero Local API：读取 item、creator、tag、dateModified、attachment 映射。
   - Zotero SQLite（回退）：当 Local API 不可用时读取本地数据库。
   - Zotero storage：读取 PDF/HTML 等附件文件。
2. 解析层
   - PDF 文本抽取主链路：优先 MinerU 远程解析（`/api/v4/extract/task`）。
   - 回退链路：MinerU 失败或超时时，回退 PyMuPDF/pdfplumber 本地解析。
   - OCR 兜底（扫描件场景，可选，放在本地解析后）。
   - 统一清洗与标准化（去脚注噪声、换行修复、编码清理）。
   - 解析结果质量打分：按文本长度、可读字符占比、页覆盖率决定是否触发回退。
3. 切块层
   - 按 token 切块（默认 500-800 tokens，overlap 100）。
   - 保留定位信息：`item_key`、`chunk_id`、`page_start/page_end`、`section`。
   - 生成文本定位偏移：`char_start/char_end` 或 `token_start/token_end`。
4. 向量层
   - Dense embedding（可切换 OpenAI/本地模型）。
   - 可选 Sparse/BM25 字段，支持 hybrid。
   - 支持向量量化压缩（float16/scalar/PQ）。
5. 存储层
   - 向量库：Qdrant collection（默认），存向量与payload，不存完整正文。
   - 文本定位索引：SQLite/Postgres（建议），存 `chunk_id -> doc_id + offset` 与构建状态。
   - 可选文本缓存层：压缩文本缓存（zstd），用于加速回填，不复制 PDF 文件本体。
6. 检索层
   - 召回：dense/hybrid。
   - 重排：cross-encoder rerank（可选）。
   - 回填：根据 `doc_id + offset/page` 延迟取文（late materialization）。
   - 聚合：chunk -> item 级结果聚合，生成引用清单。
7. 接口层
   - CLI：`build`, `update`, `search`, `status`, `snapshot`, `eval`。
   - API：`/search`, `/fetch`, `/health`, `/metrics`（可选）。

## 4. 数据模型规范
1. 向量 ID：`{item_key}:{chunk_index}`。
2. 每个 chunk payload 字段：
   - `item_key`
   - `chunk_index`
   - `title`
   - `authors`
   - `year`
   - `doi`
   - `citation_key`
   - `tags`
   - `page_start`
   - `page_end`
   - `source_type` (`pdf`/`html`)
   - `text_hash`
   - `date_modified`
   - `char_start`
   - `char_end`
3. 文本定位索引字段（SQLite/Postgres）：
   - `chunk_id`
   - `item_key`
   - `attachment_key`
   - `file_path`
   - `page_start`
   - `page_end`
   - `char_start`
   - `char_end`
   - `file_hash`
4. 文献级索引映射（用于聚合与删除同步）：
   - `item_key -> [chunk_ids]`

## 5. 实施阶段与里程碑
### Phase 0: 基线定义（1 周）
1. 明确数据规模、目标硬件、可接受延迟和检索质量指标。
2. 选定第一版 embedding 模型、向量库（Qdrant）和重排策略（先关闭）。
3. 冻结 MVP 范围：先 dense + chunk + metadata filter + 延迟取文。
4. 明确 MinerU 使用策略：`MINERU_API_TOKEN`、超时阈值、重试次数、成本预算。

### Phase 1: 数据抽取与切块（1-2 周）
1. 实现 Zotero 抽取器：Local API 优先，SQLite 回退；元数据、附件定位、增量扫描（基于 `dateModified` + file hash）。
2. 实现 PDF 解析器与清洗器：
   - `MinerUParser`：提交任务、轮询结果、下载解析文本。
   - `LocalPdfParser`：PyMuPDF/pdfplumber 本地解析。
   - `FallbackOrchestrator`：按失败原因自动回退（HTTP 非 2xx、结果为空、质量分不足、超时）。
3. 实现 chunker 与 offset 映射器，并输出调试样本（可查看每 chunk 页码与偏移）。
4. 验收：随机抽样 200 篇文献，正文解析成功率 >= 95%。

### Phase 2: 向量写入与索引管理（1 周）
1. 实现向量写入器（批量 upsert、断点续跑、失败重试），禁止写入全文正文到向量库。
2. 实现 collection schema 初始化与版本管理。
3. 实现删除同步（文献删除时移除全部 chunk）与定位索引同步清理。
4. 验收：5GB 库全量构建完成，可重复运行且幂等。

### Phase 3: 检索与引用聚合（1 周）
1. 实现 query pipeline：embedding -> topK chunk -> 延迟取文 -> item 聚合。
2. 输出结构：命中片段、相似度、参考文献条目（title/authors/year/item_key/doi）与定位信息（page/offset）。
3. 增加 metadata 过滤（年份、标签、文献类型）。
4. 验收：测试问题集上 `Recall@10` 达到基线目标。

### Phase 4: `zotero-mcp` 适配（0.5-1 周）
1. 抽象 `VectorClient` 接口，增加 `QdrantVectorClient`。
2. 在 `semantic_search` 中切换为外部后端（配置驱动）。
3. 保持工具层 API 不变：`search/fetch` 与 `zotero_semantic_search` 可直接复用。
4. 验收：`zotero-mcp` 在不改调用方协议下可用新检索能力。

### Phase 5: 质量与运维（持续）
1. 建立评测集（100-300 条真实问题）。
2. 接入快照备份与恢复流程。
3. 输出监控指标：索引量、失败率、平均检索延迟、命中质量、存储占用率（含向量压缩收益）。

## 6. 推荐技术选型（第一版）
1. 向量库：Qdrant（本地 Docker 或二进制部署）。
2. 解析：
   - 主解析：MinerU API（`model_version=vlm`）。
   - 回退解析：PyMuPDF + pdfplumber（OCR 可选 PaddleOCR/Tesseract）。
3. 切块：token-based chunking（tiktoken 或 sentencepiece 兼容方案）。
4. Embedding：
   - 云端：`text-embedding-3-large` 或同级模型。
   - 本地：`bge-m3` / `e5` 家族。
5. Rerank（第二版启用）：`bge-reranker` 或 cross-encoder。

## 7. CLI 草案
1. `zotero-vector init`
   - 生成配置模板（Zotero 连接、Qdrant、Embedding、MinerU）。
2. `zotero-vector build --full`
   - 全量扫描本地文献库，解析并写入向量库。
3. `zotero-vector update --since-last`
   - 增量同步（新增/更新/删除）。
4. `zotero-vector search --query "..." --top-k 20`
   - 检索并返回片段 + 引文。
5. `zotero-vector status`
   - 查看索引规模、最近任务状态、失败统计。
6. `zotero-vector snapshot create`
   - 备份向量库与定位索引。
7. `zotero-vector eval --dataset eval/questions.jsonl`
   - 跑评测集输出指标。
8. `zotero-vector doctor --check-offset-integrity`
   - 校验定位偏移与文件 hash。

### CLI 配置建议（`~/.config/zotero-vector/config.toml`）
1. `zotero.connection = "local_api"`（可选 `sqlite`）
2. `zotero.local_api_url = "http://127.0.0.1:23119/api"`
3. `zotero.sqlite_path` 与 `zotero.storage_dir`（回退模式必填）
4. `mineru.enabled = true`
5. `mineru.api_token = "${MINERU_API_TOKEN}"`
6. `mineru.task_url = "https://mineru.net/api/v4/extract/task"`
7. `mineru.model_version = "vlm"`
8. `mineru.timeout_seconds = 60`
9. `mineru.max_retries = 2`
10. `parse.fallback_order = ["mineru", "pymupdf", "pdfplumber"]`

## 8. 与 zotero-mcp 的集成策略
1. 在 `~/.config/zotero-mcp/config.json` 增加：
   - `semantic_search.backend = "qdrant"`
   - `semantic_search.qdrant.url`
   - `semantic_search.qdrant.collection`
2. `zotero-mcp` 的 `semantic_search.py` 仅负责：
   - 调用外部向量检索服务
   - 回填 Zotero item 元数据
   - 格式化返回结果
3. 向量构建任务由独立工具负责，`zotero-mcp` 不再承担重索引主流程。

## 9. 验收标准
1. 功能
   - 支持全量构建与增量更新。
   - 查询可返回正文片段与参考文献列表。
   - 向量库存储不包含完整 chunk 正文，仅包含向量与定位payload。
2. 性能
   - 单次查询 P95 < 2s（本地单机目标，可按硬件调整）。
   - 5GB 文献库支持稳定查询和每日增量更新。
3. 质量
   - `Recall@10`、`nDCG@10` 达到预设阈值。
   - 人工抽检引用可追溯到原文页码或 chunk。
4. 可运维
   - 具备日志、失败重试、快照恢复能力。
   - 具备存储健康检查（偏移有效性、文件hash一致性）。

## 10. 风险与应对
1. PDF 解析质量不稳定
   - 应对：MinerU 优先 + 本地解析回退链 + OCR 兜底 + 解析质量打分。
2. MinerU API 不可用或限流
   - 应对：请求重试、并发节流、快速回退到本地解析，保证任务不中断。
3. 索引构建耗时过长
   - 应对：批量并发、增量更新、断点续跑。
4. 检索命中泛化不足
   - 应对：hybrid 检索 + rerank + 评测集迭代。
5. 结果不可解释
   - 应对：强制输出 chunk 证据文本、页码、item_key、doi。

## 11. 最小可行版本（MVP）
1. Qdrant + dense chunk 检索。
2. Zotero 本地文献库接入：支持 Local API 读取与 SQLite 回退。
3. PDF 解析策略：MinerU 优先，失败自动回退本地解析；向量库仅存向量与定位信息。
4. 查询返回：
   - `answer_context_chunks`
   - `references`（title/authors/year/item_key/doi）
   - `locators`（page_start/page_end/char_start/char_end）
5. `zotero-mcp` 通过配置切换到外部向量后端。

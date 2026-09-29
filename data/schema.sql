-- =====================================================================
-- E-QTSF 机械加工质量管理系统 数据库 Schema
-- 参考《机械加工企业质量管理系统的设计与实现》15 表模型，
-- 新增 4 张模型相关表：quality_event / forecast_result / quality_alert / inspection_value
-- 兼容 SQLite（开发态）与 MySQL（生产态，TEXT->VARCHAR/DATETIME 按需调整）
-- =====================================================================

-- 部门表
CREATE TABLE IF NOT EXISTS department (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  dept_name   TEXT NOT NULL,
  parent_id   INTEGER,
  sort_order  INTEGER DEFAULT 0,
  deleted     INTEGER DEFAULT 0,
  create_time TEXT DEFAULT (datetime('now','localtime'))
);

-- 用户表（角色：ADMIN/INSPECTOR/ENGINEER）
CREATE TABLE IF NOT EXISTS sys_user (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  username      TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  real_name     TEXT,
  role          TEXT NOT NULL,
  dept_id       INTEGER REFERENCES department(id),
  deleted       INTEGER DEFAULT 0,
  create_time   TEXT DEFAULT (datetime('now','localtime'))
);

-- 产品表
CREATE TABLE IF NOT EXISTS product (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  product_code TEXT UNIQUE NOT NULL,          -- PRD-001
  product_name TEXT NOT NULL,
  product_type TEXT,
  specification TEXT,
  material     TEXT,
  unit         TEXT,
  status       TEXT DEFAULT '生产中',          -- 生产中/试制/已停产
  deleted      INTEGER DEFAULT 0,
  create_time  TEXT DEFAULT (datetime('now','localtime'))
);

-- 批次表
CREATE TABLE IF NOT EXISTS product_batch (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id      INTEGER NOT NULL REFERENCES product(id),
  batch_no        TEXT UNIQUE NOT NULL,        -- BATCH-2024-0001
  production_date TEXT,
  total_quantity  INTEGER,
  deleted         INTEGER DEFAULT 0,
  create_time     TEXT DEFAULT (datetime('now','localtime'))
);

-- 缺陷代码表
CREATE TABLE IF NOT EXISTS defect_code (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  defect_code         TEXT UNIQUE NOT NULL,    -- DEF-001
  defect_name         TEXT NOT NULL,
  category            TEXT NOT NULL,           -- 尺寸/外观/性能/包装/其他
  severity            TEXT NOT NULL,           -- 致命/严重/一般/轻微
  description         TEXT,
  responsible_dept_id INTEGER REFERENCES department(id),
  deleted             INTEGER DEFAULT 0,
  create_time         TEXT DEFAULT (datetime('now','localtime'))
);

-- 质量记录表（批次级，时间序列的聚合载体）
CREATE TABLE IF NOT EXISTS quality_record (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id       INTEGER NOT NULL REFERENCES product(id),
  batch_id         INTEGER NOT NULL REFERENCES product_batch(id),
  record_date      TEXT NOT NULL,
  total_quantity   INTEGER NOT NULL,
  defect_quantity  INTEGER NOT NULL,
  pass_quantity    INTEGER NOT NULL,
  pass_rate        REAL NOT NULL,              -- %
  ppm              REAL,                       -- 百万缺陷率
  deleted          INTEGER DEFAULT 0,
  create_time      TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_qr_product_date ON quality_record(product_id, record_date);

-- 缺陷明细表
CREATE TABLE IF NOT EXISTS defect_detail (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  quality_record_id INTEGER NOT NULL REFERENCES quality_record(id),
  defect_code_id    INTEGER NOT NULL REFERENCES defect_code(id),
  defect_quantity   INTEGER NOT NULL
);

-- 检验实测值表（SPC 控制图数据源）
CREATE TABLE IF NOT EXISTS inspection_value (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id     INTEGER NOT NULL REFERENCES product(id),
  inspection_date TEXT NOT NULL,
  characteristic TEXT NOT NULL,               -- 特性名，如 轴径
  nominal        REAL,
  lsl            REAL,
  usl            REAL,
  sample_no      INTEGER,                     -- 当日第几个样本 1..5
  measured_value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_iv_product_date ON inspection_value(product_id, inspection_date);

-- =========== ★ 事件表（E-QTSF 模型核心输入） ===========
-- 六大类：equipment(设备)/tooling(刀具)/material(物料)/process(工艺)/personnel(人员)/measure(质量措施)
CREATE TABLE IF NOT EXISTS quality_event (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  event_no            TEXT UNIQUE NOT NULL,   -- EVT-2024-00001
  product_id          INTEGER NOT NULL REFERENCES product(id),
  event_date          TEXT NOT NULL,
  event_type          TEXT NOT NULL,          -- 六大类代码
  event_subtype       TEXT NOT NULL,          -- 子类型，如 刀具磨损更换
  severity            TEXT NOT NULL,          -- 致命/严重/一般/轻微
  description         TEXT,                   -- 中文描述（展示/未来接 LLM 嵌入）
  responsible_dept_id INTEGER REFERENCES department(id),
  related_batch_id    INTEGER REFERENCES product_batch(id),
  deleted             INTEGER DEFAULT 0,
  create_time         TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_event_product_date ON quality_event(product_id, event_date);

-- 不合格品报告（NCR 简化闭环）
CREATE TABLE IF NOT EXISTS nonconforming_report (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  report_no         TEXT UNIQUE NOT NULL,     -- NCR-2024-0001
  product_id        INTEGER NOT NULL REFERENCES product(id),
  batch_no          TEXT,
  defect_code_id    INTEGER REFERENCES defect_code(id),
  defect_quantity   INTEGER,
  severity          TEXT,                     -- 致命/严重/一般/轻微
  discovery_date    TEXT,
  responsible_dept_id INTEGER REFERENCES department(id),
  reporter_id       INTEGER REFERENCES sys_user(id),
  description       TEXT,
  disposition       TEXT,                     -- 返工/报废/让步接收
  status            TEXT DEFAULT '待评审',    -- 待评审/处理中/已关闭/已驳回
  close_date        TEXT,
  remark            TEXT,
  deleted           INTEGER DEFAULT 0,
  create_time       TEXT DEFAULT (datetime('now','localtime'))
);

-- NCR 状态变更日志
CREATE TABLE IF NOT EXISTS ncr_status_log (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  ncr_id      INTEGER NOT NULL REFERENCES nonconforming_report(id),
  from_status TEXT,
  to_status   TEXT,
  operator_id INTEGER REFERENCES sys_user(id),
  opinion     TEXT,
  create_time TEXT DEFAULT (datetime('now','localtime'))
);

-- =========== ★ 预测结果表（E-QTSF 推理输出落库） ===========
CREATE TABLE IF NOT EXISTS forecast_result (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id    INTEGER NOT NULL REFERENCES product(id),
  forecast_date TEXT NOT NULL,                -- 预测生成日
  target_date   TEXT NOT NULL,                -- 预测目标日
  horizon       INTEGER,                      -- 预测步长（天）
  y_pred        REAL,                         -- 合格率点预测 %
  y_lower       REAL,                         -- 95% 区间下界
  y_upper       REAL,                         -- 95% 区间上界
  exceed_prob   REAL,                         -- 跌破预警阈值概率
  model_version TEXT,
  create_time   TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_fc_product_date ON forecast_result(product_id, forecast_date);

-- =========== ★ 预警表（预测预警/SPC预警/重复NCR预警） ===========
CREATE TABLE IF NOT EXISTS quality_alert (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id        INTEGER NOT NULL REFERENCES product(id),
  alert_date        TEXT NOT NULL,
  alert_type        TEXT NOT NULL,            -- FORECAST/SPC/REPEAT_NCR
  level             TEXT,                     -- 红/橙/黄
  message           TEXT,
  attribution       TEXT,                     -- 事件归因 JSON：[{event_id,weight,desc}]
  related_event_ids TEXT,
  status            TEXT DEFAULT '未处理',    -- 未处理/已确认/已生成NCR/已忽略
  ncr_id            INTEGER REFERENCES nonconforming_report(id),
  create_time       TEXT DEFAULT (datetime('now','localtime'))
);

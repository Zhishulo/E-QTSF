/* 机械加工质量管理系统 前端 SPA（Vue3 全局构建 + Element Plus + ECharts） */
const { createApp, ref, reactive, computed, onMounted, nextTick, watch } = Vue;
const { ElMessage, ElMessageBox } = ElementPlus;

/* ---------------- API ---------------- */
const token = () => localStorage.getItem("qms_token") || "";
async function api(path, opts = {}) {
  const r = await fetch("/api" + path, {
    method: opts.method || "GET",
    headers: Object.assign(
      { "Content-Type": "application/json" },
      token() ? { Authorization: "Bearer " + token() } : {}),
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (r.status === 401) {
    localStorage.removeItem("qms_token");
    localStorage.removeItem("qms_user");
    location.hash = "";
    throw new Error("请重新登录");
  }
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
}

/* ---------------- 根组件 ---------------- */
const App = {
  setup() {
    const user = ref(JSON.parse(localStorage.getItem("qms_user") || "null"));
    const route = ref(location.hash.slice(1) || "dashboard");
    const menus = [
      { key: "dashboard", icon: "📊", label: "质量看板", roles: null },
      { key: "products", icon: "📦", label: "产品管理", roles: null },
      { key: "batches", icon: "🗂", label: "批次管理", roles: null },
      { key: "records", icon: "📋", label: "质量记录", roles: null },
      { key: "events", icon: "⚡", label: "事件管理", roles: null },
      { key: "ncr", icon: "🚨", label: "不合格品 NCR", roles: null },
      { key: "spc", icon: "📈", label: "SPC 控制图", roles: null },
      { key: "forecast", icon: "🔮", label: "预测预警", roles: null },
      { key: "users", icon: "👤", label: "用户管理", roles: ["ADMIN"] },
    ];
    const visibleMenus = computed(() =>
      menus.filter(m => !m.roles ||
        (user.value && m.roles.includes(user.value.role))));
    window.addEventListener("hashchange",
      () => { route.value = location.hash.slice(1) || "dashboard"; });
    function go(key) { location.hash = key; }
    function logout() {
      localStorage.removeItem("qms_token");
      localStorage.removeItem("qms_user");
      user.value = null;
    }
    function onLogin(u) { user.value = u; location.hash = "dashboard"; }
    return { user, route, visibleMenus, go, logout, onLogin };
  },
  template: `
  <login-view v-if="!user" @login="onLogin"></login-view>
  <div v-else class="layout">
    <div class="side">
      <div class="logo">机械加工 QMS<small>E-QTSF 预测驱动</small></div>
      <div v-for="m in visibleMenus" :key="m.key" class="menu-item"
           :class="{active: route===m.key}" @click="go(m.key)">
        <span>{{m.icon}}</span><span>{{m.label}}</span>
      </div>
    </div>
    <div class="main">
      <div class="topbar">
        <b>{{ (visibleMenus.find(m=>m.key===route)||{}).label || '' }}</b>
        <div>
          <span style="margin-right:12px;color:#606266">
            {{ user.real_name }}（{{ user.role }}）</span>
          <el-button size="small" @click="logout">退出</el-button>
        </div>
      </div>
      <div class="content">
        <dashboard-view v-if="route==='dashboard'"></dashboard-view>
        <products-view v-else-if="route==='products'"></products-view>
        <batches-view v-else-if="route==='batches'"></batches-view>
        <records-view v-else-if="route==='records'"></records-view>
        <events-view v-else-if="route==='events'"></events-view>
        <ncr-view v-else-if="route==='ncr'"></ncr-view>
        <spc-view v-else-if="route==='spc'"></spc-view>
        <forecast-view v-else-if="route==='forecast'"></forecast-view>
        <users-view v-else-if="route==='users'"></users-view>
      </div>
    </div>
  </div>`,
};

/* ---------------- 登录 ---------------- */
const LoginView = {
  emits: ["login"],
  setup(_, { emit }) {
    const form = reactive({ username: "admin", password: "" });
    const loading = ref(false);
    async function submit() {
      loading.value = true;
      try {
        const j = await api("/auth/login", { method: "POST", body: form });
        localStorage.setItem("qms_token", j.token);
        localStorage.setItem("qms_user", JSON.stringify(j.user));
        emit("login", j.user);
        ElMessage.success("登录成功");
      } catch (e) { ElMessage.error(e.message); }
      loading.value = false;
    }
    return { form, loading, submit };
  },
  template: `
  <div class="login-box">
    <div class="login-title">机械加工质量管理系统</div>
    <div class="login-sub">E-QTSF 事件感知质量时序预测 · 预测驱动闭环</div>
    <el-form @submit.prevent>
      <el-form-item><el-input v-model="form.username" placeholder="用户名" /></el-form-item>
      <el-form-item><el-input v-model="form.password" type="password"
        placeholder="密码（默认 123456）" show-password @keyup.enter="submit" /></el-form-item>
      <el-button type="primary" style="width:100%" :loading="loading"
                 @click="submit">登 录</el-button>
    </el-form>
  </div>`,
};

/* ---------------- 质量看板 ---------------- */
const DashboardView = {
  setup() {
    const data = ref(null);
    const trendEl = ref(null), defectEl = ref(null), deptEl = ref(null),
          rankEl = ref(null);
    onMounted(async () => {
      data.value = await api("/dashboard");
      await nextTick();
      const t = data.value.trend;
      echarts.init(trendEl.value).setOption({
        tooltip: { trigger: "axis" },
        xAxis: { type: "category", data: t.map(x => x.date) },
        yAxis: [{ type: "value", name: "合格率%", min: 80, max: 100 },
                { type: "value", name: "PPM" }],
        series: [
          { name: "合格率", type: "line", smooth: true, showSymbol: false,
            data: t.map(x => x.pass_rate), itemStyle: { color: "#4062BB" } },
          { name: "PPM", type: "line", yAxisIndex: 1, showSymbol: false,
            data: t.map(x => x.ppm), itemStyle: { color: "#B5442D" } },
        ],
      });
      echarts.init(defectEl.value).setOption({
        tooltip: {},
        xAxis: { type: "category",
                 data: data.value.top_defects.map(d => d.defect_name),
                 axisLabel: { rotate: 20 } },
        yAxis: { type: "value" },
        series: [{ type: "bar", data: data.value.top_defects.map(d => d.qty),
                   itemStyle: { color: "#B5442D" } }],
      });
      echarts.init(deptEl.value).setOption({
        tooltip: {},
        series: [{ type: "pie", radius: ["40%", "68%"],
                   data: data.value.dept_stats.map(d =>
                     ({ name: d.dept_name, value: d.qty })) }],
      });
      const r = data.value.product_rank;
      echarts.init(rankEl.value).setOption({
        tooltip: {},
        grid: { left: 90 },
        xAxis: { type: "value", min: 80, max: 100 },
        yAxis: { type: "category", data: r.map(x => x.product_name) },
        series: [{ type: "bar", data: r.map(x => x.pass_rate),
                   itemStyle: { color: "#4C721D" } }],
      });
      window.addEventListener("resize", () =>
        document.querySelectorAll(".chart").forEach(el => {
          const i = echarts.getInstanceByDom(el); i && i.resize();
        }));
    });
    return { data, trendEl, defectEl, deptEl, rankEl };
  },
  template: `
  <div v-if="data">
    <div class="cards">
      <div class="card"><div class="v">{{data.summary.products}}</div><div class="t">在管产品</div></div>
      <div class="card"><div class="v">{{data.summary.ncr_open}}</div><div class="t">未关闭 NCR</div></div>
      <div class="card"><div class="v">{{data.summary.events_30d}}</div><div class="t">近30天质量事件</div></div>
      <div class="card"><div class="v" :class="data.summary.alerts? 'alert-red':''">
        {{data.summary.alerts}}</div><div class="t">待处理预测预警</div></div>
    </div>
    <div class="panel"><h3>合格率 / PPM 趋势（近90天）</h3><div ref="trendEl" class="chart"></div></div>
    <div class="row2">
      <div class="panel"><h3>缺陷 Top5</h3><div ref="defectEl" class="chart"></div></div>
      <div class="panel"><h3>部门缺陷分布</h3><div ref="deptEl" class="chart"></div></div>
    </div>
    <div class="panel"><h3>产品合格率排行</h3><div ref="rankEl" class="chart"></div></div>
  </div>`,
};

/* ---------------- 通用表格页 ---------------- */
function makeTable(cfg) {
  return {
    setup() {
      const rows = ref([]);
      const q = ref("");
      const loading = ref(false);
      const dialog = ref(false);
      const form = reactive({});
      const isEdit = ref(false);
      async function load() {
        loading.value = true;
        rows.value = await api("/" + cfg.api);
        loading.value = false;
      }
      onMounted(load);
      const filtered = computed(() => {
        if (!q.value) return rows.value;
        const k = q.value.toLowerCase();
        return rows.value.filter(r =>
          cfg.searchCols.some(c =>
            String(r[c] || "").toLowerCase().includes(k)));
      });
      function openAdd() {
        isEdit.value = false;
        Object.keys(form).forEach(k => delete form[k]);
        (cfg.formFields || []).forEach(f => form[f.key] = f.def ?? null);
        dialog.value = true;
      }
      function openEdit(row) {
        isEdit.value = true;
        Object.keys(form).forEach(k => delete form[k]);
        Object.assign(form, row);
        dialog.value = true;
      }
      async function save() {
        try {
          if (isEdit.value) {
            await api(`/${cfg.api}/${form.id}`,
                      { method: "PUT", body: { fields: form } });
          } else {
            await api(`/${cfg.api}`, { method: "POST", body: { fields: form } });
          }
          ElMessage.success("已保存");
          dialog.value = false;
          load();
        } catch (e) { ElMessage.error(e.message); }
      }
      async function del(row) {
        try {
          await ElMessageBox.confirm("确认删除该记录？", "提示",
                                     { type: "warning" });
          await api(`/${cfg.api}/${row.id}`, { method: "DELETE" });
          ElMessage.success("已删除");
          load();
        } catch (e) { /* cancel */ }
      }
      return { rows, q, loading, dialog, form, isEdit, filtered,
               openAdd, openEdit, save, del, cfg };
    },
    template: `
    <div class="panel">
      <div class="toolbar">
        <el-input v-model="q" placeholder="搜索" style="width:220px" clearable />
        <el-button v-if="cfg.formFields && cfg.formFields.length"
                   type="primary" @click="openAdd">新增</el-button>
        <el-button @click="cfg.api? load() : null">刷新</el-button>
        <span style="color:#909399;font-size:12px">{{ filtered.length }} 条</span>
      </div>
      <el-table :data="filtered" v-loading="loading" size="small" border stripe>
        <el-table-column v-for="c in cfg.cols" :key="c.key" :prop="c.key"
          :label="c.label" :width="c.width" />
        <el-table-column v-if="cfg.formFields && cfg.formFields.length"
                         label="操作" width="140">
          <template #default="{ row }">
            <el-button link type="primary" @click="openEdit(row)">编辑</el-button>
            <el-button link type="danger" @click="del(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <el-dialog v-model="dialog" :title="isEdit? '编辑':'新增'" width="480px">
      <el-form label-width="90px">
        <el-form-item v-for="f in cfg.formFields" :key="f.key" :label="f.label">
          <el-select v-if="f.options" v-model="form[f.key]" style="width:100%">
            <el-option v-for="o in f.options" :key="o" :label="o" :value="o" />
          </el-select>
          <el-input v-else v-model="form[f.key]" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialog=false">取消</el-button>
        <el-button type="primary" @click="save">保存</el-button>
      </template>
    </el-dialog>`,
  };
}

const ProductsView = makeTable({
  api: "products",
  cols: [
    { key: "product_code", label: "编号", width: 110 },
    { key: "product_name", label: "名称" },
    { key: "product_type", label: "大类", width: 90 },
    { key: "specification", label: "规格" },
    { key: "material", label: "材料" },
    { key: "unit", label: "单位", width: 70 },
    { key: "status", label: "状态", width: 90 },
  ],
  searchCols: ["product_code", "product_name", "specification"],
  formFields: [
    { key: "product_name", label: "名称" },
    { key: "product_type", label: "大类",
      options: ["轴类", "盘类", "套类", "箱体类", "杂类"] },
    { key: "specification", label: "规格" },
    { key: "material", label: "材料" },
    { key: "unit", label: "单位", def: "件" },
    { key: "status", label: "状态",
      options: ["生产中", "试制", "已停产"], def: "生产中" },
  ],
});

const DefectCodesView = makeTable({
  api: "defect-codes",
  cols: [
    { key: "defect_code", label: "代码", width: 110 },
    { key: "defect_name", label: "名称" },
    { key: "category", label: "分类", width: 90 },
    { key: "severity", label: "等级", width: 90 },
    { key: "description", label: "描述" },
  ],
  searchCols: ["defect_code", "defect_name", "category"],
  formFields: [
    { key: "defect_name", label: "名称" },
    { key: "category", label: "分类",
      options: ["尺寸", "外观", "性能", "包装", "其他"] },
    { key: "severity", label: "等级",
      options: ["致命", "严重", "一般", "轻微"] },
    { key: "description", label: "描述" },
  ],
});

const BatchesView = makeTable({
  api: "batches",
  cols: [
    { key: "batch_no", label: "批次号", width: 170 },
    { key: "product_code", label: "产品", width: 100 },
    { key: "product_name", label: "名称" },
    { key: "production_date", label: "生产日期", width: 120 },
    { key: "total_quantity", label: "计划产量", width: 110 },
  ],
  searchCols: ["batch_no", "product_name"],
  formFields: [],
});

const RecordsView = makeTable({
  api: "quality-records",
  cols: [
    { key: "record_date", label: "日期", width: 110 },
    { key: "product_code", label: "产品", width: 100 },
    { key: "batch_no", label: "批次", width: 170 },
    { key: "total_quantity", label: "总数", width: 90 },
    { key: "defect_quantity", label: "缺陷", width: 90 },
    { key: "pass_quantity", label: "合格", width: 90 },
    { key: "pass_rate", label: "合格率%", width: 100 },
    { key: "ppm", label: "PPM", width: 100 },
  ],
  searchCols: ["product_code", "batch_no"],
  formFields: [],
});

/* ---------------- 事件管理（含计划事件登记 → 参与预测 what-if） ---------------- */
const EventsView = {
  setup() {
    const rows = ref([]);
    const q = ref("");
    const dialog = ref(false);
    const products = ref([]);
    const form = reactive({ product_id: null, event_date: "",
                            event_type: "equipment", event_subtype: "repair",
                            severity: "一般", description: "" });
    const SUBTYPES = {
      equipment: ["repair", "maintain"],
      tooling: ["tool_change"],
      material: ["batch_change", "supplier_change"],
      process: ["param_adjust", "fixture_adjust"],
      personnel: ["operator_change", "new_operator"],
      measure: ["capa", "strict_inspect", "poka_yoke"],
    };
    const TYPE_NAMES = {
      equipment: "设备类", tooling: "刀具类", material: "物料类",
      process: "工艺类", personnel: "人员类", measure: "质量措施类",
    };
    watch(() => form.event_type,
          () => { form.event_subtype = SUBTYPES[form.event_type][0]; });
    async function load() {
      rows.value = await api("/events");
    }
    async function openAdd() {
      products.value = await api("/products");
      if (products.value.length) form.product_id = products.value[0].id;
      // 默认日期：数据最后一天 + 7 天（预测窗内）
      const last = rows.value.length ? rows.value[0].event_date : null;
      if (last) {
        const d = new Date(last);
        d.setDate(d.getDate() + 1);
        form.event_date = d.toISOString().slice(0, 10);
      }
      dialog.value = true;
    }
    async function save() {
      try {
        await api("/events", { method: "POST", body: { fields: form } });
        ElMessage.success("计划事件已登记，将参与下次预测（what-if）");
        dialog.value = false;
        load();
      } catch (e) { ElMessage.error(e.message); }
    }
    const filtered = computed(() => {
      if (!q.value) return rows.value;
      const k = q.value.toLowerCase();
      return rows.value.filter(r =>
        ["event_no", "event_subtype", "product_name", "event_type"]
          .some(c => String(r[c] || "").toLowerCase().includes(k)));
    });
    return { rows, q, dialog, products, form, SUBTYPES, TYPE_NAMES,
             load, openAdd, save, filtered };
  },
  template: `
  <div class="panel">
    <div class="toolbar">
      <el-input v-model="q" placeholder="搜索" style="width:220px" clearable />
      <el-button type="primary" @click="openAdd">登记计划事件</el-button>
      <el-button @click="load">刷新</el-button>
      <el-tag type="info">计划事件将作为 E-QTSF 未来条件参与预测（what-if）</el-tag>
    </div>
    <el-table :data="filtered" size="small" border stripe max-height="560">
      <el-table-column prop="event_no" label="事件号" width="150" />
      <el-table-column prop="event_date" label="日期" width="110" />
      <el-table-column prop="product_name" label="产品" width="100" />
      <el-table-column prop="event_type" label="大类" width="100" />
      <el-table-column prop="event_subtype" label="子类型" width="130" />
      <el-table-column prop="severity" label="严重度" width="90">
        <template #default="{row}">
          <el-tag :type="row.severity==='致命'?'danger':row.severity==='严重'?'warning':'info'"
                  size="small">{{ row.severity }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="dept_name" label="责任部门" width="100" />
      <el-table-column prop="description" label="描述" min-width="180"
                       show-overflow-tooltip />
    </el-table>
  </div>
  <el-dialog v-model="dialog" title="登记计划事件（what-if）" width="460px">
    <el-form label-width="90px">
      <el-form-item label="产品">
        <el-select v-model="form.product_id" style="width:100%">
          <el-option v-for="p in products" :key="p.id" :value="p.id"
            :label="p.product_code + ' ' + p.product_name" />
        </el-select>
      </el-form-item>
      <el-form-item label="日期">
        <el-input v-model="form.event_date" placeholder="YYYY-MM-DD" />
      </el-form-item>
      <el-form-item label="事件大类">
        <el-select v-model="form.event_type" style="width:100%">
          <el-option v-for="(v, k) in TYPE_NAMES" :key="k" :value="k" :label="v" />
        </el-select>
      </el-form-item>
      <el-form-item label="子类型">
        <el-select v-model="form.event_subtype" style="width:100%">
          <el-option v-for="s in SUBTYPES[form.event_type]" :key="s" :value="s" />
        </el-select>
      </el-form-item>
      <el-form-item label="严重度">
        <el-select v-model="form.severity" style="width:100%">
          <el-option v-for="s in ['致命','严重','一般','轻微']" :key="s" :value="s" />
        </el-select>
      </el-form-item>
      <el-form-item label="描述">
        <el-input v-model="form.description" type="textarea" />
      </el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="dialog=false">取消</el-button>
      <el-button type="primary" @click="save">登记</el-button>
    </template>
  </el-dialog>`,
};

/* ---------------- NCR ---------------- */
const NcrView = {
  setup() {
    const rows = ref([]);
    const dialog = ref(false);
    const logDialog = ref(false);
    const logs = ref([]);
    const form = reactive({ product_id: null, defect_code_id: null,
                            defect_quantity: 0, severity: "一般",
                            description: "", disposition: "返工" });
    const products = ref([]);
    const codes = ref([]);
    const NCR_NAMES = { submit: "提交评审", reject: "评审驳回",
                        verify_pass: "验证通过", verify_fail: "验证不通过" };
    async function load() {
      rows.value = await api("/ncr");
      products.value = await api("/products");
      codes.value = await api("/defect-codes");
    }
    onMounted(load);
    async function create() {
      try {
        await api("/ncr", { method: "POST", body: form });
        ElMessage.success("已创建，状态：待评审");
        dialog.value = false;
        load();
      } catch (e) { ElMessage.error(e.message); }
    }
    async function act(row, action) {
      const { value } = await ElMessageBox.prompt(
        "处理意见", NCR_NAMES[action], { inputValue: "" });
      await api(`/ncr/${row.id}/action`,
                { method: "POST", body: { action, opinion: value } });
      ElMessage.success("状态已更新");
      load();
    }
    async function showLogs(row) {
      logs.value = await api(`/ncr/${row.id}/logs`);
      logDialog.value = true;
    }
    function actionsFor(row) {
      const map = {
        "待评审": ["submit", "reject"],
        "处理中": ["verify_pass", "verify_fail"],
        "已驳回": ["close"],
      };
      return map[row.status] || [];
    }
    return { rows, dialog, logDialog, logs, form, products, codes, load,
             create, act, showLogs, actionsFor, NCR_NAMES };
  },
  template: `
  <div class="panel">
    <div class="toolbar">
      <el-button type="danger" @click="dialog=true">创建 NCR</el-button>
      <el-button @click="load">刷新</el-button>
      <el-tag type="info">闭环：待评审 → 处理中 → 已关闭 / 已驳回</el-tag>
    </div>
    <el-table :data="rows" size="small" border stripe max-height="560">
      <el-table-column prop="report_no" label="编号" width="150" />
      <el-table-column prop="product_name" label="产品" width="100" />
      <el-table-column prop="defect_name" label="缺陷" width="110" />
      <el-table-column prop="defect_quantity" label="数量" width="70" />
      <el-table-column prop="severity" label="等级" width="80" />
      <el-table-column prop="discovery_date" label="发现日" width="105" />
      <el-table-column prop="status" label="状态" width="90">
        <template #default="{row}">
          <el-tag :type="row.status==='已关闭'?'success':
                         row.status==='处理中'?'warning':'info'" size="small">
            {{ row.status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="disposition" label="处置" width="90" />
      <el-table-column prop="description" label="描述" min-width="160"
                       show-overflow-tooltip />
      <el-table-column label="操作" width="240">
        <template #default="{row}">
          <el-button v-for="a in actionsFor(row)" :key="a" link type="primary"
            @click="act(row, a)">{{ NCR_NAMES[a] }}</el-button>
          <el-button link @click="showLogs(row)">日志</el-button>
        </template>
      </el-table-column>
    </el-table>
  </div>
  <el-dialog v-model="dialog" title="创建不合格品报告" width="480px">
    <el-form label-width="90px">
      <el-form-item label="产品">
        <el-select v-model="form.product_id" style="width:100%">
          <el-option v-for="p in products" :key="p.id" :value="p.id"
            :label="p.product_code + ' ' + p.product_name" />
        </el-select>
      </el-form-item>
      <el-form-item label="缺陷代码">
        <el-select v-model="form.defect_code_id" clearable style="width:100%">
          <el-option v-for="c in codes" :key="c.id" :value="c.id"
            :label="c.defect_code + ' ' + c.defect_name" />
        </el-select>
      </el-form-item>
      <el-form-item label="缺陷数量">
        <el-input-number v-model="form.defect_quantity" :min="0" />
      </el-form-item>
      <el-form-item label="严重度">
        <el-select v-model="form.severity">
          <el-option v-for="s in ['致命','严重','一般','轻微']" :key="s" :value="s" />
        </el-select>
      </el-form-item>
      <el-form-item label="处置">
        <el-select v-model="form.disposition">
          <el-option v-for="s in ['返工','返修','报废','让步接收']" :key="s" :value="s" />
        </el-select>
      </el-form-item>
      <el-form-item label="描述">
        <el-input v-model="form.description" type="textarea" />
      </el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="dialog=false">取消</el-button>
      <el-button type="primary" @click="create">创建</el-button>
    </template>
  </el-dialog>
  <el-dialog v-model="logDialog" title="状态变更日志" width="560px">
    <el-timeline>
      <el-timeline-item v-for="l in logs" :key="l.id" :timestamp="l.create_time"
        :type="l.to_status==='已关闭'?'success':''">
        <b>{{ l.from_status || '∅' }} → {{ l.to_status }}</b>
        <span style="color:#909399">（{{ l.real_name || '系统' }}）</span>
        <div style="color:#606266;font-size:12px">{{ l.opinion }}</div>
      </el-timeline-item>
    </el-timeline>
  </el-dialog>`,
};

/* ---------------- SPC ---------------- */
const SpcView = {
  setup() {
    const products = ref([]);
    const pid = ref(null);
    const data = ref(null);
    const xEl = ref(null), rEl = ref(null), pEl = ref(null);
    onMounted(async () => {
      products.value = await api("/products");
      if (products.value.length) pid.value = products.value[0].id;
    });
    watch(pid, load);
    async function load() {
      if (!pid.value) return;
      data.value = await api(`/spc/${pid.value}?days=60`);
      await nextTick();
      const d = data.value;
      const mk = (el, series, limits, name) => {
        echarts.init(el).setOption({
          tooltip: { trigger: "axis" },
          grid: { left: 60, right: 20 },
          xAxis: { type: "category", data: d.dates },
          yAxis: { type: "value", scale: true },
          series: [
            { name, type: "line", data: series, showSymbol: false,
              lineStyle: { color: "#4062BB" } },
            { name: "控制限", type: "line", data: [],
              markLine: { silent: true, symbol: "none",
                data: [{ yAxis: limits.ucl, lineStyle: { color: "#f56c6c" },
                         label: { formatter: "UCL" } },
                       { yAxis: limits.cl, lineStyle: { color: "#909399", type: "dashed" },
                         label: { formatter: "CL" } },
                       { yAxis: limits.lcl, lineStyle: { color: "#f56c6c" },
                         label: { formatter: "LCL" } }] } },
          ],
        });
      };
      mk(xEl.value, d.xbar, d.spc_x, "X̄");
      mk(rEl.value, d.range, d.spc_r, "R");
      echarts.init(pEl.value).setOption({
        tooltip: { trigger: "axis" },
        legend: { data: ["不合格率%"] },
        xAxis: { type: "category", data: d.p_chart.map(x => x.date) },
        yAxis: { type: "value", scale: true },
        series: [
          { name: "不合格率%", type: "line", showSymbol: false,
            data: d.p_chart.map(x => x.p) },
          { name: "UCL", type: "line", showSymbol: false,
            data: d.p_chart.map(x => x.ucl),
            lineStyle: { color: "#f56c6c", type: "dashed" } },
          { name: "LCL", type: "line", showSymbol: false,
            data: d.p_chart.map(x => x.lcl),
            lineStyle: { color: "#f56c6c", type: "dashed" } },
        ],
      });
      setTimeout(() => window.dispatchEvent(new Event("resize")), 50);
    }
    return { products, pid, data, xEl, rEl, pEl, load };
  },
  template: `
  <div class="panel">
    <div class="toolbar">
      <span>产品：</span>
      <el-select v-model="pid" style="width:220px">
        <el-option v-for="p in products" :key="p.id" :value="p.id"
          :label="p.product_code + ' ' + p.product_name" />
      </el-select>
      <el-tag v-if="data">PPM：{{ data.ppm }}（{{ data.pbar_pct }}% 不合格）</el-tag>
      <el-tag v-if="data && data.ooc_xbar>0" type="danger">
        X̄ 图失控点 {{ data.ooc_xbar }} 个</el-tag>
    </div>
    <h3>X̄ 控制图（每日 5 实测值均值）</h3><div ref="xEl" class="chart"></div>
    <h3>R 控制图（极差）</h3><div ref="rEl" class="chart"></div>
    <h3>P 控制图（批不合格率）</h3><div ref="pEl" class="chart"></div>
  </div>`,
};

/* ---------------- 预测预警 ---------------- */
const ForecastView = {
  setup() {
    const products = ref([]);
    const pid = ref(null);
    const running = ref(false);
    const result = ref(null);
    const alerts = ref([]);
    const chartEl = ref(null);
    onMounted(async () => {
      products.value = await api("/products");
      if (products.value.length) pid.value = products.value[0].id;
      alerts.value = await api("/alerts");
    });
    watch(pid, loadLatest);
    async function loadLatest() {
      if (!pid.value) return;
      const j = await api(`/forecast/latest/${pid.value}`);
      if (!j.empty) {
        const rows = j.rows;
        result.value = {
          median: rows.map(r => r.y_pred), q10: rows.map(r => r.y_lower),
          q90: rows.map(r => r.y_upper), dates: rows.map(r => r.target_date),
          exceed_prob: rows.map(r => r.exceed_prob),
          forecast_date: rows[0].forecast_date,
          threshold: null, level: null, message: null,
          attribution: [],
          history: [],
        };
        render();
      }
    }
    async function run() {
      running.value = true;
      try {
        result.value = await api(`/forecast/run/${pid.value}`,
                                 { method: "POST" });
        ElMessage.success("预测完成");
        alerts.value = await api("/alerts");
        render();
      } catch (e) { ElMessage.error(e.message); }
      running.value = false;
    }
    function render() {
      const r = result.value;
      if (!r || !chartEl.value) return;
      nextTick(() => {
        const hist = r.history || [];
        const hDates = hist.map(h => h.date);
        const hVals = hist.map(h => h.pass_rate);
        const pad = n => Array(n).fill(null);
        echarts.init(chartEl.value).setOption({
          tooltip: { trigger: "axis" },
          legend: { data: ["历史", "中位数预测", "80%区间", "超标概率"] },
          xAxis: { type: "category", data: hDates.concat(r.dates) },
          yAxis: [{ type: "value", scale: true, name: "合格率%" },
                  { type: "value", name: "P(超标)", max: 1 }],
          series: [
            { name: "历史", type: "line", showSymbol: false,
              data: hVals.concat(pad(r.dates.length)),
              lineStyle: { color: "#5A5A5A" } },
            { name: "中位数预测", type: "line", showSymbol: false,
              data: pad(Math.max(hDates.length - 1, 0))
                    .concat(hDates.length ? [hVals[hVals.length - 1]] : [])
                    .concat(r.median),
              lineStyle: { color: "#D62728", type: "dashed" } },
            { name: "80%区间", type: "line", stack: "band", symbol: "none",
              data: pad(Math.max(hDates.length - 1, 0))
                    .concat(hDates.length ? [0] : []).concat(r.q10),
              lineStyle: { opacity: 0 } },
            { name: "80%区间", type: "line", stack: "band", symbol: "none",
              data: pad(Math.max(hDates.length - 1, 0))
                    .concat(hDates.length ? [0] : [])
                    .concat(r.q90.map((v, i) => +(v - r.q10[i]).toFixed(3))),
              lineStyle: { opacity: 0 },
              areaStyle: { color: "#D62728", opacity: 0.15 } },
            { name: "超标概率", type: "bar", yAxisIndex: 1, barWidth: "40%",
              data: pad(hDates.length).concat(r.exceed_prob),
              itemStyle: { color: "#F3A712", opacity: 0.6 } },
          ],
        });
      });
    }
    async function handle(a, action) {
      if (action === "create_ncr") {
        await ElMessageBox.confirm("确认为该预警自动生成 NCR？", "提示");
      }
      await api(`/alerts/${a.id}/handle`, { method: "POST", body: { action } });
      ElMessage.success("已处理");
      alerts.value = await api("/alerts");
    }
    return { products, pid, running, result, alerts, chartEl, run, handle };
  },
  template: `
  <div class="panel">
    <div class="toolbar">
      <span>产品：</span>
      <el-select v-model="pid" style="width:220px">
        <el-option v-for="p in products" :key="p.id" :value="p.id"
          :label="p.product_code + ' ' + p.product_name" />
      </el-select>
      <el-button type="primary" :loading="running" @click="run">
        运行 E-QTSF 预测</el-button>
      <el-tag v-if="result && result.level"
        :type="result.level==='红'?'danger':result.level==='橙'?'warning':'info'">
        {{ result.level }}级预警：{{ result.message }}</el-tag>
      <el-tag v-if="result && result.threshold" type="info">
        阈值 {{ result.threshold }}%</el-tag>
      <el-tag type="info">提示：在「事件管理」登记计划事件后，预测将纳入未来条件（what-if）</el-tag>
    </div>
    <div ref="chartEl" class="chart" style="height:360px"></div>
    <el-collapse v-if="result && result.attribution && result.attribution.length">
      <el-collapse-item title="事件归因（未来28天，|Δ| = 有事件 vs 无事件预测差）">
        <el-table :data="result.attribution" size="small">
          <el-table-column prop="date" label="日期" width="120" />
          <el-table-column prop="event" label="事件" />
          <el-table-column prop="delta" label="影响强度 |Δ|" width="140" />
        </el-table>
      </el-collapse-item>
    </el-collapse>
  </div>
  <div class="panel">
    <h3>预测预警列表</h3>
    <el-table :data="alerts" size="small" border>
      <el-table-column prop="alert_date" label="日期" width="110" />
      <el-table-column prop="product_name" label="产品" width="100" />
      <el-table-column prop="level" label="级别" width="70">
        <template #default="{row}">
          <span :class="{红:'alert-red',橙:'alert-orange',黄:'alert-yellow'}[row.level]">
            {{ row.level }}</span>
        </template>
      </el-table-column>
      <el-table-column prop="alert_type" label="类型" width="110" />
      <el-table-column prop="message" label="内容" min-width="220" />
      <el-table-column prop="status" label="状态" width="100" />
      <el-table-column label="操作" width="200">
        <template #default="{row}">
          <template v-if="row.status==='未处理'">
            <el-button link type="primary"
              @click="handle(row,'confirm')">确认</el-button>
            <el-button link type="danger"
              @click="handle(row,'create_ncr')">生成NCR</el-button>
            <el-button link @click="handle(row,'ignore')">忽略</el-button>
          </template>
        </template>
      </el-table-column>
    </el-table>
  </div>`,
};

/* ---------------- 用户管理 ---------------- */
const UsersView = {
  setup() {
    const rows = ref([]);
    const dialog = ref(false);
    const form = reactive({ username: "", password: "", real_name: "",
                            role: "INSPECTOR", dept_name: "质量部" });
    async function load() { rows.value = await api("/users"); }
    onMounted(load);
    async function add() {
      try {
        await api("/users", { method: "POST", body: form });
        ElMessage.success("已创建");
        dialog.value = false;
        load();
      } catch (e) { ElMessage.error(e.message); }
    }
    return { rows, dialog, form, add };
  },
  template: `
  <div class="panel">
    <div class="toolbar">
      <el-button type="primary" @click="dialog=true">新增用户</el-button>
    </div>
    <el-table :data="rows" size="small" border>
      <el-table-column prop="username" label="用户名" width="120" />
      <el-table-column prop="real_name" label="姓名" width="120" />
      <el-table-column prop="role" label="角色" width="120" />
      <el-table-column prop="dept_name" label="部门" width="120" />
    </el-table>
  </div>
  <el-dialog v-model="dialog" title="新增用户" width="440px">
    <el-form label-width="80px">
      <el-form-item label="用户名"><el-input v-model="form.username" /></el-form-item>
      <el-form-item label="密码">
        <el-input v-model="form.password" type="password" placeholder="默认 123456" />
      </el-form-item>
      <el-form-item label="姓名"><el-input v-model="form.real_name" /></el-form-item>
      <el-form-item label="角色">
        <el-select v-model="form.role">
          <el-option value="ADMIN" label="管理员" />
          <el-option value="INSPECTOR" label="检验员" />
          <el-option value="ENGINEER" label="工程师" />
        </el-select>
      </el-form-item>
      <el-form-item label="部门"><el-input v-model="form.dept_name" /></el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="dialog=false">取消</el-button>
      <el-button type="primary" @click="add">创建</el-button>
    </template>
  </el-dialog>`,
};

/* ---------------- 挂载 ---------------- */
const app = createApp(App);
app.use(ElementPlus);
app.component("login-view", LoginView);
app.component("dashboard-view", DashboardView);
app.component("products-view", ProductsView);
app.component("defect-codes-view", DefectCodesView);
app.component("batches-view", BatchesView);
app.component("records-view", RecordsView);
app.component("events-view", EventsView);
app.component("ncr-view", NcrView);
app.component("spc-view", SpcView);
app.component("forecast-view", ForecastView);
app.component("users-view", UsersView);
app.mount("#app");

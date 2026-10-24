const SVG_NS = "http://www.w3.org/2000/svg";
const ORDERS_KEY = "aicseek-orders-v2";
const PEOPLE_KEY = "aicseek-people-v2";
const ME_KEY = "aicseek-me-v2";
const SPEEDS = [1400, 800, 420];
const SPEED_LABELS = ["慢", "正常", "快"];
const STATUS = { doing: "执行中", review: "待审核", done: "已完成" };
const ACTIONS = { confirm: "已确认", shot: "已拍照", cleared: "已清除", watch: "持续观察", other: "其他处置" };
const WEATHER = { temp: "27°C", humidity: "46%", wind: "6 米/秒", dir: "偏西风", aqi: "42" };
const DEPOT = { lat: 34.42, lng: -119.7, name: "圣巴巴拉市区" };
const SITES = [
  { id: "syp-n", code: "SYP-N", name: "圣伊内斯峰北", line: "圣伊内斯峰走廊", region: "圣巴巴拉县", lat: 34.53, lng: -119.98, elev: 1310, live: true, kind: "固定相机" },
  { id: "syp-s", code: "SYP-S", name: "圣伊内斯峰南", line: "圣伊内斯峰走廊", region: "圣巴巴拉县", lat: 34.512, lng: -119.972, elev: 1280, live: false, tone: "ok", status: "正常", detail: "南向没有单独的连续画面，跟着北向那台相机一起看。" },
  { id: "cach", code: "CCH-S", name: "卡楚马湖南", line: "圣伊内斯峰走廊", region: "圣巴巴拉县", lat: 34.583, lng: -119.858, elev: 240, live: false, tone: "ok", status: "正常", detail: "本包没有这个点的连续画面。" },
  { id: "fig", code: "FIG-1", name: "菲格罗亚", line: "圣伊内斯峰走廊", region: "圣巴巴拉县", lat: 34.738, lng: -120.008, elev: 980, live: false, tone: "ok", status: "正常", detail: "山脊方向，本包没有独立画面。" },
  { id: "lac", code: "LAC-N", name: "拉昆布雷北", line: "圣伊内斯峰走廊", region: "圣巴巴拉县", lat: 34.495, lng: -119.712, elev: 1215, live: false, tone: "ok", status: "正常", detail: "本包没有这个点的连续画面。" },
  { id: "gib", code: "GIB-W", name: "吉布拉塔尔西", line: "圣伊内斯峰走廊", region: "圣巴巴拉县", lat: 34.478, lng: -119.682, elev: 420, live: false, tone: "ok", status: "正常", detail: "本包没有这个点的连续画面。" },
  { id: "ref", code: "REF-2", name: "里弗吉奥", line: "圣伊内斯峰走廊", region: "圣巴巴拉县", lat: 34.442, lng: -119.851, elev: 180, live: false, tone: "ok", status: "正常", detail: "本包没有这个点的连续画面。" },
  { id: "cam", code: "CAM-E", name: "卡马略东", line: "圣伊内斯峰走廊", region: "圣巴巴拉县", lat: 34.398, lng: -119.518, elev: 260, live: false, tone: "ok", status: "正常", detail: "本包没有这个点的连续画面。" },
];
const state = {
  surface: "desk",
  desk: "focus",
  field: "home",
  siteId: "syp-n",
  mode: "standard",
  index: 0,
  sumKey: "",
  playing: false,
  speed: 1,
  layers: { suspect: true, track: true, crop: true },
  orderFilter: "all",
  assetTab: "sites",
  assetQuery: "",
  peopleQuery: "",
  orderId: "",
  personId: "",
  authMode: "login",
  skill: "",
  draftActions: null,
  draftOrderId: "",
  picker: null,
  addForm: "",
  demo: null,
  ready: null,
  gps: null,
  weather: null,
  placeName: "",
  amapKey: "",
  amapSecurity: "",
  maps: {},
  report: { preview: "", filename: "", source: "", file: null, text: "", edge: null, phase: "", message: "" },
  pendingPhoto: null,
  shot: null,
  fieldAdvice: null,
  navId: "",
  talk: { phase: "idle", error: "", messages: [] },
  judge: { sig: "", phase: "idle", word: "暂无异常", headline: "", paragraphs: [], message: "" },
  watch: { status: "idle", done: 0, total: 0, message: "" },
};

let playTimer = 0;
let watchTimer = 0;
let toastTimer = 0;
let pollTimer = 0;

const $ = (id) => document.getElementById(id);
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}
function uid() { return Math.random().toString(16).slice(2, 10); }
function toast(text) {
  const node = $("toast");
  node.textContent = text;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 2200);
}
const board = { rev: 0, orders: [], people: [], sites: [], cameras: [] };
let pushTimer = 0;
let pushing = false;
let dirty = false;

function typing() {
  const tag = document.activeElement && document.activeElement.tagName;
  return tag === "INPUT" || tag === "TEXTAREA";
}
function settleOrders(orders) {
  let changed = false;
  (orders || []).forEach((order) => {
    if (order.status !== "pending") return;
    order.status = "doing";
    if (!order.startedAt) order.startedAt = order.createdAt || Date.now();
    changed = true;
  });
  return changed;
}
function takeBoard(data) {
  board.rev = data.rev;
  board.orders = data.orders || [];
  board.people = data.people || [];
  board.sites = data.sites || [];
  board.cameras = data.cameras || [];
  if (settleOrders(board.orders)) {
    dirty = true;
    schedulePush();
  }
}
function loadOrders() { return board.orders; }
function loadPeople() { return board.people; }
function saveOrders(orders) { board.orders = orders; dirty = true; schedulePush(); }
function savePeople(people) { board.people = people; dirty = true; schedulePush(); }
function savePlaces() { dirty = true; schedulePush(); }
function schedulePush() {
  clearTimeout(pushTimer);
  pushTimer = setTimeout(pushBoard, 180);
}
async function pushBoard() {
  if (pushing) { schedulePush(); return; }
  if (!dirty) return;
  pushing = true;
  dirty = false;
  try {
    const response = await fetch("/api/board", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rev: board.rev, orders: board.orders, people: board.people, sites: board.sites, cameras: board.cameras }),
    });
    const data = await response.json();
    if (response.status === 409) {
      dirty = false;
      takeBoard(data);
      if (!typing()) paint();
      toast("另一端刚更新，已接上最新工单");
    } else if (response.ok) {
      board.rev = data.rev;
      if (dirty) schedulePush();
    } else {
      dirty = true;
    }
  } catch {
    dirty = true;
    schedulePush();
  }
  pushing = false;
}
async function pullBoard() {
  if (pushing || dirty || typing()) return;
  try {
    const response = await fetch("/api/board");
    if (!response.ok) return;
    const data = await response.json();
    if (data.rev === board.rev) return;
    takeBoard(data);
    paint();
  } catch { /* 下一轮再拉 */ }
}
function currentAir() { return state.weather || WEATHER; }
async function loadLive(lat, lng) {
  try {
    const data = await fetch(`/api/live?lat=${encodeURIComponent(lat)}&lng=${encodeURIComponent(lng)}`).then((item) => item.json());
    if (data.weather) state.weather = data.weather;
    if (data.place && data.place.name) state.placeName = data.place.name;
    if (!typing()) paint();
  } catch { /* 天气接口暂时不可用时，页面仍用回放参考 */ }
}
function meId() { return localStorage.getItem(ME_KEY) || ""; }
function setMe(id) { localStorage.setItem(ME_KEY, id); }
function mePerson() {
  const id = meId();
  return id ? loadPeople().find((item) => item.id === id) || null : null;
}
function meName() { return personName(meId()); }
function phoneDigits(value) { return String(value || "").replace(/\D/g, ""); }
function validIp(value) {
  const parts = String(value || "").trim().split(".");
  if (parts.length !== 4) return false;
  return parts.every((part) => /^\d{1,3}$/.test(part) && Number(part) <= 255);
}
function validPhone(value) { return /^1\d{10}$/.test(phoneDigits(value)); }
function rosterPeople() { return loadPeople().filter((person) => person.added); }
function allSites() {
  const known = new Set(SITES.map((site) => site.id));
  return SITES.concat((board.sites || []).filter((site) => site && site.id && !known.has(site.id)));
}
function siteById(id) { return allSites().find((site) => site.id === id) || SITES[0]; }
function loadCameras() { return board.cameras || []; }
function personName(id) {
  const person = loadPeople().find((item) => item.id === id);
  return person ? person.name : "未指定";
}
function km(a, b) {
  const rad = Math.PI / 180;
  const dLat = (b.lat - a.lat) * rad;
  const dLng = (b.lng - a.lng) * rad;
  const s = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLng / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(s));
}
function distText(kmValue) {
  if (kmValue < 1) return `${Math.round(kmValue * 1000)} 米`;
  return `${kmValue.toFixed(1)} 公里`;
}
function here() { return state.gps || DEPOT; }
function frameAll() {
  Object.values(state.maps).forEach((map) => { if (map) map.fitPoints(allSites()); });
}
function chooseSite(id) {
  if (state.surface === "field") {
    const site = allSites().find((item) => item.id === id);
    if (!site) return;
    state.navId = id;
    paint();
    const map = state.maps["near-map"];
    if (map) map.focusUpper(site);
    return;
  }
  state.siteId = id;
  paint();
  const site = siteById(id);
  Object.values(state.maps).forEach((map) => { if (map) map.focusUpper(site); });
}
function bearingText(from, to) {
  const rad = Math.PI / 180;
  const y = Math.sin((to.lng - from.lng) * rad) * Math.cos(to.lat * rad);
  const x = Math.cos(from.lat * rad) * Math.sin(to.lat * rad) - Math.sin(from.lat * rad) * Math.cos(to.lat * rad) * Math.cos((to.lng - from.lng) * rad);
  const names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"];
  return names[Math.round(((Math.atan2(y, x) * 180 / Math.PI) + 360) % 360 / 45) % 8];
}
function navHref(site) {
  const origin = here();
  const fromName = state.gps ? "我的位置" : DEPOT.name;
  const from = `${origin.lng},${origin.lat},${encodeURIComponent(fromName)}`;
  const to = `${site.lng},${site.lat},${encodeURIComponent(site.name)}`;
  return `https://uri.amap.com/navigation?from=${from}&to=${to}&mode=car&policy=1&src=yanwuliaowang&coordinate=gps&callnative=1`;
}
function clockLabel(offset) {
  const minute = Math.round(offset / 60);
  if (minute === 0) return "烟刚可见";
  if (minute > 0) return `烟可见后 ${minute} 分钟`;
  return `烟可见前 ${-minute} 分钟`;
}
function frameWhen(filename) {
  const stamp = Number(String(filename || "").slice(0, 10));
  if (!stamp) return "";
  return new Date(stamp * 1000).toLocaleString("zh-CN", {
    timeZone: "America/Los_Angeles", hour12: false,
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}
function scoreText(score) {
  if (score == null || Number.isNaN(Number(score))) return "";
  const pct = Math.round(Number(score) * 100);
  return pct <= 0 ? "把握很低" : `把握大约 ${pct}%`;
}
function scorePct(score) {
  if (score == null || Number.isNaN(Number(score))) return null;
  return Math.round(Number(score) * 100);
}
function alarmIndex() {
  const offset = state.demo.modes[state.mode].alarm_offset;
  const index = state.demo.frames.findIndex((frame) => frame.offset === offset);
  return index >= 0 ? index : 0;
}
function appearIndex() {
  const index = state.demo.frames.findIndex((frame) => frame.offset === 0);
  return index >= 0 ? index : 0;
}
function viewFromIndex(index) {
  const frame = state.demo.frames[index];
  const stats = frame[state.mode];
  return {
    filename: frame.filename,
    width: state.demo.width,
    height: state.demo.height,
    shapes: stats.shapes || [],
    offset: frame.offset,
    candidates: stats.candidates,
    kept: stats.kept,
    alarm: stats.alarm,
    score: stats.score,
    age: stats.age,
    pending: Boolean(stats.pending),
    index,
  };
}
function describe(view) {
  if (view.pending) {
    if (state.watch.status === "error") return { word: "暂无异常", level: "ok", text: state.watch.message || "自动检查没有完成。", score: "" };
    return { word: "正在检查", level: "ok", text: "正在对原图做自动检查。", score: "" };
  }
  let word = "暂无异常";
  let level = "ok";
  let text = "这一张没有标出可疑区域。";
  if (view.alarm) {
    word = "建议查看";
    level = "alarm";
    text = view.age > 0
      ? `提醒还留着，这一张没有新的标记，依据的是 ${Math.round(view.age / 60)} 分钟前的画面。`
      : "这一张里还能看到那处痕迹，而且比较像烟雾。建议有人看一眼。";
  } else if (view.kept) {
    word = "发现可疑";
    level = "watch";
    text = "同一处已经连续出现，按发现可疑继续看。";
  } else if (view.candidates) {
    word = "发现可疑";
    level = "watch";
    text = "这一张有可疑的地方。只闪一次不会提醒，要看后面还在不在。";
  }
  return { word, level, text, score: view.kept || view.alarm ? scoreText(view.score) : "" };
}
function monitorWord(view) {
  if (!view || view.pending) return "暂无异常";
  if (view.alarm) return "建议查看";
  if (view.candidates || view.kept) return "发现可疑";
  return "暂无异常";
}
function monitorLevel(word) {
  if (word === "建议查看") return "alarm";
  if (word === "发现可疑") return "watch";
  return "ok";
}
function phaseMarks() {
  const frames = state.demo.frames;
  const mode = state.mode;
  const marks = [
    ["appear", frames.findIndex((frame) => frame.offset === 0), "烟刚可见"],
    ["suspect", frames.findIndex((frame) => frame.offset >= 0 && frame[mode].candidates > 0), "第一次发现可疑"],
    ["kept", frames.findIndex((frame) => frame.offset >= 0 && frame[mode].kept > 0), "开始连续出现"],
    ["alarm", frames.findIndex((frame) => frame.offset >= 0 && frame[mode].alarm), "第一次建议查看"],
  ];
  return marks.filter((item) => item[1] >= 0).map(([id, index, title]) => ({ id, index, title, offset: frames[index].offset }));
}
function clearedFrameIndexes() {
  const open = new Set();
  const done = new Set();
  loadOrders().forEach((order) => {
    const index = order.media && order.media.index;
    if (!Number.isInteger(index)) return;
    if (order.status === "done") done.add(index);
    else open.add(index);
  });
  return new Set([...done].filter((index) => !open.has(index)));
}
function siteVM(site) {
  if (site.id !== "syp-n") return { ...site, level: "ok", status: "正常" };
  if (site.live && state.demo) {
    const summary = describe(viewFromIndex(state.index));
    return { ...site, level: summary.level, status: summary.word, detail: summary.text };
  }
  return { ...site, level: site.tone || "ok", status: site.status, detail: site.detail };
}
function currentSite() { return siteVM(siteById(state.siteId)); }
function openOrders() { return loadOrders().filter((order) => order.status !== "done"); }
function countStatus(status) { return loadOrders().filter((order) => order.status === status).length; }
function countOpenFor(personId) { return loadOrders().filter((order) => order.assigneeId === personId && order.status !== "done").length; }
function countDoneFor(personId) { return loadOrders().filter((order) => order.assigneeId === personId && order.status === "done").length; }
function personScore(person) {
  const done = person.historyDone + countDoneFor(person.id);
  const total = done + countOpenFor(person.id);
  return total ? Math.round(done / total * 100) : 100;
}
function bestPerson(level) {
  const prefer = level === "alarm" ? ["烟雾复核", "现场处置"] : level === "watch" ? ["持续观察", "现场处置"] : ["台账维护", "持续观察"];
  const people = rosterPeople();
  const ranked = [...people].sort((a, b) => {
    const as = prefer.indexOf(a.skill);
    const bs = prefer.indexOf(b.skill);
    const ap = as < 0 ? 9 : as;
    const bp = bs < 0 ? 9 : bs;
    return ap - bp || countOpenFor(a.id) - countOpenFor(b.id);
  });
  return ranked[0] || null;
}
function adviceFor(summary, site) {
  if (summary.level === "alarm") return `智能助手：${site.name} 已到建议查看。请核对圈出的位置，确认是烟、云，还是尘和光斑，再回传结果。`;
  if (summary.level === "watch") return `智能助手：${site.name} 发现可疑。同一处连续出现，继续看下一分钟是否留在原处。`;
  return `智能助手：${site.name} 这一张没有需要出动的标记。可以记一笔后离开。`;
}
function levelTag(level, text) {
  return `<span class="tag ${esc(level)}">${esc(text)}</span>`;
}
function fmtDur(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  const hours = Math.floor(minutes / 60);
  if (hours) return `${hours}:${String(minutes % 60).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}
function elapsed(order) {
  if (!order) return "0:00";
  const end = order.status === "done" ? (order.finishedAt || Date.now()) : Date.now();
  return fmtDur(end - (order.startedAt || order.createdAt || Date.now()));
}
function mediaSrc(media, size) {
  if (!media) return "";
  if (media.kind === "demo") return `/api/media/demo/${encodeURIComponent(media.filename)}?size=${size || 1440}`;
  if (media.kind === "mine") return `/api/media/mine/${media.sessionId}/${encodeURIComponent(media.filename)}?size=${size || 1440}`;
  if (media.kind === "chat") return `/api/media/chat/${encodeURIComponent(media.filename)}?size=${size || 1440}`;
  return "";
}
function orderTitle(site, summary) {
  if (summary.level === "alarm") return `核对${site.name}烟雾`;
  if (summary.level === "watch") return `${site.name}持续观察`;
  return `${site.name}例行查看`;
}

function makeOrder(options) {
  const site = siteById(options.siteId || "syp-n");
  const view = Number.isInteger(options.index) ? viewFromIndex(options.index) : null;
  const summary = view ? describe(view) : { word: site.status, level: options.level || site.tone || "ok", text: site.detail || "", score: "" };
  const person = options.assigneeId ? loadPeople().find((item) => item.id === options.assigneeId) : bestPerson(summary.level);
  const now = Date.now() - (options.ago || 0);
  const status = STATUS[options.status] ? options.status : "doing";
  return {
    id: options.id || uid(),
    siteId: site.id,
    title: options.title || orderTitle(site, summary),
    level: options.level || summary.level,
    summary: summary.text,
    score: summary.score,
    advice: adviceFor(summary, site),
    status,
    assigneeId: person ? person.id : "",
    action: options.action || "",
    note: options.note || "",
    support: false,
    createdAt: now,
    startedAt: now,
    finishedAt: status === "done" ? now + 1000 : null,
    urgedAt: 0,
    afterShot: false,
    media: view ? { kind: "demo", filename: view.filename, index: view.index } : null,
    events: [{ at: now, who: "中控", text: "生成工单" }],
    messages: [],
  };
}
function createOrder(options) {
  const orders = loadOrders();
  const siteId = options.siteId || state.siteId;
  const existing = orders.find((order) => order.status !== "done" && order.siteId === siteId && (!Number.isInteger(options.index) || (order.media && order.media.index === options.index)));
  if (existing) return existing;
  const order = makeOrder({ ...options, siteId });
  orders.unshift(order);
  saveOrders(orders);
  return order;
}
function clearPendingPhoto() {
  if (state.pendingPhoto && state.pendingPhoto.url) URL.revokeObjectURL(state.pendingPhoto.url);
  state.pendingPhoto = null;
}
function savedActions(order) {
  if (!order) return [];
  const listed = Array.isArray(order.actions) ? order.actions : String(order.action || "").split(",");
  return listed.filter((id) => ACTIONS[id]);
}
function shownActions(order) {
  if (!order) return [];
  if (state.draftOrderId === order.id && Array.isArray(state.draftActions)) return state.draftActions.filter((id) => ACTIONS[id]);
  return savedActions(order);
}
async function submitField(orderId) {
  const picked = state.draftOrderId === orderId ? Object.keys(ACTIONS).filter((id) => (state.draftActions || []).includes(id)) : [];
  const pending = state.pendingPhoto && state.pendingPhoto.orderId === orderId ? state.pendingPhoto : null;
  if (!picked.length && !pending) { toast("请先选择处置"); return; }
  if (pending) {
    try {
      const media = await storePhoto(pending.file);
      setOrder(orderId, { afterShot: true, after: media, event: { who: meName(), text: "上传了处理后照片", media } });
      clearPendingPhoto();
    } catch (error) {
      toast(error.message);
      return;
    }
  }
  if (picked.length) {
    picked.forEach((id, index) => {
      const patch = { actions: picked, action: picked.join(","), event: { who: meName(), text: ACTIONS[id] } };
      if (index === picked.length - 1) {
        if (picked.some((item) => item === "confirm" || item === "watch" || item === "other")) patch.status = "doing";
        if (picked.includes("cleared")) patch.status = "review";
      }
      setOrder(orderId, patch);
    });
    state.draftActions = null;
    state.draftOrderId = "";
  }
  toast(picked.length ? `已提交${picked.map((id) => ACTIONS[id]).join("、")}` : "已提交照片");
  paint();
}
function setOrder(id, patch) {
  const orders = loadOrders();
  const order = orders.find((item) => item.id === id);
  if (!order) return null;
  const event = patch.event;
  delete patch.event;
  if (patch.status === "doing" && !order.startedAt) order.startedAt = Date.now();
  if (patch.status === "done") order.finishedAt = Date.now();
  Object.assign(order, patch);
  if (event) {
    const entry = { at: Date.now(), who: event.who, text: event.text };
    if (event.media) entry.media = event.media;
    order.events.push(entry);
  }
  saveOrders(orders);
  return order;
}
function deleteOrder(id) {
  const orders = loadOrders().filter((item) => item.id !== id);
  saveOrders(orders);
  if (state.orderId === id) state.orderId = (orders.find((item) => item.status !== "done") || orders[0] || {}).id || "";
  if (state.draftOrderId === id) {
    state.draftOrderId = "";
    state.draftActions = null;
  }
  if (state.pendingPhoto && state.pendingPhoto.orderId === id) clearPendingPhoto();
  if (state.shot && state.shot.orderId === id) closeShot();
}
function addMessage(id, who, text, media) {
  const orders = loadOrders();
  const order = orders.find((item) => item.id === id);
  const body = String(text || "").trim();
  if (!order || (!body && !media)) return;
  const entry = { at: Date.now(), who, text: body };
  if (media && media.filename) entry.media = { kind: media.kind || "chat", filename: media.filename };
  order.messages.push(entry);
  saveOrders(orders);
}
function messageBubble(item, mine) {
  const shot = item.media ? `<img alt="" src="${esc(mediaSrc(item.media, 1440))}">` : "";
  const text = item.text ? `<span>${esc(item.text)}</span>` : "";
  return `<div class="msg${mine ? " mine" : ""}"><b>${esc(item.who)}</b>${shot}${text}</div>`;
}
function currentOrder() {
  const orders = loadOrders();
  return orders.find((order) => order.id === state.orderId) || orders.find((order) => order.status !== "done") || orders[0] || null;
}

function latLngToWorld(lat, lng, zoom) {
  const size = 256 * 2 ** zoom;
  const rad = lat * Math.PI / 180;
  return {
    x: (lng + 180) / 360 * size,
    y: (1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2 * size,
  };
}
function worldToLatLng(x, y, zoom) {
  const size = 256 * 2 ** zoom;
  const lng = x / size * 360 - 180;
  const lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * y / size))) * 180 / Math.PI;
  return { lat, lng };
}

class CorridorMap {
  constructor(host) {
    this.host = host;
    this.zoom = 11;
    this.lat = 34.55;
    this.lng = -119.86;
    this.drag = null;
    this.tiles = new Map();
    this.failed = new Set();
    this.map = null;
    this.canvas = null;
    this.tileLayer = null;
    if (window.AMap) {
      try { this.mountAmap(); }
      catch { this.mountLocal(); }
    } else {
      this.mountLocal();
    }
    host.addEventListener("click", (event) => {
      const button = event.target.closest("[data-zoom]");
      if (button) {
        if (button.dataset.zoom === "fit") frameAll();
        else this.zoomBy(Number(button.dataset.zoom));
        return;
      }
      const pin = event.target.closest(".hit");
      if (!pin) return;
      chooseSite(pin.dataset.site);
    });
    this.observer = new ResizeObserver(() => {
      if (this.map) this.map.resize();
      this.draw();
    });
    this.observer.observe(host);
  }
  shell(credit) {
    this.host.innerHTML = `<div class="amap-view"></div><canvas></canvas><div class="tiles"></div><svg></svg><div class="map-tools"><button type="button" data-zoom="1" aria-label="放大">+</button><button type="button" data-zoom="-1" aria-label="缩小">−</button><button type="button" data-zoom="fit" class="wide">全程</button></div><p class="map-credit">${credit}</p>`;
    this.canvas = this.host.querySelector("canvas");
    this.tileLayer = this.host.querySelector(".tiles");
    this.svg = this.host.querySelector("svg");
  }
  mountAmap() {
    this.shell("© 高德地图");
    this.host.querySelector("canvas").hidden = true;
    this.host.querySelector(".tiles").hidden = true;
    this.map = new window.AMap.Map(this.host.querySelector(".amap-view"), {
      zoom: this.zoom,
      center: [this.lng, this.lat],
      viewMode: "2D",
      zooms: [3, 18],
      mapStyle: "amap://styles/normal",
    });
    this.map.on("mapmove", () => this.drawPins());
    this.map.on("zoomend", () => { this.zoom = this.map.getZoom(); this.drawPins(); });
    this.map.on("moveend", () => this.drawPins());
    this.map.on("complete", () => {
      this.drawPins();
      if (!this.didFit) {
        this.didFit = true;
        this.fitPoints(allSites());
      }
    });
  }
  mountLocal() {
    this.map = null;
    this.shell("示意海岸，街道图 © OpenStreetMap");
    this.host.querySelector(".amap-view").hidden = true;
    const host = this.host;
    host.addEventListener("wheel", (event) => { event.preventDefault(); this.zoomBy(event.deltaY > 0 ? -1 : 1); }, { passive: false });
    host.addEventListener("pointerdown", (event) => {
      if (event.target.closest("button, .hit")) return;
      this.drag = { x: event.clientX, y: event.clientY, lat: this.lat, lng: this.lng, moved: false };
      host.setPointerCapture(event.pointerId);
    });
    host.addEventListener("pointermove", (event) => {
      if (!this.drag) return;
      const dx = event.clientX - this.drag.x;
      const dy = event.clientY - this.drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) this.drag.moved = true;
      const start = latLngToWorld(this.drag.lat, this.drag.lng, this.zoom);
      const next = worldToLatLng(start.x - dx, start.y - dy, this.zoom);
      this.lat = next.lat;
      this.lng = next.lng;
      this.draw();
    });
    host.addEventListener("pointerup", () => { this.drag = null; });
  }
  zoomBy(delta) {
    if (this.map) {
      this.map.setZoom(Math.max(3, Math.min(18, this.map.getZoom() + delta)));
      return;
    }
    this.zoom = Math.max(3, Math.min(15, this.zoom + delta));
    this.draw();
  }
  focusUpper(point) {
    const width = this.host.clientWidth;
    const height = this.host.clientHeight;
    if (width < 20 || height < 20) return;
    if (this.map) {
      this.map.setZoomAndCenter(this.map.getZoom(), [point.lng, point.lat], true);
      const shifted = this.map.containerToLngLat(new window.AMap.Pixel(width / 2, height * 0.75));
      const lng = typeof shifted.getLng === "function" ? shifted.getLng() : shifted.lng;
      const lat = typeof shifted.getLat === "function" ? shifted.getLat() : shifted.lat;
      this.map.setCenter([lng, lat], true);
      this.drawPins();
      return;
    }
    const world = latLngToWorld(point.lat, point.lng, this.zoom);
    const center = worldToLatLng(world.x, world.y + height / 4, this.zoom);
    this.lat = center.lat;
    this.lng = center.lng;
    this.draw();
  }
  centerOn(point, zoom) {
    if (this.map) {
      this.map.setZoomAndCenter(zoom, [point.lng, point.lat], false, 350);
      return;
    }
    this.zoom = zoom;
    this.lat = point.lat;
    this.lng = point.lng;
    this.draw();
  }
  fitPoints(points) {
    const clean = points.filter((point) => point && Number.isFinite(point.lat) && Number.isFinite(point.lng));
    if (!clean.length) return;
    const lats = clean.map((point) => point.lat);
    const lngs = clean.map((point) => point.lng);
    const south = Math.min(...lats);
    const north = Math.max(...lats);
    const west = Math.min(...lngs);
    const east = Math.max(...lngs);
    const mid = { lat: (south + north) / 2, lng: (west + east) / 2 };
    if (km({ lat: south, lng: west }, { lat: north, lng: east }) < 2) {
      this.centerOn(mid, this.host.id === "near-map" ? 12 : 13);
      return;
    }
    if (this.map) {
      const bounds = new window.AMap.Bounds([west, south], [east, north]);
      const avoid = this.host.id === "near-map" ? [28, 28, 28, 28] : [36, 320, 340, 36];
      this.map.setBounds(bounds, false, avoid);
      return;
    }
    const width = Math.max(80, this.host.clientWidth - (this.host.id === "near-map" ? 56 : 360));
    const height = Math.max(80, this.host.clientHeight - (this.host.id === "near-map" ? 56 : 280));
    let zoom = 15;
    for (; zoom > 3; zoom -= 1) {
      const a = latLngToWorld(north, west, zoom);
      const b = latLngToWorld(south, east, zoom);
      if (Math.abs(b.x - a.x) <= width && Math.abs(b.y - a.y) <= height) break;
    }
    this.zoom = zoom;
    this.lat = mid.lat;
    this.lng = mid.lng;
    this.draw();
  }
  origin() {
    const center = latLngToWorld(this.lat, this.lng, this.zoom);
    return { x: center.x - this.host.clientWidth / 2, y: center.y - this.host.clientHeight / 2 };
  }
  toScreen(lat, lng) {
    if (this.map) {
      const point = this.map.lngLatToContainer([lng, lat]);
      return { x: point.x, y: point.y };
    }
    const world = latLngToWorld(lat, lng, this.zoom);
    const origin = this.origin();
    return { x: world.x - origin.x, y: world.y - origin.y };
  }
  draw() {
    const width = this.host.clientWidth;
    const height = this.host.clientHeight;
    if (!width || !height) return;
    if (this.map) { this.drawPins(); return; }
    this.drawBase(width, height);
    this.drawTiles(width, height);
    this.drawPins();
  }
  drawBase(width, height) {
    const ctx = this.canvas.getContext("2d");
    this.canvas.width = width;
    this.canvas.height = height;
    const wash = ctx.createLinearGradient(0, 0, 0, height);
    wash.addColorStop(0, "#d7ebe3");
    wash.addColorStop(1, "#c5d7e6");
    ctx.fillStyle = wash;
    ctx.fillRect(0, 0, width, height);
    ctx.beginPath();
    const north = this.toScreen(35.2, -120.45);
    ctx.moveTo(north.x, north.y);
    ctx.lineTo(this.toScreen(35.2, -119.05).x, this.toScreen(35.2, -119.05).y);
    for (let lng = -119.05; lng >= -120.45; lng -= 0.05) {
      const lat = 34.4 + (lng + 119.8) * 0.11 + Math.sin((lng + 120) * 5) * 0.02;
      const point = this.toScreen(lat, lng);
      ctx.lineTo(point.x, point.y);
    }
    ctx.closePath();
    ctx.fillStyle = "#d5eadf";
    ctx.fill();
    ctx.fillStyle = "#56756c";
    ctx.font = "13px sans-serif";
    const ocean = this.toScreen(34.28, -120.02);
    const town = this.toScreen(34.44, -119.75);
    ctx.fillText("太平洋", ocean.x, ocean.y);
    ctx.fillText("圣巴巴拉", town.x, town.y);
  }
  drawTiles(width, height) {
    const origin = this.origin();
    const zoom = this.zoom;
    const max = 2 ** zoom;
    const x0 = Math.max(0, Math.floor(origin.x / 256));
    const y0 = Math.max(0, Math.floor(origin.y / 256));
    const x1 = Math.min(max - 1, Math.floor((origin.x + width) / 256));
    const y1 = Math.min(max - 1, Math.floor((origin.y + height) / 256));
    const need = new Set();
    for (let x = x0; x <= x1; x += 1) {
      for (let y = y0; y <= y1; y += 1) {
        const key = `${zoom}/${x}/${y}`;
        need.add(key);
        if (this.failed.has(key)) continue;
        let img = this.tiles.get(key);
        if (!img) {
          img = new Image();
          img.alt = "";
          img.referrerPolicy = "no-referrer";
          img.src = `https://tile.openstreetmap.org/${key}.png`;
          img.addEventListener("error", () => { this.failed.add(key); img.remove(); this.tiles.delete(key); });
          this.tiles.set(key, img);
          this.tileLayer.append(img);
        }
        img.style.left = `${x * 256 - origin.x}px`;
        img.style.top = `${y * 256 - origin.y}px`;
      }
    }
    for (const [key, img] of this.tiles) {
      if (!need.has(key)) { img.remove(); this.tiles.delete(key); }
    }
  }
  drawPins() {
    const svg = this.svg;
    svg.replaceChildren();
    allSites().forEach((site) => {
      const point = this.toScreen(site.lat, site.lng);
      const view = siteVM(site);
      const color = view.level === "alarm" ? "#e11d48" : view.level === "watch" ? "#f59e0b" : "#0e9f6e";
      const hit = document.createElementNS(SVG_NS, "circle");
      hit.setAttribute("class", "hit");
      hit.dataset.site = site.id;
      hit.setAttribute("cx", point.x); hit.setAttribute("cy", point.y); hit.setAttribute("r", "16");
      hit.setAttribute("fill", "transparent");
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", point.x); dot.setAttribute("cy", point.y);
      dot.setAttribute("r", site.id === state.siteId ? "8" : "6");
      dot.setAttribute("fill", color);
      dot.setAttribute("stroke", "white");
      dot.setAttribute("stroke-width", "2");
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", point.x + 12);
      label.setAttribute("y", point.y + (site.id === "syp-s" ? 16 : 4));
      label.setAttribute("fill", "#143028");
      label.setAttribute("stroke", "white");
      label.setAttribute("stroke-width", "3");
      label.setAttribute("paint-order", "stroke");
      label.setAttribute("font-size", "12");
      label.textContent = site.code;
      svg.append(hit, dot, label);
    });
  }
}

function ensureAmap() {
  if (!state.amapKey || window.AMap) return Promise.resolve();
  if (state.amapSecurity) window._AMapSecurityConfig = { securityJsCode: state.amapSecurity };
  return new Promise((resolve) => {
    const script = document.createElement("script");
    script.src = `https://webapi.amap.com/maps?v=2.0&key=${encodeURIComponent(state.amapKey)}`;
    script.onload = () => resolve();
    script.onerror = () => { state.amapKey = ""; resolve(); };
    document.head.appendChild(script);
  });
}
function mountMap(id) {
  const host = $(id);
  if (!host || host.dataset.ready) return;
  host.dataset.ready = "1";
  state.maps[id] = new CorridorMap(host);
  requestAnimationFrame(() => state.maps[id].draw());
}
function paintStage(img, svg, chip, view, summary) {
  if (!img || !view) return;
  svg.setAttribute("viewBox", `0 0 ${view.width} ${view.height}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  svg.replaceChildren();
  view.shapes.forEach((shape) => {
    const kind = shape.kind === "gap" ? "track" : shape.kind;
    if (state.layers[kind] === false) return;
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", shape.x);
    rect.setAttribute("y", shape.y);
    rect.setAttribute("width", shape.w);
    rect.setAttribute("height", shape.h);
    rect.setAttribute("class", shape.kind);
    svg.append(rect);
  });
  chip.textContent = summary.word;
  chip.dataset.level = summary.level;
  const url = `/api/media/demo/${encodeURIComponent(view.filename)}?size=1440`;
  if (img.dataset.url !== url) {
    img.dataset.url = url;
    img.src = url;
  }
  img.alt = summary.word;
}
function fill(id, signature, html) {
  const node = $(id);
  if (!node || node.dataset.sig === signature) return;
  node.dataset.sig = signature;
  node.innerHTML = html;
}
function setText(id, text) {
  const node = $(id);
  if (node && node.textContent !== String(text)) node.textContent = text;
}

function paintChrome() {
  const orders = loadOrders();
  setText("count-sites", allSites().length);
  setText("count-live", allSites().filter((site) => site.live).length);
  setText("count-open", orders.filter((order) => order.status !== "done").length);
  const ready = state.ready && state.ready.model;
  const pill = $("model-pill");
  pill.textContent = ready ? "模型已就绪" : "实时可看";
  pill.classList.toggle("warn", !ready);
  const button = $("field-switch");
  if (button) button.textContent = "现场工作台";
  document.querySelectorAll("[data-desk]").forEach((node) => {
    if (state.surface === "desk" && node.dataset.desk === state.desk) node.setAttribute("aria-current", "page");
    else node.removeAttribute("aria-current");
  });
  document.querySelectorAll("[data-field-tab]").forEach((node) => {
    if (state.surface === "field" && node.dataset.fieldTab === state.field) node.setAttribute("aria-current", "page");
    else node.removeAttribute("aria-current");
  });
  document.querySelectorAll("[data-mode]").forEach((node) => {
    node.setAttribute("aria-pressed", String(node.dataset.mode === state.mode));
  });
  document.querySelectorAll("[data-layer]").forEach((node) => {
    node.setAttribute("aria-pressed", String(state.layers[node.dataset.layer] !== false));
  });
  document.querySelectorAll("[data-filter]").forEach((node) => {
    node.setAttribute("aria-pressed", String(node.dataset.filter === state.orderFilter));
  });
  document.querySelectorAll("[data-asset-tab]").forEach((node) => {
    node.setAttribute("aria-pressed", String(node.dataset.assetTab === state.assetTab));
  });
  setText("watch-play", state.playing ? "暂停" : "继续");
  setText("speed", SPEED_LABELS[state.speed]);
}
function paintNow() {
  setText("now", new Date().toLocaleTimeString("zh-CN", { hour12: false }));
}
function paintStages() {
  if (!state.demo) return;
  const view = viewFromIndex(state.index);
  const summary = describe(view);
  const live = currentSite().live;
  paintStage($("pip-img"), $("pip-svg"), $("pip-chip"), view, summary);
  const empty = $("board-empty");
  if (live) {
    empty.hidden = true;
    $("board-img").hidden = false;
    paintStage($("board-img"), $("board-svg"), $("board-chip"), view, summary);
    if (!$("lightbox").hidden) paintStage($("full-img"), $("full-svg"), $("full-chip"), view, summary);
  } else {
    $("board-img").hidden = true;
    $("board-svg").replaceChildren();
    empty.hidden = false;
    empty.innerHTML = `<div><strong>${esc(currentSite().name)}</strong></div>`;
    $("board-chip").textContent = currentSite().status;
    $("board-chip").dataset.level = currentSite().level;
  }
  $("pos").hidden = !live;
  const scrub = $("scrub");
  if (scrub && document.activeElement !== scrub) {
    scrub.max = String(state.demo.frames.length - 1);
    scrub.value = String(state.index);
  }
  setText("pos", `${state.index + 1} / ${state.demo.frames.length}`);
  setText("full-title", `${currentSite().name} · ${clockLabel(view.offset)}`);
}
function paintFocus() {
  const site = currentSite();
  const view = viewFromIndex(state.index);
  const summary = describe(view);
  setText("float-title", `${site.code} ${site.name}`);
  setText("float-meta", `${site.kind || "瞭望点"} · 海拔 ${site.elev} 米 · ${site.lat.toFixed(2)}, ${site.lng.toFixed(2)}`);
  const status = $("float-status");
  status.textContent = site.status;
  status.className = `tag ${site.level}`;
  $("float-text").hidden = !site.live;
  $("timeline").hidden = !site.live;
  if (site.live) {
    setText("float-text", site.detail);
    const ticks = state.demo.frames.map((frame, index) => {
      const level = frame[state.mode].alarm ? "alarm" : (frame[state.mode].kept || frame[state.mode].candidates) ? "watch" : "ok";
      return `<button class="tick ${level}${index === state.index ? " on" : ""}" type="button" data-act="seek" data-index="${index}" title="${esc(clockLabel(frame.offset))}"></button>`;
    }).join("");
    fill("timeline", `${state.mode}:${state.index}`, ticks);
  } else {
    setText("float-text", "");
    fill("timeline", "off", "");
  }
  const alarmCount = state.demo.frames.filter((frame) => frame[state.mode].alarm).length;
  setText("kpi-alarm", alarmCount);
  setText("kpi-score", summary.score ? `${scorePct(view.score)}%` : "—");
  setText("kpi-open", openOrders().length);
  const cleared = clearedFrameIndexes();
  const marks = phaseMarks().filter((mark) => !cleared.has(mark.index));
  const liveName = siteById("syp-n").name;
  const alertHtml = marks.length ? marks.map((mark) => `
    <button type="button" data-act="seek" data-index="${mark.index}" class="${state.sumKey === `seek:${mark.index}` ? "on" : ""}">
      <strong>${esc(mark.title)}</strong><span>${esc(clockLabel(mark.offset))} · ${esc(liveName)}</span>
    </button>`).join("") : `<p class="empty">暂无预警</p>`;
  fill("sum-alerts", `${state.sumKey}|${marks.map((mark) => mark.index).join(",")}|${[...cleared].join(",")}`, alertHtml);
  setText("sum-count", `${marks.length} 个节点`);
  const air = currentAir();
  fill("weather-grid", `${air.temp}|${air.humidity}|${air.wind}|${air.dir}`, ["气温", "湿度", "风速", "风向"].map((label, index) => {
    const value = [air.temp, air.humidity, air.wind, air.dir][index];
    return `<div><b>${esc(value)}</b><span>${label}</span></div>`;
  }).join(""));
  setText("weather-tag", state.weather ? "实时天气" : "回放参考");
  setText("compare-line", state.weather
    ? `数据来自 ${state.placeName || "圣伊内斯峰"} · ${state.weather.source}`
    : "数据来自回放参考 · 2017-07-08 下午 · 圣巴巴拉");
  const orders = loadOrders();
  const top = orders.find((order) => order.level === "alarm" && order.status !== "done") || orders.find((order) => order.status !== "done") || orders[0];
  const away = top ? distText(km(here(), siteById(top.siteId))) : "";
  fill("sum-order", `${top ? top.id + top.status : "none"}:${away}`, top ? `
    <div class="sum-copy">
      <strong>${esc(top.title)}</strong>
      <p class="copy">${esc(siteById(top.siteId).name)} · ${esc(STATUS[top.status])}</p>
      <p class="muted">距${esc(state.gps ? "当前定位" : DEPOT.name)} ${esc(away)}</p>
    </div>
    <div class="row">
      <button class="btn primary" type="button" data-act="quick-dispatch">快速派单</button>
      <button class="btn" type="button" data-act="goto-order" data-id="${top.id}">打开工单</button>
    </div>` : `<p>还没有工单。</p><div class="row"><button class="btn primary" type="button" data-act="quick-dispatch">快速派单</button></div>`);
  setText("sum-count-orders", `${orders.length} 张工单`);
  if (state.maps.map) state.maps.map.drawPins();
}
function paintAlert() {
  const site = currentSite();
  const view = viewFromIndex(state.index);
  const summary = describe(view);
  const word = site.live ? monitorWord(view) : "暂无异常";
  const level = monitorLevel(word);
  fill("rail", `${word}:${allSites().map((item) => item.id).join(",")}:${state.siteId}`, allSites().map((item) => {
    const itemWord = item.live ? monitorWord(viewFromIndex(state.index)) : "暂无异常";
    const itemLevel = monitorLevel(itemWord);
    return `<button type="button" data-act="pick-site" data-id="${item.id}" class="${item.id === state.siteId ? "on" : ""}"><strong>${esc(item.code)} ${esc(item.name)}</strong><span class="rail-state ${itemLevel}">${esc(itemWord)}</span></button>`;
  }).join(""));
  const lock = $("cam-lock");
  if (lock) {
    lock.textContent = word;
    lock.dataset.level = level;
  }
  const progress = `${state.watch.done || 0} / ${state.watch.total || state.demo.frames.length}`;
  setText("cam-source", site.live ? "固定相机 · 原图" : "这个点没有连续画面");
  setText("cam-edge", site.live ? (view.pending ? (state.watch.status === "error" ? "没有开始" : "正在检查") : (summary.score || "这一张没有把握")) : "—");
  setText("cam-cloud", site.live && state.watch.status === "running" ? `正在检查原图 ${progress}` : state.watch.status === "error" ? (state.watch.message || "自动检查没有完成") : "模型已就绪");
  const transport = document.querySelector("#page-alert .transport");
  if (transport) transport.hidden = !site.live;
  ensureJudge(site, view, word);
  renderJudge();
  const strip = currentAir();
  ["env-temp", "env-hum", "env-wind", "env-dir"].forEach((id, index) => {
    setText(id, [strip.temp, strip.humidity, strip.wind, strip.dir][index]);
  });
  setText("env-note", state.weather ? `实时 · ${state.placeName || "圣伊内斯峰"} · ${state.weather.source}` : "天气暂未接通");
}
function renderJudge() {
  const node = $("judge");
  if (!node) return;
  const judge = state.judge;
  const labels = { idle: "持续监测", loading: "正在生成", ready: "已生成", error: "未完成" };
  setText("judge-state", labels[judge.phase] || "持续监测");
  let html = "";
  if (judge.phase === "loading") {
    html = `<p class="kicker">发现烟雾</p><h3>正在生成研判</h3><p>这一路已经出现烟雾迹象，研判写好后显示在这里。</p>`;
  } else if (judge.phase === "ready") {
    const steps = (judge.actions || []).map((item) => `<li>${esc(item)}</li>`).join("");
    html = `<p class="kicker">${esc(judge.word)}</p><h3>${esc(judge.headline)}</h3><p class="kicker">分析</p>${(judge.paragraphs || []).map((item) => `<p>${esc(item)}</p>`).join("")}${steps ? `<p class="kicker">处置方案</p><ol>${steps}</ol>` : ""}`;
  } else if (judge.phase === "error") {
    html = `<p class="kicker">研判未完成</p><h3>${esc(judge.word || "发现烟雾")}</h3><p>${esc(judge.message)}</p><button class="btn" type="button" data-act="judge-retry">重新生成</button>`;
  } else if (state.watch.status === "error") {
    html = `<p class="kicker">自动检查</p><h3>没有完成</h3><p>${esc(state.watch.message || "自动检查没有完成。")}</p>`;
  } else if (currentSite().live && state.demo && viewFromIndex(state.index).pending) {
    html = `<p class="kicker">自动检查</p><h3>正在检查原图</h3><p>模型正在看这一张原图。发现烟雾后再生成研判。</p>`;
  } else {
    const name = currentSite().name;
    html = `<p class="kicker">持续监测</p><h3>暂无异常</h3><p>${esc(name)}这一路正在看。画面里没有需要研判的烟雾。</p>`;
  }
  const signature = `${judge.sig}|${judge.phase}|${judge.headline}|${(judge.actions || []).join("|")}|${judge.message}`;
  if (node.dataset.sig === signature) return;
  node.dataset.sig = signature;
  node.innerHTML = html;
}
function ensureJudge(site, view, word) {
  if (!site.live || word === "暂无异常") {
    const sig = `${site.id}:ok`;
    if (state.judge.sig !== sig || state.judge.phase !== "idle") {
      state.judge = { sig, phase: "idle", word: "暂无异常", headline: "", paragraphs: [], actions: [], message: "" };
    }
    return;
  }
  const sig = `${site.id}:${word}`;
  if (state.judge.sig === sig && state.judge.phase !== "idle" && !state.judgeRefresh) return;
  const refresh = Boolean(state.judgeRefresh);
  state.judgeRefresh = false;
  state.judge = { sig, phase: "loading", word, headline: "", paragraphs: [], actions: [], message: "" };
  const air = state.weather;
  const payload = {
    site: site.name,
    code: site.code,
    elev: site.elev,
    status: word,
    refresh,
    candidates: view.candidates,
    kept: view.kept,
    watched: clockLabel(view.offset),
    temp: air ? air.temp : "",
    humidity: air ? air.humidity : "",
    wind: air ? air.wind : "",
    dir: air ? air.dir : "",
  };
  fetch("/api/judge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then(async (response) => {
    const data = await response.json().catch(() => ({}));
    if (state.judge.sig !== sig) return;
    if (!response.ok) {
      state.judge = { sig, phase: "error", word, headline: "", paragraphs: [], message: data.message || "研判没有生成。" };
    } else {
      state.judge = {
        sig,
        phase: "ready",
        word,
        headline: data.headline || word,
        paragraphs: Array.isArray(data.paragraphs) ? data.paragraphs : [],
        actions: Array.isArray(data.actions) ? data.actions : [],
        message: "",
      };
    }
    renderJudge();
  }).catch(() => {
    if (state.judge.sig !== sig) return;
    state.judge = { sig, phase: "error", word, headline: "", paragraphs: [], message: "研判没有生成，请再试一次。" };
    renderJudge();
  });
}
function orderRows(orders) {
  return `<table class="dispatch-table"><colgroup><col class="c-level"><col class="c-title"><col class="c-site"><col class="c-person"><col class="c-status"><col class="c-time"></colgroup><thead><tr><th>等级</th><th>工单</th><th>点位</th><th>处理人</th><th>状态</th><th>用时</th></tr></thead><tbody>${
    orders.map((order) => `<tr data-act="goto-order" data-id="${order.id}" class="${order.id === state.orderId ? "on" : ""}"><td>${levelTag(order.level, order.level === "alarm" ? "紧急" : order.level === "watch" ? "关注" : "一般")}</td><td>${esc(order.title)}</td><td>${esc(siteById(order.siteId).code)}</td><td>${esc(personName(order.assigneeId))}</td><td>${esc(STATUS[order.status])}</td><td class="time" data-timer="${order.id}">${elapsed(order)}</td></tr>`).join("")
  }</tbody></table>`;
}
function recordsHtml(order) {
  const events = (order.events || []).filter((item) => !String(item.text || "").startsWith("留言："));
  const listed = events.some((item) => item.media && order.after && item.media.filename === order.after.filename);
  const rows = events.map((item) => {
    const shot = item.media ? `<img class="record-shot" alt="现场照片" src="${esc(mediaSrc(item.media, 720))}">` : "";
    return `<div class="audit">${esc(item.who)} · ${esc(item.text)}${shot}</div>`;
  }).join("");
  const tail = order.after && !listed ? `<div class="audit">现场 · 处理后照片<img class="record-shot" alt="现场照片" src="${esc(mediaSrc(order.after, 720))}"></div>` : "";
  return rows + tail;
}
function detailHtml(order) {
  if (!order) return "<p>还没有工单。</p>";
  const site = siteById(order.siteId);
  const people = rosterPeople();
  return `
    <p class="kicker">${esc(site.code)} · ${esc(order.id)}</p>
    <h2>${esc(order.title)}</h2>
    <p>${levelTag(order.level, order.level === "alarm" ? "紧急" : order.level === "watch" ? "关注" : "一般")} ${levelTag("muted", STATUS[order.status])}</p>
    <p class="copy">${esc(order.summary)}</p>
    ${order.media ? `<img alt="" src="${esc(mediaSrc(order.media, 1440))}" style="border-radius:12px;max-height:180px;object-fit:cover">` : ""}
    <div class="advice">${esc(order.advice)}</div>
    <p class="muted">${esc(site.region)} · ${site.lat.toFixed(2)}, ${site.lng.toFixed(2)} · 距${state.gps ? "当前定位" : DEPOT.name} ${esc(distText(km(here(), site)))}</p>
    <label>处理人<select data-act="assign" data-id="${order.id}"><option value="" ${order.assigneeId ? "" : "selected"}>未指定</option>${people.map((person) => `<option value="${person.id}" ${person.id === order.assigneeId ? "selected" : ""}>${esc(person.name)} · ${esc(person.phone)}</option>`).join("")}</select></label>
    <div class="actions">
      ${Object.entries(STATUS).map(([id, label]) => `<button class="btn" type="button" data-act="status" data-id="${order.id}" data-status="${id}">${label}</button>`).join("")}
    </div>
    <div class="actions">
      <button class="btn" type="button" data-act="urge" data-id="${order.id}">催办</button>
      <button class="btn ${order.support ? "good" : ""}" type="button" data-act="support" data-id="${order.id}">${order.support ? "已请求支援" : "请求支援"}</button>
      <button class="btn good" type="button" data-act="review" data-id="${order.id}" data-review="pass">通过复核</button>
      <button class="btn danger" type="button" data-act="review" data-id="${order.id}" data-review="return">删除</button>
    </div>
    <p class="muted">用时 <span data-timer="${order.id}">${elapsed(order)}</span></p>
    <h3>记录</h3>
    <div class="records">${recordsHtml(order)}</div>`;
}
function paintDispatch() {
  if (state.orderFilter !== "all" && !STATUS[state.orderFilter]) state.orderFilter = "all";
  const orders = loadOrders();
  const filtered = state.orderFilter === "all" ? orders : orders.filter((order) => order.status === state.orderFilter);
  fill("d-counts", orders.map((order) => order.status).join(""), Object.entries(STATUS).map(([id, label]) => `<button type="button" data-filter="${id}">${label} ${countStatus(id)}</button>`).join(""));
  document.querySelectorAll("#d-counts [data-filter], #d-filters [data-filter]").forEach((node) => {
    node.setAttribute("aria-pressed", String(node.dataset.filter === state.orderFilter));
  });
  fill("d-rows", `${state.orderFilter}:${orders.map((order) => order.id + order.status + order.assigneeId).join(",")}:${state.orderId}`, filtered.length ? orderRows(filtered) : "<p>这一栏还没有工单。</p>");
  const order = currentOrder();
  if (order) state.orderId = order.id;
  fill("d-detail", order ? `${order.id}:${order.status}:${order.action}:${order.support}:${order.messages.length}:${order.events.length}:${order.assigneeId}:${photoKey(order)}:${rosterPeople().map((person) => person.id).join(",")}` : "empty", detailHtml(order));
}
function paintAssets() {
  $("asset-view").hidden = state.assetTab !== "sites";
  $("people-view").hidden = state.assetTab !== "people";
  $("a-detail").hidden = state.assetTab !== "sites";
  $("p-detail").hidden = state.assetTab !== "people";
  const sites = allSites().map(siteVM);
  const counts = {
    all: sites.length,
    alarm: sites.filter((site) => site.level === "alarm").length,
    watch: sites.filter((site) => site.level === "watch").length,
    ok: sites.filter((site) => site.level === "ok").length,
  };
  fill("a-chips", JSON.stringify(counts), [
    ["点位", counts.all], ["建议查看", counts.alarm], ["观察", counts.watch], ["正常", counts.ok],
  ].map(([label, value]) => `<div class="chip-stat"><b>${value}</b><span>${label}</span></div>`).join(""));
  const query = state.assetQuery.trim();
  const shown = sites.filter((site) => `${site.code}${site.name}${site.region}${site.line}`.includes(query));
  const siteForm = document.querySelector("form[data-act='add-site']");
  const cameraForm = document.querySelector("form[data-act='add-camera']");
  if (siteForm) siteForm.hidden = state.addForm !== "site";
  if (cameraForm) cameraForm.hidden = state.addForm !== "camera";
  document.querySelectorAll("[data-act='show-add']").forEach((node) => node.setAttribute("aria-pressed", String(state.addForm === node.dataset.add)));
  fill("camera-sites", allSites().map((site) => site.id).join(","), allSites().map((site) => `<option value="${esc(site.id)}">${esc(site.code)} ${esc(site.name)}</option>`).join(""));
  fill("a-rows", `${query}:${shown.map((site) => site.id + site.level).join(",")}:${loadCameras().map((item) => item.id).join(",")}:${state.siteId}`, `<table><thead><tr><th>编号</th><th>点位</th><th>走廊</th><th>区域</th><th>状态</th><th>海拔</th><th>摄像头</th></tr></thead><tbody>${
    shown.map((site) => `<tr data-act="pick-site" data-id="${site.id}" class="${site.id === state.siteId ? "on" : ""}"><td>${esc(site.code)}</td><td>${esc(site.name)}</td><td>${esc(site.line)}</td><td>${esc(site.region)}</td><td>${levelTag(site.level, site.status)}</td><td>${site.elev} 米</td><td>${loadCameras().filter((item) => item.siteId === site.id).length}</td></tr>`).join("")
  }</tbody></table>`);
  const site = currentSite();
  const related = loadOrders().filter((order) => order.siteId === site.id);
  const cameras = loadCameras().filter((item) => item.siteId === site.id);
  fill("a-detail", `${site.id}:${site.level}:${related.length}:${cameras.map((item) => item.id + (item.ip || "")).join(",")}`, `
    <h2>${esc(site.name)}</h2>
    <p>${esc(site.code)} · ${esc(site.kind || "瞭望点")}</p>
    <p>${levelTag(site.level, site.status)}</p>
    <p class="copy">${esc(site.detail)}</p>
    <p class="muted">${site.lat.toFixed(3)}, ${site.lng.toFixed(3)} · 海拔 ${site.elev} 米</p>
    ${cameras.length ? `<p>摄像头 ${cameras.map((item) => `${esc(item.name)}${item.ip ? ` · ${esc(item.ip)}` : ""}`).join("、")}</p>` : ""}
    <p>相关工单 ${related.length} 张</p>
    <div class="row site-actions"><button class="btn primary" type="button" data-act="open-live">去研判</button><button class="btn" type="button" data-act="create-site" data-id="${site.id}">为这个点建单</button></div>`);
  const peopleQuery = state.peopleQuery.trim();
  const people = rosterPeople().filter((person) => `${person.name}${person.phone}${person.skill}${person.region}`.includes(peopleQuery))
    .sort((a, b) => personScore(b) - personScore(a));
  fill("p-rows", `${peopleQuery}:${people.map((person) => person.id + personScore(person) + countOpenFor(person.id)).join(",")}`, people.length ? `<table><thead><tr><th></th><th>姓名</th><th>电话</th><th>擅长</th><th>区域</th><th>未结</th><th>完成</th><th>匹配</th></tr></thead><tbody>${
    people.map((person, index) => `<tr data-act="pick-person" data-id="${person.id}" class="${person.id === state.personId ? "on" : ""}"><td>${index + 1}</td><td>${esc(person.name)}</td><td>${esc(person.phone)}</td><td>${esc(person.skill)}</td><td>${esc(person.region || "未填")}</td><td>${countOpenFor(person.id)}</td><td>${person.historyDone + countDoneFor(person.id)}</td><td>${personScore(person)}%</td></tr>`).join("")
  }</tbody></table>` : "<p>还没有处理人。现场工作台注册后，到运维调度输入电话号添加。</p>");
  const person = people.find((item) => item.id === state.personId) || people[0];
  if (person) {
    const bars = [0, 1, 2, 3, 4, 5, 6, 7].map((index) => 28 + ((person.historyDone * (index + 3)) % 18));
    fill("p-detail", person.id + personScore(person), `
      <h2>${esc(person.name)}</h2>
      <p>${esc(person.phone)} · ${esc(person.skill)} · ${esc(person.region || "未填")}</p>
      <p>未结 ${countOpenFor(person.id)} · 已完成 ${person.historyDone + countDoneFor(person.id)} · 匹配 ${personScore(person)}%</p>
      <div class="bars">${bars.map((height) => `<i style="height:${height}px"></i>`).join("")}</div>
      <p class="muted">近期负荷按这个人已有的完成量排开，新工单会改未结和完成。</p>
      <button class="btn danger" type="button" data-act="drop-person" data-id="${person.id}">移出名单</button>`);
  } else {
    fill("p-detail", "empty", "<h2>处理人</h2><p class=\"copy\">现场工作台注册后，到运维调度输入电话号添加。</p>");
  }
  const modelCard = $("model-card");
  if (modelCard) {
    modelCard.hidden = true;
    modelCard.innerHTML = "";
  }
}
function paintCollab() {
  const orders = loadOrders();
  const order = currentOrder();
  if (order) state.orderId = order.id;
  fill("pool", `${order ? order.id : ""}:${orders.map((item) => `${item.id}${item.status}${item.assigneeId}${item.messages.length}`).join(",")}`, `
    <div class="panel-title"><h2>任务</h2><span>${orders.length}</span></div>
    ${orders.map((item) => `<button type="button" data-act="select-order" data-id="${item.id}" class="${order && item.id === order.id ? "on" : ""}" ${order && item.id === order.id ? "aria-current=\"true\"" : ""}><strong>${esc(item.title)}</strong><span>${esc(personName(item.assigneeId))} · ${esc(STATUS[item.status])}</span></button>`).join("") || "<p>还没有任务。</p>"}`);
  const person = order ? loadPeople().find((item) => item.id === order.assigneeId) : null;
  const site = order ? siteById(order.siteId) : null;
  const shot = order && order.media ? `<img class="task-shot" alt="${esc(order.title)}" src="${esc(mediaSrc(order.media, 1440))}">` : `<p class="empty">这张任务没有画面。</p>`;
  fill("desk-task", order ? `${order.id}:${order.status}:${order.media ? order.media.filename : ""}:${order.summary}` : "empty", order ? `
    <div class="panel-title"><h2>任务资料</h2></div>
    ${shot}
    <h2>${esc(order.title)}</h2>
    <p>${levelTag(order.level, order.level === "alarm" ? "紧急" : order.level === "watch" ? "关注" : "一般")} ${levelTag("muted", STATUS[order.status])}</p>
    <p>${esc(site.code)} · ${esc(site.name)}</p>
    <p class="muted">${esc(person ? person.name : "未指定处理人")}${person && person.skill ? ` · ${esc(person.skill)}` : ""} · ${esc(site.region)} · 海拔 ${site.elev} 米</p>
    <p class="copy">${esc(order.summary || "")}</p>` : `<p class="empty">选一个任务，查看画面和信息。</p>`);
  const rows = order ? order.messages.map((item) => messageBubble(item, item.who === "中控")).join("") : "";
  const empty = order && !order.messages.length ? `<p class="muted">和${esc(person ? person.name : "处理人")}说现在要核对什么。</p>` : "";
  const chatKey = order ? order.messages.map((item) => item.who + item.text + (item.media ? item.media.filename : "")).join("|") : "";
  fill("desk-chat", order ? `${order.id}:${order.assigneeId}:${chatKey}` : "empty", order ? `
    <div>
      <h2>${esc(order.title)}</h2>
      <p class="muted">${esc(person ? person.name : "未指定处理人")}${person && person.skill ? ` · ${esc(person.skill)}` : ""} · ${esc(site.name)}</p>
    </div>
    <div class="desk-thread" id="desk-thread">${empty}${rows}</div>
    <form data-act="chat" data-id="${order.id}"><input name="text" placeholder="和这位工人打字" aria-label="和工人打字" maxlength="200"><label class="chat-file">图片<input type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png" aria-label="选择图片"></label><button class="btn primary" type="submit">发送</button></form>` : `<p class="empty">选一个任务，再和工人说话。</p>`);
  const thread = $("desk-thread");
  if (thread) thread.scrollTop = thread.scrollHeight;
}
function myOrders() {
  const id = meId();
  if (!id) return [];
  return loadOrders().filter((order) => order.assigneeId === id);
}
function fieldHome() {
  const orders = myOrders();
  const urgent = orders.find((order) => order.status !== "done" && order.level === "alarm") || orders.find((order) => order.status !== "done");
  const done = orders.filter((order) => order.status === "done").length;
  const rate = orders.length ? Math.round(done / orders.length * 100) : 0;
  const openAlarm = orders.filter((order) => order.status !== "done" && order.level === "alarm").length;
  const registered = Boolean(mePerson());
  const open = orders.filter((order) => order.status !== "done");
  return `
    <p class="kicker">烟雾瞭望现场工作台</p>
    <div class="em">
      <div class="muted">当前任务 ${urgent ? "进行中" : "空闲"}</div>
      <strong>${urgent ? esc(urgent.title) : "没有待办"}</strong>
      <p>${urgent ? `${levelTag(urgent.level, urgent.level === "alarm" ? "紧急" : "关注")} ${esc(siteById(urgent.siteId).code)}` : ""}</p>
      <p class="copy">${urgent ? esc(urgent.summary) : (registered ? "分给你的预警会出现在这里。" : "注册并成为处理人后，才能看到分给你的预警。")}</p>
    </div>
    <div class="stats">
      <div><span>我完成的</span><b>${done}</b></div>
      <div><span>我的紧急</span><b>${openAlarm}</b></div>
    </div>
    <div class="em">
      <span>我的完成率</span>
      <b class="rate">${rate}%</b>
      <div class="bar"><i style="width:${rate}%"></i></div>
    </div>
    <div class="quick">
      <button type="button" data-act="field" data-field="report"><b>一键上报</b><span>拍照、定位、描述等</span></button>
      <button type="button" data-act="field" data-field="nearby"><b>附近点位</b><span>按距离看走廊</span></button>
      <button type="button" data-act="field" data-field="talk"><b>现场助手</b><span>直接问怎么判断</span></button>
    </div>
    ${registered ? "" : `<button class="btn block" type="button" data-act="field" data-field="me">先注册，调度中心才能添加你</button>`}
    ${open.length ? `<div class="list">${open.map((order) => `<button type="button" data-act="field-order" data-id="${order.id}"><strong>${esc(order.title)}</strong><span>${esc(siteById(order.siteId).code)} · ${esc(STATUS[order.status])}</span></button>`).join("")}</div>` : ""}
    ${urgent ? `<button class="btn primary block" type="button" data-act="field-order" data-id="${urgent.id}">继续处理当前工单</button>` : ""}`;
}
function fieldOrders() {
  const orders = myOrders();
  const count = (status) => orders.filter((order) => order.status === status).length;
  return `
    <p class="kicker">工单中心</p>
    <h2>任务列表</h2>
    <div class="stats quad">
      ${["doing", "review", "done"].map((id) => `<div><span>${STATUS[id]}</span><b>${count(id)}</b></div>`).join("")}
    </div>
    <div class="list">${orders.length ? orders.map((order) => `<button type="button" data-act="field-order" data-id="${order.id}"><strong>${esc(order.title)}</strong><span>${esc(siteById(order.siteId).code)} · ${esc(STATUS[order.status])}</span></button>`).join("") : "<p>还没有分给你的预警。</p>"}</div>`;
}
function photoKey(order) {
  const names = (order.events || []).map((item) => (item.media && item.media.filename) || "");
  return `${names.join(",")}|${order.after ? order.after.filename : ""}|${order.afterShot ? 1 : 0}`;
}
function afterBlock(order) {
  const pending = state.pendingPhoto && state.pendingPhoto.orderId === order.id ? state.pendingPhoto : null;
  if (pending) {
    return `<div class="photo-slot"><div class="shot"><img alt="待提交照片" src="${esc(pending.url)}"><button class="photo-x" type="button" data-act="delete-photo" data-id="${esc(order.id)}" aria-label="删除照片">×</button></div></div>`;
  }
  if (!order.after) return "";
  return `<div class="shot"><img alt="处理后照片" src="${esc(mediaSrc(order.after, 1440))}"></div>`;
}
function fieldAdviceKey(order) {
  return `${order.id}|${order.level}|${order.summary || ""}`;
}
function fieldAdviceHtml(order) {
  const current = state.fieldAdvice;
  const ready = current && current.orderId === order.id && current.sig === fieldAdviceKey(order) && current.phase === "ready" && current.text;
  if (ready) return `<div class="advice">智能助手：${esc(current.text)}</div>`;
  if (current && current.orderId === order.id && current.phase === "error") {
    return `<div class="advice"><p>智能助手：${esc(current.text || "没有生成。")}</p><button class="btn" type="button" data-act="advice-retry" data-id="${esc(order.id)}">重新生成</button></div>`;
  }
  return `<div class="advice">智能助手：正在根据这张画面生成…</div>`;
}
function ensureFieldAdvice(order) {
  if (!order || order.assigneeId !== meId()) return;
  const sig = fieldAdviceKey(order);
  const current = state.fieldAdvice;
  if (current && current.sig === sig && current.phase !== "idle") return;
  state.fieldAdvice = { sig, orderId: order.id, phase: "loading", text: "" };
  const site = siteById(order.siteId);
  const air = state.weather;
  fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: "根据这张画面，给现场工人写现在该怎么做。两三句话，直接说核对哪里、怎么区分烟、云、尘和光斑。",
      history: [],
      media: order.media,
      site: site.name,
      code: site.code,
      elev: site.elev,
      status: order.level === "alarm" ? "建议查看" : order.level === "watch" ? "发现可疑" : "暂无异常",
      summary: order.summary || "",
      temp: air ? air.temp : "",
      humidity: air ? air.humidity : "",
      wind: air ? air.wind : "",
      dir: air ? air.dir : "",
    }),
  }).then(async (response) => {
    const data = await response.json().catch(() => ({}));
    if (!state.fieldAdvice || state.fieldAdvice.sig !== sig) return;
    if (!response.ok) {
      state.fieldAdvice = { sig, orderId: order.id, phase: "error", text: data.message || "没有生成，请再试一次。" };
    } else {
      const text = String(data.answer || "").replace(/^智能助手[：:]\s*/, "").trim();
      state.fieldAdvice = { sig, orderId: order.id, phase: text ? "ready" : "error", text: text || "没有生成，请再试一次。" };
    }
    if (state.surface === "field") paint();
  }).catch(() => {
    if (!state.fieldAdvice || state.fieldAdvice.sig !== sig) return;
    state.fieldAdvice = { sig, orderId: order.id, phase: "error", text: "没有生成，请再试一次。" };
    if (state.surface === "field") paint();
  });
}
function fieldOrder() {
  const order = currentOrder();
  if (!order || order.assigneeId !== meId()) return `<button class="back" type="button" data-act="field" data-field="orders">返回</button><p>这张工单不是分给你的。</p>`;
  const site = siteById(order.siteId);
  return `
    <button class="back" type="button" data-act="field" data-field="orders">返回工单</button>
    <p>${levelTag(order.level, order.level === "alarm" ? "紧急" : "关注")} ${esc(site.code)}</p>
    <h2>${esc(order.title)}</h2>
    ${order.media ? `<button class="shot" type="button" data-act="open-shot" data-id="${order.id}" aria-label="放大查看冒烟画面"><img alt="" src="${esc(mediaSrc(order.media, 1440))}"></button><p class="muted">点击画面可以放大，并询问这张图。</p>` : ""}
    <div class="em"><strong>位置</strong><p>${esc(site.name)} · ${site.lat.toFixed(3)}, ${site.lng.toFixed(3)}</p><p class="muted">距${state.gps ? "当前定位" : DEPOT.name} ${esc(distText(km(here(), site)))}</p></div>
    ${fieldAdviceHtml(order)}
    <div class="quick five">
      ${Object.entries(ACTIONS).map(([id, label]) => {
        const picked = shownActions(order).includes(id);
        return `<button type="button" data-act="pick-action" data-id="${order.id}" data-action="${id}" aria-pressed="${picked}"><b>${label}</b></button>`;
      }).join("")}
    </div>
    <button class="btn primary block" type="button" data-act="submit-action" data-id="${order.id}">提交</button>
    <div class="actions">
      <button class="btn" type="button" data-act="after">上传处理后照片</button>
      <button class="btn ${order.support ? "good" : ""}" type="button" data-act="support" data-id="${order.id}">${order.support ? "已请求支援" : "请求支援"}</button>
    </div>
    ${afterBlock(order)}
    <label>备注<textarea data-act="order-note" data-id="${order.id}">${esc(order.note)}</textarea></label>
    <div class="list">${order.messages.map((item) => messageBubble(item, item.who === meName())).join("")}</div>
    <form data-act="chat" data-id="${order.id}"><input name="text" placeholder="回复中控" aria-label="回复中控" maxlength="200"><label class="chat-file">图片<input type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png" aria-label="选择图片"></label><button class="btn" type="submit">发送</button></form>
    <div class="actions">
      <button class="btn primary" type="button" data-act="status" data-id="${order.id}" data-status="review">提交复核</button>
      <button class="btn" type="button" data-act="field" data-field="orders">稍后处理</button>
    </div>`;
}
function fieldReport() {
  const report = state.report;
  const edge = report.edge || [];
  return `
    <button class="back" type="button" data-act="field" data-field="home">返回工作台</button>
    <h2>一键上报</h2>
    <p class="copy">先拍照，再写看到了什么。当前画面可以直接拿来做初判；新照片可以提交给中控，有模型时再做云端复核。</p>
    <div class="shot">${report.preview ? `<img alt="待上报" src="${esc(report.preview)}">` : "<span>还没有照片</span>"}</div>
    <div class="row">
      <button class="btn primary" type="button" data-act="shoot">拍照或选图</button>
      <button class="btn" type="button" data-act="use-live">用当前画面</button>
    </div>
    ${edge.length ? `<div class="em"><strong>${report.source === "demo" ? "边缘初判 · 当前画面" : "云端复核"}</strong>${edge.map((item) => `<div class="bar-row"><span>${esc(item.name)}</span><div class="bar"><i style="width:${item.value}%"></i></div><b>${item.value}%</b></div>`).join("")}</div>` : ""}
    <p class="muted">${esc(report.message || (state.ready && state.ready.model ? "新照片可以做云端复核。" : "没有模型包时，新照片先提交到中控。"))}</p>
    <div class="em"><strong>定位</strong><p>${state.gps ? `${state.gps.lat.toFixed(5)}, ${state.gps.lng.toFixed(5)}` : `${DEPOT.lat}, ${DEPOT.lng}`}</p><p class="muted">${esc(state.placeName || DEPOT.name)}</p><button class="btn" type="button" data-act="locate">重新定位</button></div>
    <label>现场描述<textarea data-act="report-text" placeholder="例如：山脊上有一团灰白色，停在那里没有散">${esc(report.text)}</textarea></label>
    <div class="actions">
      <button class="btn" type="button" data-act="cloud">云端复核</button>
      <button class="btn primary" type="button" data-act="submit-report">提交到中控</button>
    </div>`;
}
function fieldNearby() {
  const origin = here();
  const rows = allSites().map((site) => ({ site: siteVM(site), distance: km(origin, site) })).sort((a, b) => a.distance - b.distance);
  const mineSites = new Set(myOrders().filter((order) => order.status !== "done").map((order) => order.siteId));
  return `
    <button class="back" type="button" data-act="field" data-field="home">返回工作台</button>
    <h2>附近点位</h2>
    <div id="near-map" class="map-host" style="height:180px;border-radius:16px;overflow:hidden"></div>
    <div class="list">${rows.map(({ site, distance }) => `<button type="button" data-act="pick-site" data-id="${site.id}"><strong>${esc(site.name)}</strong><span>${esc(distText(distance))}${mineSites.has(site.id) ? ` · ${esc(site.status)}` : ""}</span></button>`).join("")}</div>`;
}
function fieldTalk() {
  const talk = state.talk;
  const rows = talk.messages.map((item) => `<div class="msg${item.role === "assistant" ? "" : " mine"}"><b>${item.role === "assistant" ? "现场助手" : "我"}</b><span>${esc(item.text)}</span></div>`).join("");
  const waiting = talk.phase === "loading" ? `<p class="muted">正在想…</p>` : "";
  const error = talk.error ? `<p class="ask-error">${esc(talk.error)}</p>` : "";
  const empty = !talk.messages.length && talk.phase !== "loading" ? `<p class="muted">可以直接问。比如烟和云怎么分，或者到了现场先核对哪里。</p>` : "";
  return `
    <button class="back" type="button" data-act="field" data-field="home">返回工作台</button>
    <h2>现场助手</h2>
    <div class="talk-thread">${empty}${rows}${waiting}${error}</div>
    <div class="shot-prompts">
      <button type="button" data-act="talk-prompt" data-q="烟和云怎么分？">烟和云怎么分</button>
      <button type="button" data-act="talk-prompt" data-q="到了现场先核对哪里？">先核对哪里</button>
      <button type="button" data-act="talk-prompt" data-q="工单怎么提交？">工单怎么提交</button>
    </div>
    <form data-act="ask-talk"><input name="text" placeholder="问烟、云、工单或现场怎么做" aria-label="问现场助手" maxlength="200"><button class="btn primary" type="submit">发送</button></form>`;
}
function fieldMe() {
  const me = mePerson();
  if (!me) {
    return `
    <h2>登录注册</h2>
    <p class="copy">注册后，调度中心输入这个电话号，才会把你加为处理人。已有账号直接登录。</p>
    <form data-act="register" autocomplete="off">
      <label data-auth="register" ${state.authMode === "register" ? "" : "hidden"}>姓名<input name="name" autocomplete="name" placeholder="你的姓名" aria-label="姓名"></label>
      <label>电话<input name="phone" autocomplete="tel" inputmode="numeric" placeholder="11位手机号" aria-label="电话"></label>
      <div class="choice" data-auth="register" ${state.authMode === "register" ? "" : "hidden"}>
        <span>擅长</span>
        <div class="skill-pick" role="radiogroup" aria-label="擅长">
          ${["烟雾复核", "现场处置", "持续观察", "台账维护"].map((name) => `<button type="button" data-act="pick-skill" data-skill="${name}" aria-pressed="${state.skill === name}">${name}</button>`).join("")}
        </div>
        <input type="hidden" name="skill" value="${esc(state.skill || "烟雾复核")}" autocomplete="off">
      </div>
      <label data-auth="register" ${state.authMode === "register" ? "" : "hidden"}>区域<input name="region" autocomplete="off" placeholder="所在区域" aria-label="区域"></label>
      <label>密码<input name="password" type="password" autocomplete="current-password" placeholder="密码" aria-label="密码"></label>
      <p class="auth-note" data-auth-note></p>
      <div class="actions auth-actions">
        <button class="btn" type="submit" value="login" formnovalidate>登陆</button>
        <button class="btn" type="submit" value="register" formnovalidate>注册</button>
      </div>
    </form>`;
  }
  return `
    <h2>我的</h2>
    <p class="copy">${esc(me.name)} · ${esc(me.phone)}</p>
    <p class="muted">${esc(me.skill)}${me.region ? ` · ${esc(me.region)}` : ""}</p>
    <p class="copy">${me.added ? "调度中心已经把你加入处理人。" : "把电话号告诉调度中心。对方在运维调度输入后，你才会出现在处理人里。"}</p>
    <div class="stats"><div><span>分给我的未结</span><b>${countOpenFor(me.id)}</b></div><div><span>我完成的</span><b>${countDoneFor(me.id)}</b></div></div>
    <button class="btn block logout" type="button" data-act="logout">退出登陆</button>`;
}
function paintField() {
  const body = $("phone-body");
  const order = state.field === "order" ? currentOrder() : null;
  if (order && order.assigneeId === meId()) ensureFieldAdvice(order);
  const adviceSig = state.fieldAdvice ? `${state.fieldAdvice.phase}:${state.fieldAdvice.text}` : "";
  const talk = state.talk;
  const talkSig = `${talk.phase}:${talk.error}:${talk.messages.map((item) => item.role + item.text).join("|")}`;
  const signature = [state.field, state.orderId, state.siteId, state.draftOrderId, (state.draftActions || []).join(","), adviceSig, talkSig, state.report.phase, state.report.filename, state.report.message, state.gps ? "gps" : "depot", state.pendingPhoto ? state.pendingPhoto.orderId + state.pendingPhoto.url : "", loadOrders().map((item) => [item.id, item.status, item.action, item.urgedAt, item.support ? 1 : 0, photoKey(item), item.messages.length].join("~")).join(","), meId(), loadPeople().map((person) => person.id + person.phone + (person.added ? "1" : "0")).join(","), allSites().map((site) => site.id).join(","), loadCameras().map((item) => item.id).join(",")].join("|");
  if (body.dataset.sig !== signature) {
    body.dataset.sig = signature;
    const pages = { home: fieldHome, orders: fieldOrders, order: fieldOrder, report: fieldReport, nearby: fieldNearby, talk: fieldTalk, me: fieldMe };
    body.innerHTML = (pages[state.field] || fieldHome)();
    if (state.field === "nearby") mountMap("near-map");
  }
  body.classList.toggle("talking", state.field === "talk");
  const thread = body.querySelector(".talk-thread");
  if (thread) thread.scrollTop = thread.scrollHeight;
  body.querySelectorAll("[data-act='pick-site']").forEach((button) => button.classList.toggle("on", button.dataset.id === state.navId));
  body.querySelectorAll("[data-timer]").forEach((node) => { node.textContent = elapsed(loadOrders().find((item) => item.id === node.dataset.timer)); });
  if (order && document.activeElement && document.activeElement.dataset.act !== "order-note") {
    const note = body.querySelector("[data-act='order-note']");
    if (note && note.value !== order.note) note.value = order.note;
  }
}
function paintDesk() {
  ["focus", "alert", "dispatch", "assets", "collab"].forEach((name) => {
    $(`page-${name}`).hidden = state.desk !== name;
  });
  if (state.desk === "focus") { mountMap("map"); paintFocus(); }
  if (state.desk === "alert") paintAlert();
  if (state.desk === "dispatch") paintDispatch();
  if (state.desk === "assets") paintAssets();
  if (state.desk === "collab") paintCollab();
  paintStages();
}
function paint() {
  if (!state.demo) return;
  document.body.dataset.surface = state.surface;
  $("desk").hidden = state.surface !== "desk";
  $("field").hidden = state.surface !== "field";
  paintChrome();
  paintNow();
  if (state.surface === "field") paintField();
  else paintDesk();
  paintNav();
  paintPicker();
  paintShot();
}
function paintPicker() {
  const node = $("picker");
  if (!node) return;
  node.hidden = !state.picker;
  if (!state.picker) return;
  const people = rosterPeople();
  fill("picker-list", `${state.picker.assigneeId}:${people.map((person) => person.id).join(",")}`, people.map((person) => `<button type="button" data-act="pick-assignee" data-id="${person.id}" aria-pressed="${person.id === state.picker.assigneeId}">${esc(person.name)} · ${esc(person.phone)}</button>`).join("") || "<p>还没有处理人。</p>");
}
function paintNav() {
  const node = $("nav-sheet");
  if (!node) return;
  const site = state.surface === "field" && state.field === "nearby" ? allSites().find((item) => item.id === state.navId) : null;
  if (!site) { node.hidden = true; return; }
  const origin = here();
  const distance = distText(km(origin, site));
  const way = bearingText(origin, site);
  fill("nav-sheet", `${site.id}|${distance}|${way}`, `
    <div><strong>${esc(site.name)}</strong><span>${esc(site.code)} · 向${way} · ${esc(distance)}</span></div>
    <div class="actions">
      <button class="btn" type="button" data-act="close-nav">关闭</button>
      <button class="btn primary" type="button" data-act="start-nav">导航</button>
    </div>`);
  node.hidden = false;
}
function openShot(orderId) {
  const order = loadOrders().find((item) => item.id === orderId && item.media);
  if (!order) return;
  if (!state.shot || state.shot.orderId !== orderId) {
    state.shot = { orderId, open: true, zoom: 1, phase: "idle", error: "", messages: [] };
  } else {
    state.shot.open = true;
  }
  paintShot();
}
function closeShot() {
  if (state.shot) state.shot.open = false;
  paintShot();
}
function paintShot() {
  const node = $("shot-view");
  if (!node) return;
  const shot = state.shot;
  const order = shot && shot.open ? loadOrders().find((item) => item.id === shot.orderId && item.assigneeId === meId() && item.media) : null;
  const open = Boolean(order && state.surface === "field");
  node.hidden = !open;
  if (!open) return;
  setText("shot-title", siteById(order.siteId).name);
  const img = $("shot-img");
  const src = mediaSrc(order.media, 1440);
  if (img && img.getAttribute("src") !== src) img.src = src;
  if (img) img.style.width = `${Math.round((shot.zoom || 1) * 100)}%`;
  const stage = $("shot-stage");
  if (stage) stage.classList.toggle("can-drag", (shot.zoom || 1) > 1);
  const rows = shot.messages.map((item) => `<div class="msg${item.role === "assistant" ? "" : " mine"}"><b>${item.role === "assistant" ? "现场助手" : "我"}</b><span>${esc(item.text)}</span></div>`).join("");
  const waiting = shot.phase === "loading" ? `<p class="muted">正在看这张画面…</p>` : "";
  const error = shot.error ? `<p class="ask-error">${esc(shot.error)}</p>` : "";
  const empty = !shot.messages.length && shot.phase !== "loading" ? `<p class="muted">可以问这是烟还是云，或者到了现场先看哪里。</p>` : "";
  const signature = `${shot.phase}|${shot.error}|${shot.messages.map((item) => item.role + item.text).join("|")}`;
  fill("shot-thread", signature, empty + rows + waiting + error);
  const thread = $("shot-thread");
  if (thread) thread.scrollTop = thread.scrollHeight;
}
async function askShot(question) {
  const shot = state.shot;
  const text = String(question || "").trim();
  if (!shot || !shot.open || shot.phase === "loading") return;
  if (!text) { toast("先写一个问题"); return; }
  const order = loadOrders().find((item) => item.id === shot.orderId);
  if (!order) return;
  const site = siteById(order.siteId);
  const prior = shot.messages.slice();
  const history = prior.length && prior[prior.length - 1].role === "user" ? prior.slice(0, -1) : prior;
  const air = state.weather;
  shot.messages = shot.messages.concat({ role: "user", text: text.slice(0, 200) });
  shot.phase = "loading";
  shot.error = "";
  paintShot();
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: text.slice(0, 200),
        history: history.slice(-6).map((item) => ({ role: item.role, text: item.text })),
        media: order.media,
        site: site.name,
        code: site.code,
        elev: site.elev,
        status: order.level === "alarm" ? "建议查看" : order.level === "watch" ? "发现可疑" : "暂无异常",
        summary: order.summary || "",
        temp: air ? air.temp : "",
        humidity: air ? air.humidity : "",
        wind: air ? air.wind : "",
        dir: air ? air.dir : "",
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.message || "询问没有完成，请再试一次。");
    shot.messages = shot.messages.concat({ role: "assistant", text: data.answer || "询问没有完成，请再试一次。" });
    shot.phase = "idle";
  } catch (error) {
    shot.phase = "idle";
    shot.error = error.message || "询问没有完成，请再试一次。";
  }
  paintShot();
}
async function askTalk(question) {
  const talk = state.talk;
  const text = String(question || "").trim();
  if (!talk || talk.phase === "loading") return;
  if (state.surface !== "field" || state.field !== "talk") return;
  if (!text) { toast("先写一个问题"); return; }
  const prior = talk.messages.slice();
  const history = prior.length && prior[prior.length - 1].role === "user" ? prior.slice(0, -1) : prior;
  talk.messages = talk.messages.concat({ role: "user", text: text.slice(0, 200) });
  talk.phase = "loading";
  talk.error = "";
  paint();
  try {
    const response = await fetch("/api/talk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: text.slice(0, 200),
        history: history.slice(-6).map((item) => ({ role: item.role, text: item.text })),
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.message || "询问没有完成，请再试一次。");
    talk.messages = talk.messages.concat({ role: "assistant", text: data.answer || "询问没有完成，请再试一次。" });
    talk.phase = "idle";
  } catch (error) {
    talk.phase = "idle";
    talk.error = error.message || "询问没有完成，请再试一次。";
  }
  paint();
}
function openPicker(options) {
  if (!rosterPeople().length) {
    toast("还没有处理人。现场工作台注册后，到运维调度输入电话号添加。");
    return;
  }
  state.picker = { assigneeId: "", ...options };
  paint();
}
function finishDispatch() {
  const pick = state.picker;
  if (!pick || !pick.assigneeId) { toast("请先选择处理人"); return; }
  const name = personName(pick.assigneeId);
  let order = createOrder({ siteId: pick.siteId, index: pick.index, status: "doing", assigneeId: pick.assigneeId, level: pick.level });
  if (order.assigneeId !== pick.assigneeId) order = setOrder(order.id, { assigneeId: pick.assigneeId, event: { who: "中控", text: `派给${name}` } }) || order;
  else if (!(order.events || []).some((item) => item.text === `派给${name}`)) order = setOrder(order.id, { event: { who: "中控", text: `派给${name}` } }) || order;
  state.orderId = order.id;
  state.picker = null;
  toast(`已派给${name}`);
  go(`dispatch/${order.id}`);
}
function go(hash) {
  const next = `#${hash}`;
  if (location.hash === next) applyHash();
  else location.hash = next;
}
function applyHash() {
  play(false);
  const parts = (location.hash.replace(/^#/, "") || "focus").split("/");
  if (parts[0] === "field") {
    state.surface = "field";
    state.field = parts[1] === "know" ? "talk" : (parts[1] || "home");
    if (parts[2]) state.orderId = parts[2];
  } else {
    state.surface = "desk";
    const names = ["focus", "alert", "dispatch", "assets", "collab"];
    state.desk = names.includes(parts[0]) ? parts[0] : "focus";
    if (parts[1] && (state.desk === "alert" || state.desk === "assets")) state.siteId = parts[1];
    if (parts[1] && (state.desk === "dispatch" || state.desk === "collab")) state.orderId = parts[1];
  }
  paint();
  if (state.surface === "desk" && state.desk === "alert") play(true);
}
function play(on) {
  state.playing = Boolean(on);
  clearInterval(playTimer);
  if (state.playing) playTimer = setInterval(() => step(1, true), SPEEDS[state.speed]);
  setText("watch-play", state.playing ? "暂停" : "继续");
}
function step(dir, fromPlay) {
  if (!state.demo) return;
  const next = state.index + dir;
  if (fromPlay && state.watch.status !== "done" && state.watch.status !== "error") {
    const upcoming = state.demo.frames[next];
    if (upcoming && upcoming[state.mode].pending) return;
  }
  if (next < 0 || next >= state.demo.frames.length) {
    if (fromPlay && state.surface === "desk" && state.desk === "alert") {
      state.index = dir > 0 ? 0 : state.demo.frames.length - 1;
      paint();
      return;
    }
    if (fromPlay) play(false);
    return;
  }
  state.index = next;
  paint();
}
function edgeFromView(view) {
  const smoke = scorePct(view.score) || 0;
  const risk = view.alarm ? Math.max(smoke, 70) : view.kept ? Math.max(smoke, 45) : view.candidates ? 28 : 8;
  return [
    { name: "烟雾", value: smoke },
    { name: "建议查看", value: risk },
    { name: "不像烟", value: Math.max(0, 100 - smoke) },
  ];
}
async function ensureSession() {
  if (state.sessionId) return state.sessionId;
  const response = await fetch("/api/sessions", { method: "POST" });
  const payload = await response.json();
  state.sessionId = payload.id;
  return state.sessionId;
}
function safeName(name) {
  const clean = String(name || "photo.jpg").replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_");
  return `${Date.now()}_${clean}`.slice(-80);
}
async function storeChatPhoto(file) {
  const filename = safeName(file.name);
  const response = await fetch(`/api/chat-photos?name=${encodeURIComponent(filename)}`, { method: "POST", body: file });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.message || "图片没有发出去");
  return { kind: "chat", filename: payload.filename };
}
async function storePhoto(file) {
  const id = await ensureSession();
  const filename = safeName(file.name);
  const response = await fetch(`/api/sessions/${id}/photos?name=${encodeURIComponent(filename)}`, { method: "POST", body: file });
  if (!response.ok) throw new Error((await response.json()).message || "照片没有传上去。");
  return { kind: "mine", sessionId: id, filename };
}
async function exportWord() {
  const site = currentSite();
  const view = viewFromIndex(state.index);
  let image = "";
  try {
    const response = await fetch(`/api/media/demo/${encodeURIComponent(view.filename)}?size=1440`);
    const blob = await response.blob();
    image = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.readAsDataURL(blob);
    });
  } catch { image = ""; }
  const word = site.live ? monitorWord(view) : "暂无异常";
  const steps = (state.judge.actions || []).map((item) => `<li>${esc(item)}</li>`).join("");
  const written = state.judge.phase === "ready" ? `<h2>${esc(state.judge.headline)}</h2>${(state.judge.paragraphs || []).map((item) => `<p>${esc(item)}</p>`).join("")}${steps ? `<ol>${steps}</ol>` : ""}` : `<p>${esc(word)}</p>`;
  const html = `<html><head><meta charset="utf-8"></head><body><h1>烟雾瞭望研判 ${site.name}</h1><p>${clockLabel(view.offset)} · ${word}</p>${image ? `<img src="${image}" width="640">` : ""}${written}<p>${site.lat}, ${site.lng} · 海拔 ${site.elev} 米</p></body></html>`;
  const blob = new Blob([html], { type: "application/msword" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `烟雾瞭望研判-${site.code}.doc`;
  document.body.append(link);
  link.click();
  link.remove();
  toast("已导出，可以用 Word 打开再改");
}
async function cloudCheck() {
  if (!state.report.file) { toast("先选一张照片，或改用当前画面"); return; }
  state.report.phase = "running";
  state.report.message = "正在做云端复核…";
  paint();
  try {
    const media = await storePhoto(state.report.file);
    state.report.media = media;
    const response = await fetch(`/api/sessions/${media.sessionId}/run`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
    });
    if (!response.ok) throw new Error((await response.json()).message || "没有开始复核。");
    clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      const status = await fetch(`/api/sessions/${media.sessionId}`).then((item) => item.json());
      if (status.status === "running") return;
      clearInterval(pollTimer);
      if (status.status === "done" && status.verdict) {
        const score = Math.round((status.verdict.score || 0) * 100);
        state.report.edge = [
          { name: "烟雾", value: score },
          { name: "建议查看", value: status.verdict.alarm ? Math.max(score, 70) : 20 },
          { name: "不像烟", value: Math.max(0, 100 - score) },
        ];
        state.report.source = "cloud";
        state.report.message = status.verdict.alarm ? "云端复核建议有人看一眼。" : "云端复核没有到建议查看的程度。";
      } else {
        state.report.message = status.error || "复核没有完成。";
        if (status.retry === "cpu") state.report.message += " 可以再试一次，改用本机计算。";
      }
      state.report.phase = status.status;
      paint();
    }, 1000);
  } catch (error) {
    state.report.phase = "error";
    state.report.message = error.message;
    paint();
  }
}
async function submitReport() {
  const report = state.report;
  if (!report.preview && !report.filename) { toast("先放入一张照片"); return; }
  if (report.file && !report.media) {
    try { report.media = await storePhoto(report.file); }
    catch (error) { toast(error.message); }
  }
  const site = siteById("syp-n");
  const view = report.source === "demo" ? viewFromIndex(state.index) : null;
  const summary = view ? describe(view) : { level: "watch", text: report.text || "现场上报了一张照片，等待中控看。", score: "" };
  const order = makeOrder({
    siteId: site.id,
    index: view ? view.index : null,
    title: report.text ? report.text.slice(0, 28) : "现场上报",
    level: summary.level === "ok" ? "watch" : summary.level,
    status: "doing",
    note: report.text,
  });
  if (report.media) order.media = report.media;
  if (!view) order.media = report.media || null;
  order.summary = report.text || summary.text;
  order.events.push({ at: Date.now(), who: meName(), text: "现场提交上报" });
  const orders = loadOrders();
  orders.unshift(order);
  saveOrders(orders);
  state.orderId = order.id;
  state.report = { preview: "", filename: "", source: "", file: null, text: "", edge: null, phase: "", message: "" };
  toast("已提交到中控");
  go(`field/order/${order.id}`);
}
async function resetDemo() {
  const response = await fetch("/api/board/reset", { method: "POST" });
  const data = await response.json();
  dirty = false;
  takeBoard(data);
  toast("已恢复示例台账");
  paint();
}
function onNote(node) {
  setOrder(node.dataset.id, { note: node.value });
}

function authNote(form, text) {
  const note = form && form.querySelector("[data-auth-note]");
  if (note) note.textContent = text || "";
}
document.addEventListener("click", (event) => {
  const node = event.target.closest("[data-act], [data-desk], [data-field-tab], [data-mode], [data-layer], [data-filter], [data-asset-tab], [data-zoom]");
  if (!node || node.tagName === "FORM") return;
  if (node.dataset.desk) { go(node.dataset.desk); return; }
  if (node.dataset.fieldTab) { go(`field/${node.dataset.fieldTab}`); return; }
  if (node.dataset.mode) { state.mode = node.dataset.mode; paint(); return; }
  if (node.dataset.layer) {
    state.layers[node.dataset.layer] = node.getAttribute("aria-pressed") !== "true";
    paint();
    return;
  }
  if (node.dataset.filter && node.closest("#d-filters, #d-counts")) { state.orderFilter = node.dataset.filter; paint(); return; }
  if (node.dataset.assetTab) { state.assetTab = node.dataset.assetTab; paint(); return; }
  const act = node.dataset.act;
  if (act === "pick-skill") {
    state.skill = node.dataset.skill;
    const group = node.parentElement;
    group.querySelectorAll("button").forEach((button) => button.setAttribute("aria-pressed", String(button === node)));
    const input = node.closest("form").querySelector("[name=skill]");
    if (input) input.value = node.dataset.skill;
    return;
  }
  if (act === "logout") {
    localStorage.removeItem(ME_KEY);
    state.authMode = "login";
    state.skill = "";
    toast("已退出");
    paint();
    return;
  }
  if (act === "field") { go(`field/${node.dataset.field}`); return; }
  if (act === "open-live") { state.siteId = currentSite().live ? state.siteId : "syp-n"; go("alert"); return; }
  if (act === "fullscreen") { $("lightbox").hidden = false; paintStages(); return; }
  if (act === "close-full") { $("lightbox").hidden = true; return; }
  if (act === "open-shot") { openShot(node.dataset.id); return; }
  if (act === "close-shot") { closeShot(); return; }
  if (act === "shot-zoom") {
    if (!state.shot) return;
    const steps = [1, 1.5, 2, 3];
    const index = Math.max(0, steps.indexOf(state.shot.zoom || 1));
    const next = index + (node.dataset.dir === "-1" ? -1 : 1);
    state.shot.zoom = steps[Math.min(steps.length - 1, Math.max(0, next))];
    paintShot();
    return;
  }
  if (act === "ask-prompt") { askShot(node.dataset.q); return; }
  if (act === "talk-prompt") { askTalk(node.dataset.q); return; }
  if (act === "advice-retry") {
    const order = loadOrders().find((item) => item.id === node.dataset.id);
    if (state.fieldAdvice && order && state.fieldAdvice.orderId === order.id) state.fieldAdvice.phase = "idle";
    ensureFieldAdvice(order);
    paint();
    return;
  }
  if (act === "seek") {
    play(false);
    state.index = Number(node.dataset.index);
    if (node.closest("#sum-alerts")) {
      state.sumKey = `seek:${state.index}`;
      state.siteId = "syp-n";
      paint();
      const site = siteById("syp-n");
      Object.values(state.maps).forEach((map) => { if (map) map.focusUpper(site); });
      return;
    }
    paint();
    return;
  }
  if (act === "step") { play(false); step(Number(node.dataset.dir)); return; }
  if (act === "judge-retry") {
    const view = state.demo ? viewFromIndex(state.index) : null;
    const word = view && currentSite().live ? monitorWord(view) : "暂无异常";
    if (word === "暂无异常") { toast("当前没有需要研判的烟雾"); return; }
    state.judgeRefresh = true;
    state.judge = { sig: "", phase: "idle", word: "", headline: "", paragraphs: [], message: "" };
    paint();
    return;
  }
  if (act === "play") { play(!state.playing); return; }
  if (act === "speed") { state.speed = (state.speed + 1) % SPEEDS.length; if (state.playing) play(true); paint(); return; }
  if (act === "pick-site") { chooseSite(node.dataset.id); return; }
  if (act === "close-nav") { state.navId = ""; paint(); return; }
  if (act === "start-nav") {
    const site = allSites().find((item) => item.id === state.navId);
    if (!site) return;
    const opened = window.open(navHref(site), "_blank", "noopener");
    if (!opened) toast("浏览器拦住了导航页面");
    return;
  }
  if (act === "goto-order") { state.orderId = node.dataset.id; go(`dispatch/${node.dataset.id}`); return; }
  if (act === "select-order") { state.orderId = node.dataset.id; go(`collab/${node.dataset.id}`); return; }
  if (act === "field-order") {
    if (!node.dataset.id) { toast("现在没有紧急工单"); return; }
    state.orderId = node.dataset.id;
    go(`field/order/${node.dataset.id}`);
    return;
  }
  if (act === "dispatch-now") {
    const site = siteById(state.siteId);
    if (!site.live || !state.demo) { toast("这个点没有连续画面"); return; }
    const word = monitorWord(viewFromIndex(state.index));
    if (word === "暂无异常") { toast("当前没有需要派单的烟雾"); return; }
    play(false);
    openPicker({ mode: "dispatch-now", siteId: site.id, index: state.index, level: word === "建议查看" ? "alarm" : "watch" });
    return;
  }
  if (act === "quick-dispatch") {
    const site = siteById(state.siteId);
    openPicker({ mode: "quick-dispatch", siteId: site.id, index: site.live ? alarmIndex() : null });
    return;
  }
  if (act === "pick-assignee") {
    if (!state.picker) return;
    state.picker.assigneeId = node.dataset.id;
    paintPicker();
    return;
  }
  if (act === "picker-cancel") { state.picker = null; paintPicker(); return; }
  if (act === "picker-ok") { finishDispatch(); return; }
  if (act === "show-add") {
    state.addForm = state.addForm === node.dataset.add ? "" : node.dataset.add;
    paint();
    return;
  }
  if (act === "create-site") {
    const site = siteById(node.dataset.id);
    const order = createOrder({ siteId: site.id, index: site.live ? state.index : null });
    state.orderId = order.id;
    toast("已建单");
    go(`dispatch/${order.id}`);
    return;
  }
  if (act === "status") {
    const status = node.dataset.status === "pending" ? "doing" : node.dataset.status;
    const label = STATUS[status];
    if (!label) return;
    setOrder(node.dataset.id, { status, event: { who: state.surface === "field" ? meName() : "中控", text: `状态改为${label}` } });
    toast(label);
    paint();
    return;
  }
  if (act === "pick-action") {
    const orderId = node.dataset.id;
    const action = node.dataset.action;
    if (!ACTIONS[action]) return;
    const order = loadOrders().find((item) => item.id === orderId);
    const current = state.draftOrderId === orderId && Array.isArray(state.draftActions) ? state.draftActions : savedActions(order);
    state.draftOrderId = orderId;
    state.draftActions = current.includes(action) ? current.filter((id) => id !== action) : current.concat(action);
    paint();
    return;
  }
  if (act === "submit-action") { submitField(node.dataset.id); return; }
  if (act === "urge") {
    setOrder(node.dataset.id, { urgedAt: Date.now(), event: { who: "中控", text: "催办：请尽快处理" } });
    toast("已催办");
    paint();
    return;
  }
  if (act === "support") {
    const current = loadOrders().find((item) => item.id === node.dataset.id);
    if (!current) return;
    const who = state.surface === "field" ? meName() : "中控";
    if (current.support) {
      setOrder(current.id, { support: false, event: { who, text: "取消了支援请求" } });
      toast("已取消支援");
    } else {
      setOrder(current.id, { support: true, event: { who, text: "请求支援" } });
      toast("已请求支援");
    }
    paint();
    return;
  }
  if (act === "delete-photo") {
    if (!state.pendingPhoto || state.pendingPhoto.orderId !== node.dataset.id) return;
    clearPendingPhoto();
    paint();
    return;
  }
  if (act === "review") {
    if (node.dataset.review === "pass") {
      setOrder(node.dataset.id, { status: "done", event: { who: "中控", text: "复核通过" } });
      toast("已通过");
    } else {
      deleteOrder(node.dataset.id);
      toast("已删除");
    }
    paint();
    return;
  }
  if (act === "export") { exportWord(); return; }
  if (act === "shoot") { $("file-report").click(); return; }
  if (act === "after") { $("file-after").click(); return; }
  if (act === "use-live") {
    const view = viewFromIndex(state.index);
    state.report.preview = `/api/media/demo/${encodeURIComponent(view.filename)}?size=1440`;
    state.report.filename = view.filename;
    state.report.source = "demo";
    state.report.file = null;
    state.report.edge = edgeFromView(view);
    state.report.media = { kind: "demo", filename: view.filename, index: state.index };
    state.report.message = "这是当前画面已经算好的初判。";
    paint();
    return;
  }
  if (act === "locate") {
    if (!navigator.geolocation) { toast("这台浏览器没有定位"); return; }
    navigator.geolocation.getCurrentPosition((pos) => {
      state.gps = { lat: pos.coords.latitude, lng: pos.coords.longitude };
      toast("已更新定位");
      loadLive(state.gps.lat, state.gps.lng);
    }, () => toast("没有拿到定位，仍用市区参考点"));
    return;
  }
  if (act === "cloud") { cloudCheck(); return; }
  if (act === "submit-report") { submitReport(); return; }
  if (act === "reset-demo") { resetDemo(); return; }
  if (act === "pick-person") { state.personId = node.dataset.id; paint(); return; }
  if (act === "drop-person") {
    const people = loadPeople();
    const person = people.find((item) => item.id === node.dataset.id);
    if (person) person.added = false;
    savePeople(people);
    toast("已从处理人移出");
    paint();
    return;
  }
});
document.addEventListener("input", (event) => {
  const node = event.target;
  if (node.dataset.act === "asset-q") { state.assetQuery = node.value; paintAssets(); return; }
  if (node.dataset.act === "people-q") { state.peopleQuery = node.value; paintAssets(); return; }
  if (node.dataset.act === "report-text") { state.report.text = node.value; return; }
  if (node.dataset.act === "order-note") onNote(node);
  if (node.id === "scrub") { play(false); state.index = Number(node.value); paint(); }
});
document.addEventListener("change", async (event) => {
  const node = event.target;
  if (node.id === "scrub" && state.surface === "desk" && state.desk === "alert") play(true);
  if (node.dataset.act === "assign") {
    setOrder(node.dataset.id, { assigneeId: node.value, event: { who: "中控", text: node.value ? `改派给${personName(node.value)}` : "取消了处理人" } });
    toast("已改派");
    paint();
  }
  if (node.dataset.act === "me") { setMe(node.value); paint(); }
  if (node.id === "file-report" && node.files[0]) {
    const file = node.files[0];
    state.report.file = file;
    state.report.preview = URL.createObjectURL(file);
    state.report.filename = file.name;
    state.report.source = "file";
    state.report.edge = null;
    state.report.media = null;
    state.report.message = "照片已在本地。可以提交到中控，或做云端复核。";
    node.value = "";
    paint();
  }
  if (node.type === "file" && node.closest("form[data-act='chat']")) {
    const label = node.closest(".chat-file");
    if (label) label.classList.toggle("picked", Boolean(node.files && node.files[0]));
  }
  if (node.id === "file-after" && node.files[0]) {
    clearPendingPhoto();
    state.pendingPhoto = { orderId: state.orderId, url: URL.createObjectURL(node.files[0]), file: node.files[0] };
    node.value = "";
    paint();
  }
});
document.addEventListener("submit", (event) => {
  const form = event.target.closest("form[data-act]");
  if (!form) return;
  event.preventDefault();
  if (form.dataset.act === "ask-shot") {
    const input = form.querySelector("input");
    const question = input.value;
    input.value = "";
    askShot(question);
    return;
  }
  if (form.dataset.act === "ask-talk") {
    const input = form.querySelector("input");
    const question = input.value;
    input.value = "";
    askTalk(question);
    return;
  }
  if (form.dataset.act === "chat") {
    const input = form.querySelector("[name=text]");
    const fileInput = form.querySelector("input[type=file]");
    const text = input ? input.value : "";
    const file = fileInput && fileInput.files[0];
    const who = state.surface === "field" ? meName() : "中控";
    const id = form.dataset.id;
    if (!String(text || "").trim() && !file) { toast("写一句话，或选一张图片"); return; }
    if (input) input.value = "";
    if (fileInput) fileInput.value = "";
    const label = form.querySelector(".chat-file");
    if (label) label.classList.remove("picked");
    if (!file) {
      addMessage(id, who, text);
      paint();
      return;
    }
    toast("正在发送图片");
    storeChatPhoto(file).then((media) => {
      addMessage(id, who, text, media);
      paint();
    }).catch((error) => toast(error.message || "图片没有发出去"));
  }
  if (form.dataset.act === "register") {
    const data = new FormData(form);
    const intent = event.submitter && event.submitter.value === "login" ? "login" : "register";
    const name = String(data.get("name") || "").trim();
    const phone = phoneDigits(data.get("phone"));
    const password = String(data.get("password") || "");
    if (intent === "register" && state.authMode !== "register") {
      state.authMode = "register";
      form.querySelectorAll("[data-auth='register']").forEach((node) => { node.hidden = false; });
      authNote(form, "填写姓名、擅长和区域后，再点一次注册。");
      const nameInput = form.querySelector("[name=name]");
      if (nameInput) nameInput.focus();
      return;
    }
    if (intent === "login") {
      state.authMode = "login";
      form.querySelectorAll("[data-auth='register']").forEach((node) => { node.hidden = true; });
    }
    if (!validPhone(phone)) { authNote(form, "请填写11位手机号。"); toast("请填写11位手机号"); return; }
    if (!password) { authNote(form, "请填写密码。"); toast("请填写密码"); return; }
    if (intent === "login") {
      const person = loadPeople().find((item) => item.phone === phone && item.password === password);
      if (!person) { authNote(form, "电话或密码不对。"); toast("电话或密码不对。"); return; }
      setMe(person.id);
      toast("已登录");
      paint();
      return;
    }
    if (!name) { authNote(form, "请填写姓名。"); toast("请填写姓名"); return; }
    if (loadPeople().some((person) => person.phone === phone)) { authNote(form, "这个电话已经注册过。"); toast("这个电话已经注册过"); return; }
    const person = {
      id: uid(),
      name,
      phone,
      password,
      skill: String(data.get("skill") || "烟雾复核"),
      region: String(data.get("region") || "").trim(),
      historyDone: 0,
      added: false,
    };
    savePeople(loadPeople().concat(person));
    setMe(person.id);
    toast("已注册");
    paint();
  }
  if (form.dataset.act === "add-by-phone") {
    const phone = phoneDigits(new FormData(form).get("phone"));
    if (!validPhone(phone)) { toast("请填写11位手机号"); return; }
    const person = loadPeople().find((item) => item.phone === phone);
    if (!person) { toast("这个电话还没有在现场工作台注册"); return; }
    if (person.added) { toast("这个处理人已经添加过"); return; }
    person.added = true;
    savePeople(loadPeople());
    form.reset();
    toast(`已添加${person.name}`);
    paint();
  }
  if (form.dataset.act === "add-site") {
    const data = new FormData(form);
    const name = String(data.get("name") || "").trim();
    const code = String(data.get("code") || "").trim().toUpperCase();
    const lat = Number(String(data.get("lat") || "").trim());
    const lng = Number(String(data.get("lng") || "").trim());
    const elevText = String(data.get("elev") || "").trim();
    const elev = elevText ? Number(elevText) : 0;
    const region = String(data.get("region") || "").trim();
    if (!name) { toast("请填写点位名称"); return; }
    if (!/^[A-Z0-9-]{2,12}$/.test(code)) { toast("编号用 2 到 12 位字母或数字"); return; }
    if (allSites().some((site) => site.code.toUpperCase() === code)) { toast("这个编号已经有了"); return; }
    if (!Number.isFinite(lat) || lat < -90 || lat > 90 || !Number.isFinite(lng) || lng < -180 || lng > 180) { toast("请填写有效的经纬度"); return; }
    if (!Number.isFinite(elev) || elev < -500 || elev > 9000) { toast("海拔请填数字"); return; }
    board.sites = (board.sites || []).concat({
      id: uid(),
      code,
      name: name.slice(0, 20),
      line: "自加点位",
      region: region.slice(0, 20),
      lat,
      lng,
      elev: Math.round(elev),
      live: false,
      kind: "瞭望点",
      tone: "ok",
      status: "正常",
      detail: "手动添加的点位，没有连续画面。",
    });
    savePlaces();
    form.reset();
    state.addForm = "";
    toast("已添加点位");
    paint();
    Object.values(state.maps).forEach((map) => { if (map) map.drawPins(); });
  }
  if (form.dataset.act === "add-camera") {
    const data = new FormData(form);
    const siteId = String(data.get("siteId") || "");
    const name = String(data.get("name") || "").trim();
    if (!siteById(siteId) || !allSites().some((site) => site.id === siteId)) { toast("请选择点位"); return; }
    if (!name) { toast("请填写摄像头名称"); return; }
    const ip = String(data.get("ip") || "").trim();
    if (!validIp(ip)) { toast("请填写摄像头 IP"); return; }
    if (loadCameras().some((item) => item.siteId === siteId && item.name === name)) { toast("这个点位已经有同名摄像头"); return; }
    board.cameras = loadCameras().concat({ id: uid(), siteId, name: name.slice(0, 20), ip });
    savePlaces();
    form.reset();
    state.addForm = "";
    state.siteId = siteId;
    toast("已添加摄像头");
    paint();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    $("lightbox").hidden = true;
    closeShot();
  }
  const tag = document.activeElement && document.activeElement.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || state.surface !== "desk") return;
  if (event.key === "ArrowRight") step(1);
  if (event.key === "ArrowLeft") step(-1);
  if (event.key === " ") { event.preventDefault(); play(!state.playing); }
  if (event.key === "Escape") $("lightbox").hidden = true;
});
window.addEventListener("hashchange", applyHash);
window.addEventListener("storage", (event) => { if (!event.key || event.key.startsWith("aicseek-")) paint(); });

async function pullWatch() {
  if (!state.demo) return;
  try {
    const response = await fetch("/api/watch");
    if (!response.ok) return;
    const data = await response.json();
    const before = `${state.watch.status}:${state.watch.done}`;
    state.watch = { status: data.status || "idle", done: data.done || 0, total: data.total || 0, message: data.message || "" };
    let changed = before !== `${state.watch.status}:${state.watch.done}`;
    for (const row of data.frames || []) {
      const frame = state.demo.frames.find((item) => item.filename === row.filename);
      if (!frame || !frame.standard.pending) continue;
      frame.standard = row.standard;
      frame.sensitive = row.sensitive || row.standard;
      changed = true;
    }
    if (data.modes) state.demo.modes = data.modes;
    if (changed) paint();
    if (watchTimer && (state.watch.status === "done" || state.watch.status === "error")) {
      clearInterval(watchTimer);
      watchTimer = 0;
    }
  } catch { /* 原图仍可播放，检查结果稍后接上 */ }
}
function mountOverlayScroll() {
  const vertical = document.createElement("div");
  const horizontal = document.createElement("div");
  vertical.className = "hover-scroll y";
  horizontal.className = "hover-scroll x";
  vertical.hidden = true;
  horizontal.hidden = true;
  document.body.append(vertical, horizontal);
  let active = null;
  const canScroll = (el, axis) => {
    const style = getComputedStyle(el);
    const mode = axis === "y" ? style.overflowY : style.overflowX;
    if (!/(auto|scroll|overlay)/.test(mode)) return false;
    return axis === "y" ? el.scrollHeight > el.clientHeight + 1 : el.scrollWidth > el.clientWidth + 1;
  };
  const findScroll = (node) => {
    while (node && node !== document.documentElement) {
      if (node.id === "shot-stage") { node = node.parentElement; continue; }
      if (node.nodeType === 1 && (canScroll(node, "y") || canScroll(node, "x"))) return node;
      node = node.parentElement;
    }
    return null;
  };
  const place = () => {
    if (!active || !active.isConnected) {
      vertical.hidden = true;
      horizontal.hidden = true;
      return;
    }
    const box = active.getBoundingClientRect();
    if (canScroll(active, "y")) {
      const max = active.scrollHeight - active.clientHeight;
      const thumb = Math.max(28, box.height * active.clientHeight / active.scrollHeight);
      const travel = Math.max(1, box.height - thumb);
      vertical.hidden = false;
      vertical.style.left = `${box.right - 10}px`;
      vertical.style.top = `${box.top + (active.scrollTop / max) * travel}px`;
      vertical.style.height = `${thumb}px`;
    } else vertical.hidden = true;
    if (canScroll(active, "x")) {
      const max = active.scrollWidth - active.clientWidth;
      const thumb = Math.max(28, box.width * active.clientWidth / active.scrollWidth);
      const travel = Math.max(1, box.width - thumb);
      horizontal.hidden = false;
      horizontal.style.top = `${box.bottom - 10}px`;
      horizontal.style.left = `${box.left + (active.scrollLeft / max) * travel}px`;
      horizontal.style.width = `${thumb}px`;
    } else horizontal.hidden = true;
  };
  const drag = (bar, axis) => {
    bar.addEventListener("pointerdown", (event) => {
      if (!active) return;
      event.preventDefault();
      const start = axis === "y" ? event.clientY : event.clientX;
      const origin = axis === "y" ? active.scrollTop : active.scrollLeft;
      const move = (e) => {
        const box = active.getBoundingClientRect();
        if (axis === "y") {
          const max = active.scrollHeight - active.clientHeight;
          const thumb = Math.max(28, box.height * active.clientHeight / active.scrollHeight);
          active.scrollTop = origin + ((e.clientY - start) / Math.max(1, box.height - thumb)) * max;
        } else {
          const max = active.scrollWidth - active.clientWidth;
          const thumb = Math.max(28, box.width * active.clientWidth / active.scrollWidth);
          active.scrollLeft = origin + ((e.clientX - start) / Math.max(1, box.width - thumb)) * max;
        }
        place();
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
  };
  drag(vertical, "y");
  drag(horizontal, "x");
  document.addEventListener("mouseover", (event) => {
    if (event.target === vertical || event.target === horizontal) return;
    const next = findScroll(event.target);
    if (next === active) { place(); return; }
    active = next;
    place();
  });
  document.addEventListener("mouseout", (event) => {
    if (!active) return;
    const next = event.relatedTarget;
    if (next && (next === vertical || next === horizontal || active.contains(next))) return;
    active = null;
    vertical.hidden = true;
    horizontal.hidden = true;
  });
  document.addEventListener("scroll", () => place(), true);
  window.addEventListener("resize", () => place());
}
function mountShotDrag() {
  const stage = $("shot-stage");
  if (!stage || stage.dataset.drag) return;
  stage.dataset.drag = "1";
  let drag = null;
  stage.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !state.shot || (state.shot.zoom || 1) <= 1) return;
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY, left: stage.scrollLeft, top: stage.scrollTop };
    try { stage.setPointerCapture(event.pointerId); } catch { /* 指针还在画面里时照样拖动 */ }
    stage.classList.add("dragging");
  });
  stage.addEventListener("pointermove", (event) => {
    if (!drag || drag.id !== event.pointerId) return;
    stage.scrollLeft = drag.left - (event.clientX - drag.x);
    stage.scrollTop = drag.top - (event.clientY - drag.y);
  });
  const end = (event) => {
    if (!drag || (event && drag.id !== event.pointerId)) return;
    drag = null;
    stage.classList.remove("dragging");
  };
  stage.addEventListener("pointerup", end);
  stage.addEventListener("pointercancel", end);
}
async function boot() {
  try {
    const [demoRes, readyRes, boardRes, configRes] = await Promise.all([fetch("/api/demo"), fetch("/api/ready"), fetch("/api/board"), fetch("/api/config")]);
    if (!demoRes.ok || !boardRes.ok) throw new Error("demo");
    state.demo = await demoRes.json();
    state.ready = readyRes.ok ? await readyRes.json() : { model: false };
    if (configRes.ok) {
      const config = await configRes.json();
      state.amapKey = config.amapKey || "";
      state.amapSecurity = config.amapSecurity || "";
    }
    takeBoard(await boardRes.json());
  } catch {
    $("boot").textContent = "没有连上本机页面。请先启动后再刷新。";
    return;
  }
  state.index = 0;
  state.sumKey = `seek:${appearIndex()}`;
  const fieldEntry = location.pathname === "/field";
  if (fieldEntry && (!location.hash || location.hash === "#")) {
    history.replaceState(null, "", "/field#field/home");
  }
  mountOverlayScroll();
  mountShotDrag();
  if (fieldEntry) {
    $("boot").hidden = true;
    applyHash();
    ensureAmap();
  } else {
    await ensureAmap();
    $("boot").hidden = true;
    applyHash();
  }
  loadLive(34.53, -119.98);
  pullWatch();
  setInterval(() => { paintNow(); document.querySelectorAll("[data-timer]").forEach((node) => { const order = loadOrders().find((item) => item.id === node.dataset.timer); if (order) node.textContent = elapsed(order); }); }, 1000);
  setInterval(pullBoard, 2000);
  watchTimer = setInterval(pullWatch, 1000);
}
boot();

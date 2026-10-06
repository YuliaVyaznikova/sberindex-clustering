const COLORS = { 2: "#3A86E0", 1: "#E0662E", 0: "#18A77B", 3: "#C08400", 4: "#D8628F", 5: "#8F6FE6" };
const LEGEND = { 2: "Города и агломерации", 1: "Промышленные малые города", 0: "Стареющая периферия", 3: "Молодые сельские районы", 4: "Удалённые высокозарплатные", 5: "Юг и Северный Кавказ" };
const PERIODS = { "2023Q4": "2023 год", "2024Q1": "апр 2023 - мар 2024", "2024Q2": "июл 2023 - июн 2024", "2024Q3": "окт 2023 - сен 2024", "2024Q4": "2024 год" };
const STEPS = { "2023Q4": "2023", "2024Q1": "до мар 24", "2024Q2": "до июн 24", "2024Q3": "до сен 24", "2024Q4": "2024" };
const CONF = [["#2E4618", "ниже 0.5"], ["#557A26", "0.5-0.7"], ["#86B032", "0.7-0.9"], ["#C7EB6A", "0.9 и выше"]];
const CHANGE = { reliable: "#B5E03A", weak: "#5E7A2A", same: "#34433A" };
const NODATA = "#1F2A23";
const MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];
const METHOD_NAMES = { "k-means": "k-means", Ward: "Уорд", GMM: "GMM", spectral: "спектральная по графу", "fused spectral": "без уточнения", "KEFRiN euclidean": "KEFRiN, евклид", "KEFRiN cosine": "KEFRiN, косинус", "refined spectral": "SPECTRA" };

Promise.all(["atlas", "shapes", "network", "extra"].map((n) => d3.json(`data/${n}.json`))).then(([atlas, shapes, network, extra]) => start(atlas, shapes, network, extra));

function start(atlas, shapes, network, extra) {
  const last = atlas.periods.length - 1;
  const order = atlas.order;
  const state = { period: last, mode: "type", filter: null, pick: null };
  const records = new Map(atlas.mo.map((r) => [r[0], { id: r[0], name: r[1], region: r[2], kinds: r[3], conf: r[4], flag: r[5] }]));
  const blank = atlas.periods.map(() => -1);
  [[9001, "Москва"], [9002, "Санкт-Петербург"]].forEach(([id, name]) => {
    if (shapes.paths[id] && !records.has(id)) records.set(id, { id, name, region: name, kinds: blank, conf: blank, flag: 0 });
  });
  const nodes = network.nodes.map((n) => ({ id: n[0], x: n[1], y: n[2], type: n[3] }));
  const netIndex = new Map(nodes.map((n, i) => [n.id, i]));
  const tip = document.getElementById("tip");
  const fmt = (v, d) => v.toFixed(d);
  const pct = (v, d = 0) => (v * 100).toFixed(d) + "%";
  const signed = (v, d = 0, unit = "%") => (v > 0 ? "+" : "") + v.toFixed(d) + unit;
  const dot = (t) => `<i class="dot" style="background:${t >= 0 ? COLORS[t] : NODATA}"></i>`;
  const norm = (s) => s.toLowerCase().replace(/ё/g, "е");
  const plural = (n, forms) => forms[n % 10 === 1 && n % 100 !== 11 ? 0 : n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 10 || n % 100 >= 20) ? 1 : 2];
  const shortName = (s) => s.replace(/^(городской|муниципальный) округ (город )?/, "").replace(" муниципальный район", " район").replace(" муниципальный округ", " округ").replace(/^город /, "");
  const shortRegion = (s) => s.replace("автономный округ", "АО").replace("область", "обл.").replace("Республика", "Респ.").replace(" край", " кр.");
  const views = [];

  function showTip(html, x, y) {
    tip.innerHTML = html;
    tip.hidden = false;
    const box = tip.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + box.width > window.innerWidth - 8) left = x - box.width - 14;
    if (top + box.height > window.innerHeight - 8) top = y - box.height - 14;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }
  const hideTip = () => { tip.hidden = true; };
  window.addEventListener("scroll", hideTip, { passive: true });

  function select(id) {
    state.pick = id;
    views.forEach((v) => v.pick(id));
    passport(id);
  }

  hero();
  tiles();
  const mapApi = mapView();
  views.push(mapApi);
  views.push(netView());
  cards();
  flows();
  changes();
  market();
  model();
  const example = atlas.mo.find((r) => r[1] === "городской округ город Томск");
  select(example ? example[0] : null);

  function hero() {
    const canvas = document.getElementById("hero-net");
    function draw() {
      const W = canvas.clientWidth, H = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
      if (!W || !H) return;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      const ctx = canvas.getContext("2d");
      ctx.scale(dpr, dpr);
      const s = Math.min(W, H * 1.4) * 0.92;
      const ox = W - s - 10, oy = (H - s / 1.4) / 2;
      const px = (n) => ox + n.x * s, py = (n) => oy + n.y * (s / 1.4);
      ctx.strokeStyle = "rgba(181,224,58,0.05)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      network.edges.forEach(([i, j]) => { ctx.moveTo(px(nodes[i]), py(nodes[i])); ctx.lineTo(px(nodes[j]), py(nodes[j])); });
      ctx.stroke();
      nodes.forEach((n) => {
        ctx.fillStyle = n.type >= 0 ? COLORS[n.type] : NODATA;
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        ctx.arc(px(n), py(n), 1.7, 0, 2 * Math.PI);
        ctx.fill();
      });
      ctx.globalAlpha = 1;
    }
    draw();
    onResize(canvas, draw);
  }

  function tiles() {
    const t = atlas.tiles;
    const items = [
      [String(t.types), `типов местных экономик нашёл метод SPECTRA в сети из ${atlas.mo.length} муниципальных образований (МО)`],
      [`${6 - t.worst} из 6`, `индексов качества, по которым SPECTRA не в худшей трети из ${t.configs} вариантов методов`],
      [fmt(t.ari_model, 2), `точность (ARI) SPECTRA на синтетических данных в самом трудном для неё случае, у k-means ${fmt(t.ari_kmeans, 2)}`],
      [String(t.robust), "МО надёжно сменили тип за 2024 год, смена повторяется в половине и более перерасчётов"],
    ];
    document.getElementById("tiles").innerHTML = items.map((d) => `<div class="tile"><div class="num">${d[0]}</div><div class="cap">${d[1]}</div></div>`).join("");
  }

  function mapView() {
    const [x0, y0, x1, y1] = shapes.box;
    const pad = 150;
    const vx = x0 - pad, vy = y0 - pad, vw = x1 - x0 + 2 * pad, vh = y1 - y0 + 2 * pad;
    const svg = d3.select("#map").attr("viewBox", `${vx} ${vy} ${vw} ${vh}`);
    const g = svg.append("g");
    const ids = Object.keys(shapes.paths).map(Number);
    const area = (id) => (shapes.boxes[id][2] - shapes.boxes[id][0]) * (shapes.boxes[id][3] - shapes.boxes[id][1]);
    ids.sort((a, b) => area(b) - area(a));
    const paths = g.selectAll("path.mo").data(ids).join("path").attr("class", "mo").attr("d", (id) => shapes.paths[id]).attr("fill-rule", "evenodd").attr("data-id", (id) => id);
    const arcs = g.append("g");
    const pickPath = g.append("path").attr("class", "pick").style("display", "none");
    let scaleK = 1;
    const unit = vw / 1000;

    const zoom = d3.zoom().scaleExtent([1, 60]).translateExtent([[vx, vy], [vx + vw, vy + vh]])
      .filter((e) => (e.type === "wheel" ? e.ctrlKey || e.metaKey : !e.button))
      .on("zoom", (e) => {
        g.attr("transform", e.transform);
        scaleK = e.transform.k;
        arcs.selectAll("circle").attr("r", (d) => (d.main ? 5 : 3.6) * unit / scaleK);
        svg.style("touch-action", scaleK > 1.05 ? "none" : "pan-y");
      });
    svg.call(zoom).on("dblclick.zoom", null);
    d3.select("#zoom-in").on("click", () => svg.transition().duration(250).call(zoom.scaleBy, 1.8));
    d3.select("#zoom-out").on("click", () => svg.transition().duration(250).call(zoom.scaleBy, 1 / 1.8));
    d3.select("#zoom-reset").on("click", () => svg.transition().duration(350).call(zoom.transform, d3.zoomIdentity));

    const periodButtons = d3.select("#periods").selectAll("button").data(atlas.periods).join("button").text((d) => PERIODS[d] || d)
      .on("click", (e, d) => { state.period = atlas.periods.indexOf(d); update(); });
    d3.selectAll("#modes button").on("click", function () { state.mode = this.dataset.mode; update(); });

    function fill(id) {
      const r = records.get(id);
      if (!r) return NODATA;
      const p = state.period;
      if (state.mode === "type") return r.kinds[p] < 0 ? NODATA : COLORS[r.kinds[p]];
      if (state.mode === "confidence") {
        const c = r.conf[p];
        return c < 0 ? NODATA : CONF[c < 0.5 ? 0 : c < 0.7 ? 1 : c < 0.9 ? 2 : 3][0];
      }
      if (r.kinds[0] < 0 || r.kinds[last] < 0) return NODATA;
      if (r.kinds[0] === r.kinds[last]) return CHANGE.same;
      return r.flag > 0 ? CHANGE.reliable : CHANGE.weak;
    }
    const faded = (id) => state.filter !== null && (!records.get(id) || records.get(id).kinds[state.period] !== state.filter);

    function legend() {
      const counts = d3.rollup(atlas.mo, (v) => v.length, (r) => r[3][state.period]);
      d3.select("#legend").selectAll("button").data(order).join("button").attr("class", "chip")
        .classed("on", (t) => state.filter === t).classed("dim", (t) => state.filter !== null && state.filter !== t)
        .html((t) => `<i style="background:${COLORS[t]}"></i>${LEGEND[t]} <b>${counts.get(t) || 0}</b>`)
        .on("click", (e, t) => { state.filter = state.filter === t ? null : t; update(); });
      let items = [];
      if (state.mode === "confidence") items = CONF.map((c) => [c[0], "уверенность " + c[1]]);
      if (state.mode === "change") items = [[CHANGE.reliable, "надёжная смена типа"], [CHANGE.weak, "смена типа, ненадёжная"], [CHANGE.same, "тип не менялся"]];
      items.push([NODATA, "нет полных данных"]);
      d3.select("#scale").selectAll("span").data(items).join("span").html((d) => `<i style="background:${d[0]}"></i>${d[1]}`);
    }

    function update() {
      periodButtons.classed("on", (d, i) => i === state.period);
      d3.selectAll("#modes button").classed("on", function () { return this.dataset.mode === state.mode; });
      paths.style("fill", fill).classed("fade", faded);
      legend();
      views.forEach((v) => v.refresh && v.refresh());
    }

    function hoverText(r) {
      const t = r.kinds[state.period], c = r.conf[state.period];
      return `<div class="name">${dot(t)}${r.name}</div><div class="reg">${r.region}</div>` +
        `<div class="row"><span>${PERIODS[atlas.periods[state.period]]}</span><b>${t < 0 ? "нет данных" : atlas.short[t]}</b></div>` +
        (c >= 0 ? `<div class="row"><span>уверенность</span><b>${fmt(c, 2)}</b></div>` : "");
    }

    let current = null;
    svg.on("pointermove", (e) => {
      if (e.pointerType === "touch") return;
      const id = e.target.dataset && e.target.dataset.id ? +e.target.dataset.id : null;
      if (id !== current) {
        if (current !== null) paths.filter((d) => d === current).classed("hover", false);
        current = id;
        if (current !== null) paths.filter((d) => d === current).classed("hover", true);
      }
      if (id === null || !records.has(id)) { hideTip(); return; }
      showTip(hoverText(records.get(id)), e.clientX, e.clientY);
    }).on("pointerleave", () => {
      if (current !== null) paths.filter((d) => d === current).classed("hover", false);
      current = null;
      hideTip();
    }).on("click", (e) => {
      const id = e.target.dataset && e.target.dataset.id ? +e.target.dataset.id : null;
      if (id !== null && records.has(id)) select(id);
    });

    function curve(a, b) {
      const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
      const dx = b[0] - a[0], dy = b[1] - a[1];
      return `M${a[0]} ${a[1]}Q${mx - dy * 0.22} ${my + dx * 0.22} ${b[0]} ${b[1]}`;
    }

    function pick(id) {
      arcs.selectAll("*").remove();
      if (id === null || !shapes.paths[id]) { pickPath.style("display", "none"); return; }
      pickPath.attr("d", shapes.paths[id]).attr("fill-rule", "evenodd").style("display", null).raise();
      const i = netIndex.get(id), from = network.anchors[id];
      if (i === undefined || !from) return;
      const ends = [{ main: true, p: from, id }];
      network.neighbors[i].forEach(([j]) => {
        const to = network.anchors[nodes[j].id];
        if (!to) return;
        arcs.append("path").attr("class", "arc").attr("data-to", nodes[j].id).attr("d", curve(from, to));
        ends.push({ main: false, p: to, id: nodes[j].id });
      });
      arcs.selectAll("circle").data(ends).join("circle").attr("class", "end").attr("cx", (d) => d.p[0]).attr("cy", (d) => d.p[1])
        .attr("r", (d) => (d.main ? 5 : 3.6) * unit / scaleK).style("fill", (d) => (d.main ? "#F2F7EC" : null));
    }

    function mark(to) {
      arcs.selectAll("path.arc").classed("on", function () { return to !== null && +this.dataset.to === to; });
    }

    const input = document.getElementById("search");
    const list = document.getElementById("results");
    const pool = [...records.values()].map((r) => ({ r, key: norm(r.name) }));
    let shown = [], active = -1;
    function suggest() {
      const q = norm(input.value.trim());
      if (q.length < 2) { list.hidden = true; shown = []; return; }
      shown = pool.filter((p) => p.key.includes(q)).sort((a, b) => a.key.indexOf(q) - b.key.indexOf(q) || a.key.length - b.key.length).slice(0, 8);
      active = shown.length ? 0 : -1;
      list.innerHTML = shown.length ? "" : "<li>ничего не найдено</li>";
      shown.forEach((p, k) => {
        const li = document.createElement("li");
        li.innerHTML = `${p.r.name}<small>${p.r.region}</small>`;
        li.className = k === active ? "on" : "";
        li.addEventListener("pointerdown", (e) => { e.preventDefault(); choose(p.r); });
        list.appendChild(li);
      });
      list.hidden = false;
    }
    function choose(r) {
      input.value = r.name;
      list.hidden = true;
      svg.transition().duration(350).call(zoom.transform, d3.zoomIdentity);
      select(r.id);
    }
    input.addEventListener("input", suggest);
    input.addEventListener("focus", suggest);
    input.addEventListener("blur", () => { list.hidden = true; });
    input.addEventListener("keydown", (e) => {
      if (!shown.length) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        active = (active + (e.key === "ArrowDown" ? 1 : shown.length - 1)) % shown.length;
        [...list.children].forEach((li, k) => (li.className = k === active ? "on" : ""));
      } else if (e.key === "Enter") { e.preventDefault(); choose(shown[active].r); }
      else if (e.key === "Escape") list.hidden = true;
    });

    window.showType = (t) => {
      state.filter = t;
      state.mode = "type";
      update();
      document.getElementById("map-section").scrollIntoView({ behavior: "smooth" });
    };
    update();
    return { pick, mark, update };
  }

  function netView() {
    const canvas = document.getElementById("net");
    let W = 0, H = 0, tree = null, hover = null;
    const px = (n) => 14 + n.x * (W - 28), py = (n) => 14 + n.y * (H - 28);

    function size() {
      W = canvas.clientWidth;
      H = Math.max(360, Math.round(W * 0.7));
      const dpr = window.devicePixelRatio || 1;
      canvas.style.height = H + "px";
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      canvas.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
      tree = d3.quadtree(nodes.map((n, i) => [px(n), py(n), i]));
      draw();
    }

    function focus(i, ctx, strong) {
      const n = nodes[i];
      ctx.strokeStyle = strong ? "rgba(242,247,236,0.95)" : "rgba(181,224,58,0.9)";
      ctx.lineWidth = strong ? 1.6 : 1.2;
      ctx.beginPath();
      network.neighbors[i].forEach(([j]) => { ctx.moveTo(px(n), py(n)); ctx.lineTo(px(nodes[j]), py(nodes[j])); });
      ctx.stroke();
      network.neighbors[i].forEach(([j]) => {
        ctx.beginPath();
        ctx.arc(px(nodes[j]), py(nodes[j]), 4.2, 0, 2 * Math.PI);
        ctx.fillStyle = COLORS[nodes[j].type] || NODATA;
        ctx.fill();
        ctx.strokeStyle = "#F2F7EC";
        ctx.lineWidth = 1;
        ctx.stroke();
      });
      ctx.beginPath();
      ctx.arc(px(n), py(n), 7, 0, 2 * Math.PI);
      ctx.fillStyle = COLORS[n.type] || NODATA;
      ctx.fill();
      ctx.strokeStyle = "#F2F7EC";
      ctx.lineWidth = 2.2;
      ctx.stroke();
    }

    function draw() {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(181,224,58,0.07)";
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      network.edges.forEach(([i, j]) => { ctx.moveTo(px(nodes[i]), py(nodes[i])); ctx.lineTo(px(nodes[j]), py(nodes[j])); });
      ctx.stroke();
      nodes.forEach((n) => {
        const off = state.filter !== null && n.type !== state.filter;
        ctx.globalAlpha = off ? 0.12 : 0.95;
        ctx.fillStyle = n.type >= 0 ? COLORS[n.type] : NODATA;
        ctx.beginPath();
        ctx.arc(px(n), py(n), 2.6, 0, 2 * Math.PI);
        ctx.fill();
      });
      ctx.globalAlpha = 1;
      const picked = state.pick !== null ? netIndex.get(state.pick) : undefined;
      if (picked !== undefined) focus(picked, ctx, true);
      if (hover !== null && hover !== picked) focus(hover, ctx, false);
    }

    canvas.addEventListener("pointermove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const found = tree.find(e.clientX - rect.left, e.clientY - rect.top, 12);
      const next = found ? found[2] : null;
      if (next !== hover) { hover = next; draw(); }
      if (next === null) { hideTip(); return; }
      const n = nodes[next], r = records.get(n.id);
      showTip(`<div class="name">${dot(n.type)}${r ? r.name : n.id}</div><div class="reg">${r ? r.region : ""}</div><div class="row"><span>тип</span><b>${n.type >= 0 ? atlas.short[n.type] : "нет данных"}</b></div>`, e.clientX, e.clientY);
    });
    canvas.addEventListener("pointerleave", () => { hover = null; draw(); hideTip(); });
    canvas.addEventListener("click", () => { if (hover !== null) select(nodes[hover].id); });

    const side = document.getElementById("net-side");
    const s = network.stats;
    side.innerHTML = `<div class="legend-list">${order.map((t) => `<span>${dot(t)}${LEGEND[t]}</span>`).join("")}</div>` +
      `<div class="fact"><b>${s.nodes} МО, ${d3.format(",")(s.edges).replace(/,/g, " ")} ${plural(s.edges, ["ребро", "ребра", "рёбер"])}</b>одна компонента, у каждого МО от ${s.degree_min} до ${s.degree_max} соседей</div>` +
      `<div class="fact"><b>${fmt(s.clustering, 2)} против ${fmt(s.clustering_random, 3)}</b>коэффициент кластеризации сети и случайного графа с теми же степенями</div>` +
      `<div class="fact"><b>около ${s.median_km} км</b>типичное расстояние до ${s.neighbors} МО с самой похожей структурой трат, в своём регионе только ${pct(s.own_region)} из них</div>` +
      `<div class="takeaway">поэтому сообщества сети это типы экономик, разбросанные по всей стране, а не регионы</div>`;

    size();
    onResize(canvas, size);
    return { pick: () => draw(), refresh: () => draw() };
  }

  function passport(id) {
    const root = document.getElementById("passport");
    if (id === null || !records.has(id)) {
      root.innerHTML = `<h3>Карточка МО</h3><p class="empty">выберите муниципальное образование на карте, в сети или через поиск</p>`;
      return;
    }
    const r = records.get(id);
    const t = r.kinds[last];
    const p = extra.passport[id];
    const typical = t >= 0 ? extra.typical[t] : null;
    const hist = atlas.periods.map((d, k) => `<div style="background:${r.kinds[k] >= 0 ? COLORS[r.kinds[k]] : NODATA}" title="${PERIODS[d]}"></div>`).join("") +
      atlas.periods.map((d) => `<span>${STEPS[d]}</span>`).join("");
    const t0 = r.kinds[0];
    let change = "нет полных данных за оба года";
    if (t0 >= 0 && t >= 0) {
      change = t0 === t ? "тип за 2024 год не менялся"
        : `сменил тип: ${atlas.short[t0]} -> ${atlas.short[t]}, ` + (r.flag > 0 ? `<span class="accent">надёжно, в ${pct(r.flag)} перерасчётов</span>` : "ненадёжно, это пограничный МО");
    }
    let rows = "";
    if (p) {
      rows = `<div class="label">МО и медиана его типа в 2024 году</div><table>` +
        `<tr><td>население</td><td>${d3.format(",")(p[0]).replace(/,/g, " ")}</td><td></td></tr>` +
        `<tr><td>траты на жителя к медиане МО</td><td>${fmt(p[1], 2)}</td><td>${typical ? fmt(typical[0], 2) : ""}</td></tr>` +
        (p[2] !== null ? `<tr><td>зарплата к медиане МО</td><td>${fmt(p[2], 2)}</td><td>${typical ? fmt(typical[1], 2) : ""}</td></tr>` : "") +
        `<tr><td>доля маркетплейсов в тратах</td><td>${pct(p[3], 1)}</td><td>${typical ? pct(typical[2], 1) : ""}</td></tr>` +
        (p[4] !== null ? `<tr><td>траты против прогноза по местной экономике</td><td>${signed(p[4] * 100)}</td><td></td></tr>` : "") +
        `</table>`;
    }
    const i = netIndex.get(id);
    let twins = "";
    if (i !== undefined) {
      const list = network.neighbors[i];
      const kms = list.map((d) => d[1]).filter((v) => v >= 0).sort((a, b) => a - b);
      const median = kms.length ? kms[Math.floor(kms.length / 2)] : null;
      const own = list.filter(([j]) => records.get(nodes[j].id) && records.get(nodes[j].id).region === r.region).length;
      twins = `<div class="label">10 МО с самой похожей структурой трат</div>` +
        `<div class="reg">в своём регионе ${own} из 10${median !== null ? `, медиана расстояния ${median} км` : ""}</div>` +
        `<ul class="twins">${list.map(([j, km]) => {
          const n = nodes[j], q = records.get(n.id);
          return `<li data-id="${n.id}" title="${q ? q.name + ", " + q.region : ""}">${dot(n.type)}<span class="nm">${q ? shortName(q.name) : n.id} <small>${q ? shortRegion(q.region) : ""}</small></span><span class="km">${km >= 0 ? km + " км" : ""}</span></li>`;
        }).join("")}</ul>`;
    }
    root.innerHTML = `<h3>${r.name}</h3><div class="reg" style="margin-bottom:8px">${r.region}</div>` +
      `<div class="now">${dot(t)}${t >= 0 ? atlas.names[t] : "нет полных данных о тратах"}</div>` +
      (r.conf[last] >= 0 ? `<div class="reg" style="margin:4px 0 0">уверенность в 2024 году ${fmt(r.conf[last], 2)}, ${change}</div>` : `<div class="reg" style="margin:4px 0 0">${change}</div>`) +
      twins + `<div class="label">тип по периодам</div><div class="hist">${hist}</div>` + rows;
    root.querySelectorAll(".twins li").forEach((li) => {
      li.addEventListener("pointerenter", () => mapApi.mark(+li.dataset.id));
      li.addEventListener("pointerleave", () => mapApi.mark(null));
      li.addEventListener("click", () => select(+li.dataset.id));
    });
  }

  function cards() {
    document.getElementById("cards").innerHTML = atlas.cards.map((c) => {
      const e = c.external;
      return `<article class="card" style="--c:${COLORS[c.id]}">
        <h4>${c.name}</h4>
        <div class="size">${c.count} МО, ${pct(c.share)} населения (${fmt(c.population, c.population >= 20 ? 0 : 1)} млн)</div>
        <ul>${c.traits.map((t) => `<li class="${t.up ? "" : "down"}">${t.text}</li>`).join("")}</ul>
        <div class="ext">
          <div><span>розница на жителя в год</span><b>${fmt(e.retail / 1000, 0)} тыс. руб.</b></div>
          <div><span>ввод жилья на 1000 жителей</span><b>${fmt(e.housing, 0)} кв. м</b></div>
          <div><span>инвестиции на жителя, 2023</span><b>${fmt(e.investment / 1000, 0)} тыс. руб.</b></div>
        </div>
        <button class="go" data-type="${c.id}">показать на карте</button>
      </article>`;
    }).join("");
    document.querySelectorAll("#cards .go").forEach((b) => b.addEventListener("click", () => window.showType(+b.dataset.type)));
  }

  function flows() {
    const m = atlas.matrix, solid = atlas.solid;
    const share = atlas.changed / atlas.total;
    const items = [
      [pct(share, 1), `МО сменили тип, ${atlas.changed} из ${atlas.total}`],
      [String(atlas.robust), "смен надёжны, они повторяются не меньше чем в половине из 300 перерасчётов"],
      ["31% и 0.4%", "МО сменили тип при уверенности ниже 0.7 и от 0.9, то есть смены идут на границах типов"],
    ];
    document.getElementById("flow-tiles").innerHTML = items.map((d) => `<div class="tile"><div class="num">${d[0]}</div><div class="cap">${d[1]}</div></div>`).join("");
    const pairs = [];
    m.forEach((row, a) => row.forEach((v, b) => { if (a !== b && v > 0) pairs.push([a, b, v]); }));
    pairs.sort((x, y) => y[2] - x[2]);
    document.getElementById("flows").innerHTML = pairs.slice(0, 5).map(([a, b, v]) =>
      `<div class="flow-row"><span>${dot(a)}${atlas.short[a].toLowerCase()}</span><span class="arrow">-></span><span>${dot(b)}${atlas.short[b].toLowerCase()}</span><b>${v}</b></div>`).join("");

    const root = d3.select("#matrix");
    const scale = d3.scaleSqrt().domain([0, d3.max(pairs, (p) => p[2])]).range([0.06, 0.85]);
    function draw() {
      const W = root.node().clientWidth || 560;
      const left = Math.min(185, W * 0.36), top = 26, cell = (W - left - 4) / 6, ch = Math.min(52, cell * 0.8);
      const H = top + ch * 6 + 4;
      root.selectAll("*").remove();
      const svg = root.append("svg").attr("viewBox", `0 0 ${W} ${H}`);
      order.forEach((t, k) => {
        svg.append("circle").attr("cx", left + k * cell + cell / 2).attr("cy", 12).attr("r", 6).attr("fill", COLORS[t]);
        svg.append("text").attr("x", left - 10).attr("y", top + k * ch + ch / 2 + 4).attr("text-anchor", "end").attr("class", "ink").style("font-size", "12.5px").text(atlas.short[t]);
      });
      order.forEach((a, row) => order.forEach((b, col) => {
        const v = m[a][b], s = solid[a][b], same = a === b;
        const cell_g = svg.append("g").attr("transform", `translate(${left + col * cell},${top + row * ch})`);
        cell_g.append("rect").attr("x", 1.5).attr("y", 1.5).attr("width", cell - 3).attr("height", ch - 3).attr("rx", 6)
          .attr("fill", same ? COLORS[a] : "#B5E03A").attr("fill-opacity", same ? 0.28 : v ? scale(v) : 0.03);
        cell_g.append("text").attr("x", cell / 2).attr("y", ch / 2 + 5).attr("text-anchor", "middle").attr("class", "ink")
          .style("font-size", "14px").style("font-weight", same || v >= 10 ? 700 : 400).style("fill", !same && v && scale(v) > 0.5 ? "#0E1A12" : null).text(v);
        cell_g.append("rect").attr("width", cell).attr("height", ch).attr("fill", "transparent")
          .on("pointermove", (e) => showTip(same
            ? `<div class="name">${dot(a)}${atlas.names[a]}</div><div class="reg">остались в типе ${v} из ${d3.sum(m[a])}</div>`
            : `<div class="name">${dot(a)}${atlas.short[a]} -> ${dot(b)}${atlas.short[b]}</div><div class="reg">${v} МО, из них надёжно ${s}</div>`, e.clientX, e.clientY))
          .on("pointerleave", hideTip);
      }));
    }
    draw();
    onResize(root.node(), draw);
    const moves = atlas.moves;
    document.getElementById("moves-title").textContent = `все ${moves.length} надёжных смен типа`;
    document.getElementById("moves").innerHTML = `<div class="table-wrap"><table class="data"><thead><tr><th>МО</th><th>из типа</th><th>в тип</th><th>доля перерасчётов со сменой</th></tr></thead><tbody>` +
      moves.map((r) => `<tr><td>${r[1]}<br><small style="color:var(--muted)">${r[2]}</small></td><td>${dot(r[3])} ${atlas.short[r[3]]}</td><td>${dot(r[4])} ${atlas.short[r[4]]}</td><td>${pct(r[5])}</td></tr>`).join("") +
      `</tbody></table></div>`;
  }

  function changes() {
    const cols = extra.changes.columns;
    const head = cols.map((c) => {
      const [name, u] = c.split(", ");
      return `<th class="${name === "маркетплейсы" ? "hl" : ""}">${name}<br><small>${u}</small></th>`;
    }).join("");
    const body = order.map((t) => {
      const row = extra.changes.rows[String(t)];
      return `<tr><td>${dot(t)} ${LEGEND[t]}</td>` + row.map((v, k) => {
        const unit = cols[k].endsWith("%") ? "%" : "";
        const cls = v >= 1 ? "up" : v <= -1 ? "down" : "small";
        return `<td class="${cls}">${(v > 0 ? "+" : "") + v.toFixed(1)}${unit}</td>`;
      }).join("") + "</tr>";
    }).join("");
    document.getElementById("changes").innerHTML = `<table class="data"><thead><tr><th>тип</th>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  function market() {
    const root = d3.select("#market");
    const d = atlas.market, n = d.dates.length;
    d3.select("#market-key").selectAll("span").data(order).join("span").html((t) => `<i style="background:${COLORS[t]}"></i>${atlas.short[t]}`);
    function draw() {
      const W = root.node().clientWidth || 720, wide = W >= 640, H = wide ? 360 : 300;
      const m = { l: 44, r: wide ? 205 : 12, t: 12, b: 30 };
      root.selectAll("*").remove();
      const svg = root.append("svg").attr("viewBox", `0 0 ${W} ${H}`);
      const all = order.flatMap((t) => d.series[t]);
      const x = d3.scaleLinear().domain([0, n - 1]).range([m.l, W - m.r]);
      const y = d3.scaleLinear().domain([Math.floor(d3.min(all) * 50) / 50, Math.ceil(d3.max(all) * 50) / 50]).range([H - m.b, m.t]);
      const grid = svg.append("g").attr("class", "grid");
      y.ticks(5).forEach((v) => {
        grid.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(v)).attr("y2", y(v));
        svg.append("text").attr("x", m.l - 8).attr("y", y(v) + 4).attr("text-anchor", "end").text(pct(v));
      });
      d.dates.forEach((s, i) => {
        const mo = +s.slice(5) - 1;
        if (mo % (wide ? 3 : 6) !== 0) return;
        svg.append("text").attr("x", x(i)).attr("y", H - 8).attr("text-anchor", "middle").text(MONTHS[mo] + (mo === 0 ? " " + s.slice(0, 4) : ""));
      });
      const line = d3.line().x((v, i) => x(i)).y((v) => y(v));
      order.forEach((t) => svg.append("path").attr("d", line(d.series[t])).attr("fill", "none").attr("stroke", COLORS[t]).attr("stroke-width", 2.2).attr("stroke-linejoin", "round"));
      if (wide) {
        const ends = order.map((t) => ({ t, y: y(d.series[t][n - 1]) })).sort((a, b) => a.y - b.y);
        for (let k = 1; k < ends.length; k++) ends[k].y = Math.max(ends[k].y, ends[k - 1].y + 15);
        ends.forEach((e) => svg.append("text").attr("x", W - m.r + 8).attr("y", e.y + 4).attr("class", "ink").style("font-size", "12.5px").text(atlas.short[e.t]));
      }
      const cross = svg.append("g").style("display", "none");
      cross.append("line").attr("y1", m.t).attr("y2", H - m.b).attr("stroke", "#93A88E");
      const marks = order.map((t) => cross.append("circle").attr("r", 4).attr("fill", COLORS[t]).attr("stroke", "#070B08").attr("stroke-width", 1.5));
      svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b).attr("fill", "transparent")
        .on("pointermove", (e) => {
          const i = Math.max(0, Math.min(n - 1, Math.round(x.invert(d3.pointer(e)[0]))));
          cross.style("display", null).select("line").attr("x1", x(i)).attr("x2", x(i));
          order.forEach((t, k) => marks[k].attr("cx", x(i)).attr("cy", y(d.series[t][i])));
          const rows = [...order].sort((a, b) => d.series[b][i] - d.series[a][i]).map((t) => `<div class="row"><span>${dot(t)}${atlas.short[t]}</span><b>${pct(d.series[t][i], 1)}</b></div>`).join("");
          showTip(`<div class="name">${MONTHS[+d.dates[i].slice(5) - 1]} ${d.dates[i].slice(0, 4)}</div>${rows}`, e.clientX, e.clientY);
        }).on("pointerleave", () => { cross.style("display", "none"); hideTip(); });
    }
    draw();
    onResize(root.node(), draw);
  }

  function model() {
    const family = (p) => (p.method === "refined_spectral" ? (p.alpha === 0.3 ? "ours" : "ours-alt")
      : ["kmeans", "ward", "gmm"].includes(p.method) ? "features" : ["spectral", "louvain", "leiden"].includes(p.method) ? "graph" : "joint");
    const style = { features: ["#C9D6C3", "только признаки"], graph: ["#7FA7C9", "только граф"], joint: ["#7E9A78", "признаки и граф"], "ours-alt": ["#5E7A2A", "SPECTRA при других α"], ours: ["#B5E03A", "SPECTRA, α = 0.3"] };
    d3.select("#scatter-key").selectAll("span").data(Object.keys(style)).join("span").html((k) => `<i style="background:${style[k][0]}"></i>${style[k][1]}`);
    const root = d3.select("#scatter");
    const label = (p) => (family(p) === "ours" ? "SPECTRA" : p.method === "kmeans" ? "k-means" : p.method === "spectral" ? "спектральная по графу" : null);
    function scatter() {
      const W = root.node().clientWidth || 520, H = Math.max(300, Math.min(420, W * 0.78));
      const m = { l: 46, r: 96, t: 12, b: 44 };
      root.selectAll("*").remove();
      const svg = root.append("svg").attr("viewBox", `0 0 ${W} ${H}`);
      const pts = atlas.methods;
      const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p.sw)).nice().range([m.l, W - m.r]);
      const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p.mq)).nice().range([H - m.b, m.t]);
      const grid = svg.append("g").attr("class", "grid");
      y.ticks(5).forEach((v) => {
        grid.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(v)).attr("y2", y(v));
        svg.append("text").attr("x", m.l - 8).attr("y", y(v) + 4).attr("text-anchor", "end").text(fmt(v, 1));
      });
      x.ticks(6).forEach((v) => svg.append("text").attr("x", x(v)).attr("y", H - m.b + 16).attr("text-anchor", "middle").text(fmt(v, 2)));
      svg.append("text").attr("class", "ink").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").text("SW, качество в пространстве признаков");
      svg.append("text").attr("class", "ink").attr("transform", `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").text("MQ, модулярность на графе трат");
      const ordered = [...pts].sort((a, b) => (family(a) === "ours") - (family(b) === "ours"));
      svg.selectAll("circle").data(ordered).join("circle").attr("cx", (p) => x(p.sw)).attr("cy", (p) => y(p.mq))
        .attr("r", (p) => (family(p) === "ours" ? 8 : 4.5)).attr("fill", (p) => style[family(p)][0]).attr("stroke", "#070B08").attr("stroke-width", 1.2)
        .on("pointermove", (e, p) => showTip(`<div class="name">${p.method}${p.alpha === null ? "" : ", α = " + p.alpha}</div><div class="reg">SW ${fmt(p.sw, 3)}, MQ ${fmt(p.mq, 3)}</div>`, e.clientX, e.clientY))
        .on("pointerleave", hideTip);
      ordered.filter(label).forEach((p) => {
        const left = p.method === "spectral";
        svg.append("text").attr("x", x(p.sw) + (left ? -10 : 12)).attr("y", y(p.mq) + 4).attr("text-anchor", left ? "end" : "start")
          .attr("class", "ink").style("font-weight", family(p) === "ours" ? 700 : 500).style("fill", family(p) === "ours" ? "#B5E03A" : null).style("font-size", "13px").text(label(p));
      });
    }
    const bars = d3.select("#bars");
    function barChart() {
      const W = bars.node().clientWidth || 440;
      const rows = [...atlas.ari].sort((a, b) => b.ari - a.ari);
      const rowH = 34, m = { l: W < 420 ? 120 : 170, r: 44, t: 6, b: 10 };
      const H = m.t + rows.length * rowH + m.b;
      bars.selectAll("*").remove();
      const svg = bars.append("svg").attr("viewBox", `0 0 ${W} ${H}`);
      const x = d3.scaleLinear().domain([0, 0.6]).range([m.l, W - m.r]);
      rows.forEach((r, k) => {
        const cy = m.t + k * rowH + rowH / 2;
        svg.append("text").attr("x", m.l - 10).attr("y", cy + 4).attr("text-anchor", "end").attr("class", "ink").style("font-weight", r.model ? 700 : 400).style("fill", r.model ? "#B5E03A" : null).style("font-size", "13.5px").text(METHOD_NAMES[r.name] || r.name);
        svg.append("rect").attr("x", m.l).attr("y", cy - 9).attr("width", x(r.ari) - m.l).attr("height", 18).attr("rx", 3).attr("fill", r.model ? "#B5E03A" : "#71866F");
        svg.append("text").attr("x", x(r.ari) + 6).attr("y", cy + 4).attr("class", "ink").style("font-weight", r.model ? 700 : 400).style("fill", r.model ? "#B5E03A" : null).style("font-size", "13.5px").text(fmt(r.ari, 2));
      });
    }
    scatter();
    barChart();
    onResize(root.node(), scatter);
    onResize(bars.node(), barChart);
  }
}

function onResize(el, fn) {
  let width = el.clientWidth, timer = null;
  new ResizeObserver(() => {
    if (Math.abs(el.clientWidth - width) < 2) return;
    width = el.clientWidth;
    clearTimeout(timer);
    timer = setTimeout(fn, 120);
  }).observe(el);
}

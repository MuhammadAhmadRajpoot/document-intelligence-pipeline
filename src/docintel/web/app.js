"use strict";

const state = { docs: [], current: -1, page: 0 };
const $ = (sel) => document.querySelector(sel);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const LOW = 0.6;

function fmtValue(v) {
  if (v === null || v === undefined || v === "") return '<span class="muted">-</span>';
  if (typeof v === "number") return v.toLocaleString(undefined, { maximumFractionDigits: 4 });
  return esc(v);
}

function confCell(ev) {
  if (!ev) return '<span class="muted">n/a</span>';
  const pct = Math.round(ev.confidence * 100);
  const cls = ev.confidence < LOW ? "low" : ev.confidence < 0.85 ? "mid" : "";
  const ground = ev.grounded ? "" : ' <span class="tag" title="Value not found verbatim in the document">inferred</span>';
  return `<div class="conf"><div class="bar ${cls}"><span style="width:${pct}%"></span></div>${pct}%${ground}</div>`;
}

function statusOf(result) {
  if (result.error) return "error";
  return result.needs_review ? "review" : "ok";
}

function renderHistory() {
  $("#history").innerHTML = state.docs.map((d, i) => `
    <button class="chip ${i === state.current ? "active" : ""}" data-index="${i}">
      <span class="dot ${statusOf(d.result)}"></span>${esc(d.name)}
    </button>`).join("");
  document.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
    select(Number(chip.dataset.index));
  }));
  const any = state.docs.length > 0;
  document.querySelectorAll("[data-export]").forEach((b) => { b.disabled = !any; });
}

function renderPreview(doc) {
  const page = doc.pages[state.page];
  const hint = $("#drop-hint");
  hint.classList.toggle("hidden", Boolean(page));
  $("#page-meta").textContent = page
    ? `page ${page.index + 1}/${doc.pages.length} | ${page.source === "ocr" ? "OCR" : "PDF text layer"}`
      + ` | mean conf ${(page.mean_confidence * 100).toFixed(0)}%`
      + (page.preprocessing.length ? ` | ${page.preprocessing.join(", ")}` : "")
    : "";
  if (!page || !page.preview) { $("#preview").innerHTML = ""; return; }
  const boxes = Object.values(doc.result.field_confidence || {})
    .filter((ev) => ev.bbox && ev.page === page.index)
    .map((ev) => {
      const [x0, y0, x1, y1] = ev.bbox;
      const style = `left:${(x0 / page.width) * 100}%;top:${(y0 / page.height) * 100}%;`
        + `width:${((x1 - x0) / page.width) * 100}%;height:${((y1 - y0) / page.height) * 100}%`;
      return `<div class="box ${ev.confidence < LOW ? "low" : ""}" data-field="${esc(ev.field)}"
        style="${style}" title="${esc(ev.field)}: ${esc(ev.value)}"></div>`;
    }).join("");
  $("#preview").innerHTML = `<img src="${page.preview}" alt="document page preview" />${boxes}`;
  $("#page-tabs").innerHTML = doc.pages.length > 1 ? doc.pages.map((p, i) =>
    `<button class="btn ${i === state.page ? "primary" : ""}" data-page="${i}">Page ${i + 1}</button>`).join("") : "";
  document.querySelectorAll("[data-page]").forEach((b) => b.addEventListener("click", () => {
    state.page = Number(b.dataset.page); renderPreview(doc);
  }));
}

function renderStatus(result) {
  const s = statusOf(result);
  const title = { ok: "Validated - no review needed", review: "Needs human review", error: "Extraction failed" }[s];
  const reasons = result.review_reasons.length
    ? `<ul>${result.review_reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : "";
  $("#status").className = `status ${s}`;
  $("#status").innerHTML = `${title}${reasons}`;
}

function renderSummary(result) {
  const d = result.data || {};
  const stats = [
    ["Type", `${result.document_type} (${Math.round(result.classification_confidence * 100)}%)`],
    ["Total", d.total != null ? `${fmtValue(d.total)} ${esc(d.currency || "")}` : "-"],
    ["Checks", result.is_valid ? "passed" : "failed"],
    ["Time", `${((result.timings_ms.total_ms || 0) / 1000).toFixed(1)} s`],
  ];
  $("#summary").innerHTML = stats.map(([k, v]) =>
    `<div class="stat"><div class="label">${k}</div><div class="value">${v}</div></div>`).join("");
}

function renderFields(result) {
  const d = result.data || {};
  const ev = result.field_confidence || {};
  let rows;
  if (Array.isArray(d.fields)) {
    rows = [["title", d.title], ["summary", d.summary], ...d.fields.map((f) => [f.key, f.value])];
  } else {
    rows = Object.entries(d).filter(([k]) => k !== "line_items");
  }
  $("#tab-fields").innerHTML = `<table><thead><tr><th>Field</th><th>Value</th><th>Confidence</th></tr></thead><tbody>
    ${rows.map(([k, v]) => `<tr class="field-row" data-field="${esc(k)}">
      <td class="key">${esc(k.replace(/_/g, " "))}</td><td>${fmtValue(v)}</td><td>${v == null ? "" : confCell(ev[k])}</td></tr>`).join("")}
  </tbody></table>`;
  document.querySelectorAll(".field-row").forEach((row) => {
    const box = () => document.querySelector(`.box[data-field="${CSS.escape(row.dataset.field)}"]`);
    row.addEventListener("mouseenter", () => box()?.classList.add("hot"));
    row.addEventListener("mouseleave", () => box()?.classList.remove("hot"));
  });
}

function renderItems(result) {
  const items = (result.data || {}).line_items || [];
  $("#tab-items").innerHTML = items.length ? `<table><thead><tr><th>#</th><th>Description</th>
    <th class="num">Qty</th><th class="num">Unit price</th><th class="num">Amount</th></tr></thead><tbody>
    ${items.map((it, i) => `<tr><td>${i + 1}</td><td>${esc(it.description)}</td><td class="num">${fmtValue(it.quantity)}</td>
      <td class="num">${fmtValue(it.unit_price)}</td><td class="num">${fmtValue(it.amount)}</td></tr>`).join("")}
  </tbody></table>` : '<p class="muted">No line items.</p>';
}

function renderChecks(result) {
  const issues = result.issues || [];
  const attempts = result.attempts.map((a) => `<div class="check"><span class="tag">attempt ${a.attempt + 1}</span>
    <span>${a.schema_error ? `schema/LLM error: ${esc(a.schema_error)}`
      : `${a.issues.filter((i) => i.severity === "error").length} rule error(s)`}</span></div>`).join("");
  $("#tab-checks").innerHTML = (issues.length ? issues.map((i) => `<div class="check">
      <span class="tag ${i.severity === "error" ? "err" : "warn"}">${esc(i.severity)}</span>
      <div><strong>${esc(i.code)}</strong><div class="muted">${esc(i.message)}</div></div></div>`).join("")
    : '<div class="check"><span class="tag ok">pass</span><span>All business rules passed.</span></div>')
    + `<h3 class="muted" style="margin-top:16px">Extraction attempts (validate &amp; repair loop)</h3>${attempts}`;
}

function select(index) {
  state.current = index; state.page = 0;
  const doc = state.docs[index];
  $("#doc-name").textContent = doc.name;
  renderHistory(); renderPreview(doc); renderStatus(doc.result); renderSummary(doc.result);
  renderFields(doc.result); renderItems(doc.result); renderChecks(doc.result);
  $("#ocr-text").textContent = doc.pages.map((p) => `--- page ${p.index + 1} ---\n${p.text}`).join("\n\n");
  $("#json-text").textContent = JSON.stringify(doc.result.data, null, 2);
}

function addDoc(name, payload) {
  state.docs.push({ name, result: payload.result, pages: payload.pages });
  select(state.docs.length - 1);
}

async function upload(files) {
  for (const file of files) {
    $("#busy-text").textContent = `Processing ${file.name}... (OCR on CPU can take a while)`;
    $("#busy").classList.remove("hidden");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/extract", { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || res.statusText);
      addDoc(file.name, body);
    } catch (err) {
      $("#status").className = "status error";
      $("#status").textContent = `${file.name}: ${err.message}`;
    } finally {
      $("#busy").classList.add("hidden");
    }
  }
}

function download(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function exportAs(fmt) {
  const results = state.docs.map((d) => d.result);
  if (fmt === "json") {
    download(new Blob([JSON.stringify(results, null, 2)], { type: "application/json" }), "docintel_results.json");
    return;
  }
  const res = await fetch(`/export/${fmt}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(results),
  });
  if (res.ok) download(await res.blob(), `docintel_results.${fmt}`);
}

function wire() {
  $("#file-input").addEventListener("change", (e) => { upload([...e.target.files]); e.target.value = ""; });
  const zone = $("#dropzone");
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragging"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault(); zone.classList.remove("dragging"); upload([...e.dataTransfer.files]);
  });
  document.querySelectorAll("[data-export]").forEach((b) => b.addEventListener("click", () => exportAs(b.dataset.export)));
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".tab-body").forEach((b) => b.classList.toggle("hidden", b.id !== `tab-${tab.dataset.tab}`));
  }));
}

async function loadExample() {
  try {
    const res = await fetch("/example");
    if (!res.ok) return;
    const example = await res.json();
    addDoc(`${example.name} (example)`, example.payload);
  } catch { /* the example is optional */ }
}

wire();
loadExample();

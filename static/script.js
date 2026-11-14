let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";
let waitingForApproval = false;

const AGENT_LABELS = {
  flight_agent: "✈ Flight planning",
  hotel_agent: "⌂ Stay finder",
  weather_agent: "☼ Weather guide",
  budget_agent: "◈ Budget planner",
  itinerary_agent: "✳ Itinerary designer"
};

const byId = (id) => document.getElementById(id);

function setPrompt(text) {
  byId("userInput").value = text;
  updateCharCount();
  byId("userInput").focus();
  byId("planner").scrollIntoView({ behavior: "smooth", block: "start" });
}

function updateCharCount() {
  const input = byId("userInput");
  const count = byId("charCount");
  if (input && count) count.textContent = `${input.value.length} / ${input.maxLength}`;
}

function setLoading(isLoading, mode = "draft") {
  const sendBtn = byId("sendBtn");
  const btnText = byId("btnText");
  const btnLoader = byId("btnLoader");
  const approveBtn = byId("approveBtn");
  const reviseBtn = byId("reviseBtn");
  sendBtn.disabled = isLoading;
  approveBtn.disabled = isLoading;
  reviseBtn.disabled = isLoading;
  if (isLoading && mode === "draft") {
    btnText.classList.add("hidden");
    btnLoader.classList.remove("hidden");
  } else {
    btnText.classList.remove("hidden");
    btnLoader.classList.add("hidden");
  }
  if (mode === "draft") btnText.innerHTML = isLoading ? "Creating your plan…" : 'Create my travel plan <span aria-hidden="true">↗</span>';
}

function showError(message) {
  const errorBox = byId("errorBox");
  errorBox.textContent = message || "Something went wrong. Please try again.";
  errorBox.classList.remove("hidden");
  errorBox.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideError() {
  const errorBox = byId("errorBox");
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

const SECTION_ICONS = {
  "trip summary": "✧",
  "flight information": "↗",
  "hotel suggestions": "⌂",
  "weather information": "☼",
  "day-by-day itinerary": "✳",
  "estimated budget": "◈",
  "final recommendations": "✓"
};

function decorateSectionHeadings(container) {
  container.querySelectorAll("h2").forEach((h2) => {
    if (h2.dataset.iconAdded) return;
    const raw = h2.textContent.trim();
    const clean = raw.replace(/^\d+[\.\)]\s*/, "").trim().toLowerCase();
    const icon = SECTION_ICONS[clean];
    if (icon) {
      h2.dataset.iconAdded = "1";
      h2.innerHTML = `<span class="section-icon" aria-hidden="true">${icon}</span>${h2.innerHTML}`;
    }
  });
}



// 1. BUDGET SECTION FORMATTING (Flash Cards + 2 Columns)
function formatBudgetSection(container) {
  const h2s = Array.from(container.querySelectorAll("h2"));
  const budgetH2 = h2s.find(h2 => h2.textContent.toLowerCase().includes("estimated budget"));
  
  if (!budgetH2) return;

  let el = budgetH2.nextElementSibling;
  let totalP = null, shareP = null;

  // Loop through elements until next H2
  while (el && el.tagName !== "H2") {
    if (el.tagName === "P" && el.textContent.includes("Estimated trip total")) {
      totalP = el;
    } else if (el.tagName === "P" && el.textContent.includes("Estimated per-person share")) {
      shareP = el;
    } else if (el.tagName === "TABLE") {
      el.classList.add("budget-table");
      el.querySelectorAll("tr").forEach(row => {
        const cells = row.querySelectorAll("th, td");
        if (cells.length === 4) cells[3].classList.add("budget-basis-cell");
      });
    } 
    // FIX: Backend sends **bold text** which is <p><strong>, so we check for that
    else if (el.tagName === "H3" || (el.tagName === "P" && el.querySelector("strong"))) {
      const text = el.textContent.toUpperCase();
      if (text.includes("TRIP ASSUMPTIONS") || text.includes("ESTIMATED COST BREAKDOWN") || text.includes("TOTAL ESTIMATED BUDGET") || text.includes("BUDGET RISKS") || text.includes("MONEY-SAVING SUGGESTIONS") || text.includes("IMPORTANT DISCLAIMER")) {
        el.classList.add("budget-subheading");
      }
    } 
    else if (el.tagName === "UL" || el.tagName === "OL") {
      el.classList.add("budget-list-grid");
    }
    el = el.nextElementSibling;
  }

  // Create Flash Cards for Totals
  if (totalP && shareP) {
    const wrapper = document.createElement("div");
    wrapper.className = "budget-cards-wrapper";
    totalP.parentNode.insertBefore(wrapper, totalP);
    
    totalP.classList.add("budget-card", "budget-card-main");
    shareP.classList.add("budget-card", "budget-card-sub");
    
    wrapper.appendChild(totalP);
    wrapper.appendChild(shareP);
  }
}

// 2. ITINERARY TABLE FORMATTING (Converting text to bullets)
function formatItineraryTable(container) {
  const tables = container.querySelectorAll("table");
  tables.forEach((table) => {
    const text = table.textContent || "";
    // Check if this is the itinerary table
    if (text.includes("Morning") && text.includes("Afternoon") && text.includes("Evening")) {
      table.querySelectorAll("td").forEach((cell, index) => {
        if (index % 4 === 0) return; // Skip the Day column
        
        const parts = cell.innerHTML.split(/<br\s*\/?>/i);
        let newHtml = "";
        parts.forEach(part => {
          if (part.trim()) {
            newHtml += `<div class="itinerary-item">${part.trim()}</div>`;
          }
        });
        if (newHtml) cell.innerHTML = newHtml;
      });
    }
  });
}


function forceBudgetTable(container) {
  const h2s = Array.from(container.querySelectorAll("h2"));
  const budgetH2 = h2s.find(h2 => h2.textContent.toLowerCase().includes("estimated budget"));
  if (!budgetH2) return;

  let el = budgetH2.nextElementSibling;
  let isInTable = false;
  let rows = [];
  let currentRow = null;
  let elementsToRemove = [];

  // Data ko scrape karo
  while (el && el.tagName !== "H2") {
    if (el.textContent.trim().toLowerCase() === "estimated cost breakdown") {
      isInTable = true;
      elementsToRemove.push(el);
      el = el.nextElementSibling;
      continue;
    }

    if (isInTable) {
      if (el.tagName === "H3" && el.textContent.toLowerCase().includes("total estimated")) {
        break;
      }

      if (el.tagName === "P" && !el.textContent.includes("Group low") && !el.textContent.includes("How this is estimated") && el.textContent.trim() !== "") {
         currentRow = { category: el.textContent.trim() };
         elementsToRemove.push(el);
      }

      if (el.tagName === "P" && el.textContent.includes("Group low")) {
         const match = el.textContent.match(/Group low([\d,]+)Group high([\d,]+)/);
         if (match && currentRow) {
             currentRow.low = match[1];
             currentRow.high = match[2];
         }
         elementsToRemove.push(el);
      }

      if (el.tagName === "P" && el.textContent.includes("How this is estimated")) {
         if (currentRow) {
             currentRow.basis = el.textContent.replace("How this is estimated", "").trim();
             rows.push(currentRow);
             currentRow = null;
         }
         elementsToRemove.push(el);
      }
    }
    el = el.nextElementSibling;
  }

  // Agar purana ganda data mila hai, toh use hata kar naya table banao
  if (rows.length > 0) {
    elementsToRemove.forEach(node => node.remove());

    const table = document.createElement("table");
    table.className = "budget-table"; // Yeh aapki CSS use karega
    let html = `
      <thead>
        <tr><th>Category</th><th>Group low (₹)</th><th>Group high (₹)</th><th>Basis</th></tr>
      </thead>
      <tbody>
    `;
    rows.forEach(r => {
      html += `<tr>
        <td>${r.category}</td>
        <td>${r.low}</td>
        <td>${r.high}</td>
        <td class="budget-basis-cell">${r.basis}</td>
      </tr>`;
    });
    html += `</tbody>`;
    table.innerHTML = html;
    
    // Table ko sahi jagah insert karo
    const breakdownHeading = Array.from(container.querySelectorAll("h3")).find(h => h.textContent.toLowerCase().includes("estimated cost breakdown"));
    if (breakdownHeading) {
       breakdownHeading.insertAdjacentElement('afterend', table);
    } else {
       budgetH2.insertAdjacentElement('afterend', table);
    }
  }
}

// 3. MAIN MARKDOWN RENDERER
function renderMarkdown(element, markdown) {
  const source = markdown || "";

  if (typeof marked !== "undefined") {
    const html = marked.parse(source, {
      breaks: true,
      gfm: true
    });

    if (typeof DOMPurify !== "undefined") {
      element.innerHTML = DOMPurify.sanitize(html, {
        USE_PROFILES: { html: true }
      });

      // Links open in new tab
      element.querySelectorAll("a[href]").forEach((link) => {
        link.setAttribute("target", "_blank");
        link.setAttribute("rel", "noopener noreferrer");
      });

      // Call the formatters in the correct order
      decorateSectionHeadings(element); 
      formatItineraryTable(element); 
      formatBudgetSection(element); 
      // Naya Table force karne ke liye
      forceBudgetTable(element); 

    } else {
      element.textContent = source;
    }
  } else {
    element.textContent = source;
  }
}

function showWorkflow(data) {
  const section = byId("workflowSection");
  const reasoning = byId("supervisorReasoning");
  const chips = byId("agentChips");
  const guardrailBadge = byId("guardrailBadge");
  reasoning.textContent = data.supervisor_reasoning || "Your travel request has been reviewed and the relevant specialists have been selected.";
  chips.replaceChildren();
  (data.selected_agents || []).forEach((agent) => {
    const chip = document.createElement("span");
    chip.className = "agent-chip";
    chip.textContent = AGENT_LABELS[agent] || String(agent).replaceAll("_", " ");
    chips.appendChild(chip);
  });
  if (data.guardrail_allowed === false) {
    guardrailBadge.textContent = "Request needs a rethink";
    guardrailBadge.classList.add("blocked");
  } else {
    guardrailBadge.textContent = "Checks complete";
    guardrailBadge.classList.remove("blocked");
  }
  section.classList.remove("hidden");
}

function showResult(answer, threadId, isDraft = false) {
  latestAnswerMarkdown = answer || "";
  renderMarkdown(byId("resultBox"), latestAnswerMarkdown);
  byId("threadInfo").textContent = threadId ? `Journey reference: ${threadId}` : "Your personalised plan";
  byId("resultTitle").textContent = isDraft ? "Your first draft" : "Your Avelora travel plan";
  byId("resultSection").classList.remove("hidden");
  byId("resultSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

function showApproval(data) {
  waitingForApproval = true;
  byId("approvalRequest").textContent = data.approval_request || "Review your draft. Approve it to continue, or tell us what you’d like to change.";
  byId("approvalSection").classList.remove("hidden");
  byId("approvalSection").scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideApproval() {
  waitingForApproval = false;
  byId("approvalSection").classList.add("hidden");
  byId("approvalFeedback").value = "";
}

async function readApiResponse(response, fallbackMessage) {
  let data;
  try { data = await response.json(); }
  catch { throw new Error(fallbackMessage); }
  if (!response.ok || !data.success) throw new Error(data.error || fallbackMessage);
  return data;
}

async function sendMessage() {
  hideError();
  if (waitingForApproval) {
    showError("Please approve or refine the current draft before starting another journey.");
    return;
  }
  const message = byId("userInput").value.trim();
  if (!message) {
    showError("Tell us a little about the journey you have in mind first.");
    byId("userInput").focus();
    return;
  }
  setLoading(true, "draft");
  try {
    const response = await fetch("/api/travel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: currentThreadId })
    });
    const data = await readApiResponse(response, "We couldn't create your plan just now. Please try again.");
    currentThreadId = data.thread_id || currentThreadId;
    if (currentThreadId) localStorage.setItem("travel_thread_id", currentThreadId);
    showWorkflow(data);
    if (data.requires_approval) {
      showResult(data.itinerary || data.answer, data.thread_id, true);
      showApproval(data);
    } else {
      hideApproval();
      showResult(data.answer || data.itinerary || "No plan content was returned.", data.thread_id, false);
    }
  } catch (error) {
    showError(error.message || "We couldn't reach the travel planner. Please try again.");
  } finally {
    setLoading(false, "draft");
  }
}

async function submitApproval(approved) {
  hideError();
  if (!currentThreadId || !waitingForApproval) {
    showError("There isn't a draft waiting for your review right now.");
    return;
  }
  const feedback = byId("approvalFeedback").value.trim();
  if (!approved && !feedback) {
    showError("Tell us what you'd like to change so we can refine your plan.");
    byId("approvalFeedback").focus();
    return;
  }
  setLoading(true, "approval");
  try {
    const response = await fetch("/api/travel/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: currentThreadId, approved, feedback })
    });
    const data = await readApiResponse(response, "We couldn't update your plan. Please try again.");
    showWorkflow(data);
    if (data.requires_approval) {
      showResult(data.itinerary || data.answer, data.thread_id, true);
      showApproval(data);
    } else {
      hideApproval();
      showResult(data.answer || data.itinerary || "Your plan was updated, but no itinerary text was returned.", data.thread_id, false);
    }
  } catch (error) {
    showError(error.message || "We couldn't resume the travel workflow. Please try again.");
  } finally {
    setLoading(false, "approval");
  }
}

async function copyResult() {
  const text = byId("resultBox").innerText.trim();
  if (!text) return showError("There isn't a travel plan to copy yet.");
  try {
    await navigator.clipboard.writeText(text);
    const button = document.querySelector(".copy-btn");
    const oldText = button.textContent;
    button.textContent = "Copied ✓";
    setTimeout(() => { button.textContent = oldText; }, 1500);
  } catch {
    showError("Copying wasn't available in this browser. You can select the plan text and copy it manually.");
  }
}

function downloadPDF() {
  const pdfContent = byId("pdfContent");
  if (!latestAnswerMarkdown || !pdfContent) return showError("Create a travel plan before downloading a PDF.");
  if (typeof html2pdf === "undefined") return showError("The PDF tool didn't load. Check your connection and try again.");
  const downloadBtn = document.querySelector(".download-btn");
  const oldText = downloadBtn.innerHTML;
  downloadBtn.textContent = "Preparing PDF…";
  downloadBtn.disabled = true;
  const options = {
    margin: [0.45, 0.45, 0.45, 0.45],
    filename: "avelora-travel-plan.pdf",
    image: { type: "jpeg", quality: 0.96 },
    html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff", logging: false },
    jsPDF: { unit: "in", format: "a4", orientation: "portrait" },
    pagebreak: { mode: ["avoid-all", "css", "legacy"] }
  };
  html2pdf().set(options).from(pdfContent).save()
    .catch(() => showError("We couldn't create the PDF. Please try again."))
    .finally(() => { downloadBtn.innerHTML = oldText; downloadBtn.disabled = false; });
}

// Make destination inspiration cards and quick prompt chips populate the planner.
document.addEventListener("DOMContentLoaded", () => {
  byId("currentYear").textContent = new Date().getFullYear();
  byId("userInput").addEventListener("input", updateCharCount);
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => setPrompt(button.dataset.prompt));
  });
  const menuToggle = byId("menuToggle");
  menuToggle.addEventListener("click", () => {
    const nav = document.querySelector(".nav-links");
    const isOpen = nav.classList.toggle("open");
    menuToggle.setAttribute("aria-expanded", String(isOpen));
    menuToggle.textContent = isOpen ? "×" : "☰";
  });
  document.querySelectorAll(".nav-links a").forEach((link) => link.addEventListener("click", () => {
    document.querySelector(".nav-links").classList.remove("open");
    menuToggle.setAttribute("aria-expanded", "false");
    menuToggle.textContent = "☰";
  }));
  updateCharCount();
});

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") sendMessage();
});

<div align="center">

# ✈️ Avelora

### *Your journey, thoughtfully planned.*

**An AI travel planner built as a LangGraph multi-agent system.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?logo=langchain&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-5A67D8)
![Groq](https://img.shields.io/badge/Groq-LLM-F55036)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)

</div>

---

Avelora turns a single message like *"Plan a 5-day trip to Jaipur from Delhi for 2 people under ₹60,000"* into a complete travel plan: flights, hotels, weather, a day-by-day itinerary and a budget estimate.

Behind the scenes, a **supervisor agent** validates the request, extracts the trip details, and hands the work to **specialist agents** that pull real flight, hotel and weather data through MCP servers. Before anything is finalized, **you stay in control**: approve the draft, or tell Avelora what to change and only the affected parts are re-researched.

---

## ✨ Features

- 🛡️ **Input guardrail** – every request is classified first; non-travel or harmful requests are blocked before any agent runs.
- 🧠 **Supervisor agent** – selects only the specialist agents the request needs and extracts structured trip constraints (destination, origin, dates, duration, traveller count, budget, hotel budget, style, preferences). LLM output is merged with deterministic extractors for dates, routes and traveller count.
- 🤖 **Specialist agents**
  - ✈️ **Flight agent** – flight schedules from AviationStack plus a separate web-search-based fare estimate.
  - 🏨 **Hotel agent** – discovers real hotel properties through category searches, verifies them, validates prices, and adds booking links.
  - 🌦️ **Weather agent** – current weather and forecast from OpenWeather, described only for the dates the forecast actually covers.
  - 🗓️ **Itinerary agent** – builds the day-by-day plan and validates that every requested day is present.
  - 💰 **Budget agent** – deterministic, group-aware cost estimate in INR, using researched prices where sources state them.
- 🙋 **Human-in-the-loop** – the graph pauses at an approval step (LangGraph `interrupt`). Approve to finalize, or reject with feedback.
- 🔁 **Targeted revisions** – feedback revises the existing draft and re-runs only the affected specialist agents, instead of restarting the whole plan.
- 💾 **Persistent state** – graph checkpoints are stored in PostgreSQL, so a paused plan can be resumed by `thread_id`.
- 🖥️ **Web UI** – single-page frontend with Markdown rendering, copy-to-clipboard and PDF download.

---

## 🏗️ Architecture

### 🧭 System overview

```mermaid
flowchart LR
    subgraph Client["Browser"]
        UI["index.html + script.js<br/>marked · DOMPurify · html2pdf"]
    end

    subgraph Server["FastAPI app (app.py)"]
        API["POST /api/travel<br/>POST /api/travel/approve<br/>GET /health"]
    end

    subgraph Core["backend.py"]
        Graph["LangGraph StateGraph<br/>supervisor + specialist agents"]
    end

    LLM["Groq LLM<br/>openai/gpt-oss-120b"]
    PG[("PostgreSQL<br/>LangGraph checkpointer")]

    subgraph MCP["MCP servers (mcp_client.py)"]
        Tavily["Tavily<br/>streamable HTTP"]
        Avi["AviationStack<br/>stdio via uvx"]
        Wx["Custom Weather server<br/>stdio · custom_weather_mcp_server.py"]
    end

    OW["OpenWeather API"]
    FX["Frankfurter FX API"]
    OSM["OpenStreetMap Nominatim"]

    UI -->|JSON| API
    API --> Graph
    Graph --> LLM
    Graph <-->|checkpoints| PG
    Graph --> Tavily
    Graph --> Avi
    Graph --> Wx
    Wx --> OW
    Graph --> FX
    Graph --> OSM
```

### 🤖 Agent workflow

```mermaid
flowchart TD
    START([Start]) --> SUP["Supervisor<br/>guardrail · agent selection · constraint extraction"]

    SUP -->|blocked| GB["Guardrail blocked"]
    GB --> END1([End])

    SUP -->|allowed| FL["Flight agent"]
    FL --> HO["Hotel agent"]
    HO --> WE["Weather agent"]
    WE --> IT["Itinerary agent"]
    IT --> BU["Budget agent"]

    BU --> HA{{"Human approval<br/>interrupt"}}

    HA -->|approved| FIN["Final agent<br/>returns the reviewed draft unchanged"]
    FIN --> END2([End])

    HA -->|rejected + feedback| REV["Revision agent<br/>edits draft · plans which agents to re-run"]
    REV -->|affected agents| RERUN["Re-run selected agents<br/>flight · hotel · weather · budget"]
    REV -->|none| RF
    RERUN --> RF["Revision finalize<br/>rebuilds final document"]
    RF --> END3([End])
```

Notes on the flow:

- The supervisor only routes to the agents it selected. Agents always run in the fixed order **flight → hotel → weather → itinerary → budget** (sequential, not parallel). The itinerary agent is always included.
- If the budget agent is not selected, the flow goes from the itinerary agent straight to human approval.
- A revised draft goes through `revision_finalize` and is returned as the final version for that request; it does not loop back to a second approval.

### 🗂️ Shared state

All agents read from and write to a single `TravelState` (a LangGraph `TypedDict`). Key fields: `user_query`, `trip_constraints`, `selected_agents`, `flight_data` / `flight_fare`, `hotel_data`, `weather_data`, `itinerary`, `budget_results`, `approved`, `human_feedback`, `revision_count`, `llm_calls`.

### 🔄 Request lifecycle

```mermaid
sequenceDiagram
    participant U as Browser
    participant A as FastAPI
    participant G as LangGraph
    participant P as PostgreSQL

    U->>A: POST /api/travel {message, thread_id?}
    A->>G: run_travel_agent()
    G->>P: checkpoint after each step
    G-->>A: paused at human approval
    A-->>U: draft + requires_approval = true + thread_id

    alt Approve
        U->>A: POST /api/travel/approve {thread_id, approved: true}
        A->>G: resume_travel_agent()
        G-->>A: final itinerary
    else Reject with feedback
        U->>A: POST /api/travel/approve {thread_id, approved: false, feedback}
        A->>G: resume_travel_agent()
        G-->>A: revised final itinerary
    end
    A-->>U: result
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI, Uvicorn, Jinja2 |
| Agent orchestration | LangGraph, LangChain |
| LLM | Groq – `openai/gpt-oss-120b` (via `langchain-groq`) |
| Tool protocol | Model Context Protocol (`mcp`, `langchain-mcp-adapters`) |
| Web search | Tavily MCP |
| Flights | AviationStack MCP (`aviationstack-mcp`, launched with `uvx`) |
| Weather | Custom FastMCP server on OpenWeather |
| Persistence | PostgreSQL (`psycopg`, `langgraph-checkpoint-postgres`) |
| Airport / geo data | `airportsdata`, `pycountry`, `timezonefinder` |
| Frontend | HTML, CSS, vanilla JavaScript, `marked`, `DOMPurify`, `html2pdf.js` |

---

## 📁 Project Structure

```
Avelora/
├── app.py                        # FastAPI app and API routes
├── backend.py                    # LangGraph graph, all agents, helpers, rendering
├── mcp_client.py                 # LLM setup and MCP client (Tavily, AviationStack, Weather)
├── custom_weather_mcp_server.py  # OpenWeather MCP server (stdio)
├── check_aviation_tools.py       # Lists the tools exposed by the AviationStack MCP server
├── requirements.txt
├── templates/
│   └── index.html                # Single-page UI
├── static/
│   ├── style.css
│   └── script.js                 # API calls, rendering, approval flow, PDF export
└── tests/                        # Unit tests and diagnostic scripts
```

---

## 🚀 Getting Started

### 📋 Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) installed (the AviationStack MCP server is launched with `uvx`; check with `uvx --version`)
- A PostgreSQL database
- API keys for Groq, Tavily, AviationStack and OpenWeather

### 📦 Installation

```bash
git clone <your-repo-url>
cd Avelora

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 🔑 Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
AVIATION_STACK_API_KEY=your_aviationstack_key
OPENWEATHER_API_KEY=your_openweather_key
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

| Variable | Used for |
|---|---|
| `GROQ_API_KEY` | LLM calls (required at startup) |
| `TAVILY_API_KEY` | Hotel, fare and cost web search |
| `AVIATION_STACK_API_KEY` | Flight schedules (`AVIATIONSTACK_API_KEY` is also accepted) |
| `OPENWEATHER_API_KEY` | Current weather and forecast |
| `DATABASE_URL` | LangGraph checkpoint storage (`sslmode=require` is appended automatically if missing) |

The checkpoint tables are created automatically on startup.

### ▶️ Run

```bash
python app.py
```

or

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000**.

---

## 🔌 API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/api/travel` | Start a plan. Body: `{ "message": "...", "thread_id": "optional" }` |
| `POST` | `/api/travel/approve` | Resume a paused plan. Body: `{ "thread_id": "...", "approved": true, "feedback": "" }` |
| `GET` | `/health` | Health check |

`feedback` is required when `approved` is `false`.

**Example**

```bash
curl -X POST http://127.0.0.1:8000/api/travel \
  -H "Content-Type: application/json" \
  -d '{"message": "Plan a 5-day trip to Jaipur from Delhi for 2 people with a budget of ₹60,000"}'
```

The response includes `thread_id`, `requires_approval`, the draft `itinerary`, per-agent results (`flight_data`, `hotel_data`, `weather_data`, `budget_results`), the `selected_agents`, the extracted `trip_constraints` and `supervisor_reasoning`.

Then approve it:

```bash
curl -X POST http://127.0.0.1:8000/api/travel/approve \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "<thread_id from previous response>", "approved": true}'
```

---

## 🧪 Testing

Tests use `unittest`. Importing `backend` connects to PostgreSQL and requires `GROQ_API_KEY`, so a valid `.env` is needed to run any test that imports it.

```bash
python -m unittest tests.test_budget_agent
python -m unittest tests.test_hotel_agent
python -m unittest tests.test_supervisor_constraints
python -m unittest tests.test_revision_agent
python -m unittest tests.test_weather_mcp_server
```

Some files in `tests/` (for example `test_aviation_*`, `test_flight_*`, `test_future_*`, `test_tavily`) are diagnostic scripts that call live external services and consume API quota.

---

## 📝 Design Notes

- **Guardrail first.** The user request is treated as untrusted data in the guardrail and supervisor prompts.
- **No invented facts.** Agents work from retrieved data; missing dates, routes or prices are reported as missing instead of being assumed.
- **Approval is a confirmation step.** On approval, `final_agent` returns the exact draft the user reviewed, without regenerating it.
- **Execution is sequential.** Independent agents (such as hotel and weather) currently run one after another.
- **Budget currency.** Budgets are estimated in INR. A budget given without a currency is assumed to be INR, and foreign-currency prices are converted using the Frankfurter exchange-rate API (cached for 6 hours).
- **Airport resolution.** A city is resolved in this order: known alias, direct IATA code, then an exact city-name match in `airportsdata`. The city-name match currently covers Indian airports only, so cities elsewhere need an alias or an IATA code. If a city has several airports, the one nearest the geocoded city centre is chosen (OpenStreetMap Nominatim).

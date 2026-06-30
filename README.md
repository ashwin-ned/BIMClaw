# BIMClaw

**BIMClaw** is an agentic spatial reasoning system for IFC/BIM building models. Upload an IFC file and ask natural-language questions — *"How many doors are on the ground floor?", "What material is the exterior wall?", "Which spaces are on Level 2?"* — and BIMClaw answers by writing and executing Python code against the live building model.

It is built as a direct adaptation of [SpatialClaw](../SpatialClaw), a training-free spatial reasoning framework originally designed for video/image understanding. BIMClaw replaces image-based perception with IFC structured data, giving it exact geometry and semantics rather than inferred estimates.

---

## Setup & Installation

### 1. Directory layout

BIMClaw depends on **SpatialClaw**, which must live as a sibling directory:

```
your-workspace/
├── SpatialClaw/     ← clone this first
└── BIMClaw/         ← this repo
```

Clone both:

```bash
git clone https://github.com/your-org/SpatialClaw.git
git clone https://github.com/your-org/BIMClaw.git
```

### 2. Install Ollama and pull the model

Install [Ollama](https://ollama.com/download) on your host machine (not inside Docker).

Then pull the default model (~18 GB):

```bash
cd BIMClaw
bash start.sh ollama-install
```

To use a different model, set `OLLAMA_MODEL` in `.env` before running:

```bash
cp .env.example .env
# edit .env — change OLLAMA_MODEL if needed
```

Supported alternatives: `qwen2.5-coder:7b` (faster, less capable), `qwen2.5:72b` (stronger, needs ~40 GB VRAM), or any model with a compatible OpenAI-compat endpoint.

### 3. Install Docker

Docker Engine and Docker Compose v2 are required. The backend and frontend run entirely inside containers — no Python or Node.js needs to be installed on the host.

- [Install Docker Engine](https://docs.docker.com/engine/install/)

### 4. Run

```bash
cd BIMClaw
bash start.sh up
```

This will:
1. Start Ollama on the host (binding to `0.0.0.0:11434` so Docker can reach it)
2. Build and start the backend (FastAPI) and frontend (Next.js) containers

Open **http://localhost:3000**.

### 5. Stop

```bash
bash start.sh down
```

---

### Running the agent outside Docker (development)

If you want to run the Python agent directly without Docker, use the conda environment:

```bash
conda env create -f environment.yml
conda activate bimclaw

# Run a single query from the CLI
python -m bim_agent.entrypoints.serve \
    --ifc BIM-GNI-Dataset/val/model_0.ifc \
    --question "How many doors are in this building?"
```

`ifcopenshell` is best installed via conda-forge (included in `environment.yml`). The pip package may be outdated.

---

## How SpatialClaw Works

SpatialClaw is a **code-writing agent** built on [LangGraph](https://github.com/langchain-ai/langgraph). Its core insight: rather than fine-tuning a model on spatial tasks, give a capable LLM a persistent Python environment with spatial perception tools, and let it write and run its own analysis code.

### The Agent Loop

Every query runs through a LangGraph state machine with seven nodes:

```
         ┌─────────────────────────────────────────────────────┐
         │                  LangGraph Graph                    │
         │                                                     │
         │   ┌────────┐   ┌────────┐   ┌──────────────┐       │
  query ─┼──►│  init  │──►│  plan  │──►│  llm_step    │◄──┐  │
         │   └────────┘   └────────┘   └──────┬───────┘   │  │
         │                                     │           │  │
         │                            ┌────────▼───────┐   │  │
         │                            │    execute     │   │  │
         │                            └────────┬───────┘   │  │
         │                                     │           │  │
         │                            ┌────────▼───────┐   │  │
         │                            │    feedback    │   │  │
         │                            └────────┬───────┘   │  │
         │                                     │           │  │
         │                            ┌────────▼───────┐   │  │
         │                            │   reflection   │   │  │
         │                            └────────┬───────┘   │  │
         │                                     │           │  │
         │                  ┌──────────────────▼──────┐   │  │
         │                  │         router          │   │  │
         │                  └──┬──────────────────┬───┘   │  │
         │                     │  continue        │       │  │
         │                     │  ─────────────────┘       │  │
         │           done/     │  force_terminate           │  │
         │           max steps ▼                            │  │
         │                   END                            │  │
         └─────────────────────────────────────────────────┘
```

| Node | Role |
|---|---|
| **init** | Sets up the Jupyter kernel, injects the model data and tools, builds the first user message |
| **plan** | Calls the LLM once with a planning prompt to generate a step-by-step execution plan (non-fatal: if this fails the agent proceeds directly) |
| **llm_step** | Calls the LLM to produce the next code cell: validated structured response with Purpose / Reasoning / Next Goal / Code sections |
| **execute** | Runs the code cell in the persistent Jupyter kernel via `jupyter_client`; captures stdout, errors, and any `show()` images |
| **feedback** | Collects execution output, diffs kernel variables, checks for the `ReturnAnswer()` sentinel; builds the next HumanMessage for the LLM |
| **reflection** | Periodically reviews the conversation to update a checklist of open questions (optional, off by default for local LLMs) |
| **router** | Decides what comes next: continue the loop, or `force_terminate` when step/failure limits are hit |

### The Persistent Jupyter Kernel

The agent's "workspace" is a live `ipykernel` process. Each code cell the LLM writes is a Jupyter cell — variables from step 1 are still in scope in step 5. The kernel is pre-loaded with domain-specific tools and data objects so the LLM never has to open files or set up imports.

### Structured Response Format

Every LLM response must follow an exact four-section format:

```
**Purpose:** What this step achieves.
**Reasoning:** Why this approach.
**Next Goal:** What the next step will do.
**Code:**
```python
# executable code here
```
```

The response validator rejects malformed outputs and counts them as failures, so the LLM is forced to stay on-format or the session terminates cleanly.

### Termination

The loop ends when:
- The agent calls `ReturnAnswer(value)` from inside a code cell
- `max_steps` is reached (default 20)
- `max_failures` is reached (default 15)

On forced termination, a fallback LLM call uses whatever context is available to produce a best-effort answer.

---

## How BIMClaw Adapts SpatialClaw for BIM

SpatialClaw was designed for video frames — it had to *reconstruct* geometry from monocular depth and camera poses. BIM files skip all of that: the **exact geometry, materials, spatial containment, properties, and quantities are already in the IFC file**. BIMClaw replaces SpatialClaw's perception pipeline with direct IFC parsing.

### What Changes

| SpatialClaw | BIMClaw |
|---|---|
| Input: video frames / RGBD | Input: `.ifc` file |
| DA3/Pi3 depth estimation | Not needed — IFC has exact geometry |
| Reconstructed point cloud | IFC spatial structure (`IfcRelContainedInSpatialStructure`) |
| `Reconstruction` object in kernel | `IFCModel` + `BIMMetadata` objects in kernel |
| `tools.Reconstruct` | `tools.IFC` + `tools.BIMGeometry` |
| Visual question answering | Semantic + geometric BIM queries |
| Floor plan: reconstructed BEV | Floor plan: IFC geometry projected to 2D |

### What Stays the Same

The entire agent loop is shared: LangGraph graph, Jupyter kernel manager, AST safety sandbox, structured response validation, feedback collection, variable tracking, logging, and the Docker frontend. BIMClaw is SpatialClaw with the perception layer swapped out and BIM-specific tools injected.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Docker Stack                           │
│                                                                 │
│  ┌──────────────────────────┐   ┌──────────────────────────┐  │
│  │      bimclaw-frontend     │   │      bimclaw-backend     │  │
│  │  (Next.js, port 3000)    │   │  (FastAPI, port 8000)    │  │
│  │                          │   │                          │  │
│  │  UploadZone              │   │  POST /api/upload        │  │
│  │  BIMViewer (IFC)    ◄────┼───┤  POST /api/query         │  │
│  │  ChatBox                 │   │  GET  /api/model/{id}    │  │
│  │  FloorPlan               │   │  GET  /api/session/{id}  │  │
│  └──────────────────────────┘   └──────────┬───────────────┘  │
│                                             │                   │
│                                    ┌────────▼────────┐         │
│                                    │  BIMAgentWorkflow│         │
│                                    │  (bim_agent/)   │         │
│                                    └────────┬────────┘         │
│                                             │                   │
│          ┌──────────────────────────────────┼──────────┐       │
│          │          LangGraph Agent Loop    │          │       │
│          │                                 │          │       │
│          │  ┌──────────┐   ┌──────────┐   ▼          │       │
│          │  │IFCModel  │   │BIMMetadata│  LLM calls   │       │
│          │  │(ifopensh)│   │(summary) │              │       │
│          │  └──────────┘   └──────────┘              │       │
│          │                                            │       │
│          │  ┌──────────────────────────────────────┐  │       │
│          │  │        Jupyter Kernel (ipykernel)    │  │       │
│          │  │                                      │  │       │
│          │  │  IFCModel     BIMMetadata            │  │       │
│          │  │  InputImages  tools.IFC              │  │       │
│          │  │  tools.BIMGeometry  tools.Geometry   │  │       │
│          │  │  show()       ReturnAnswer()         │  │       │
│          │  └──────────────────────────────────────┘  │       │
│          └────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────┬───────────────────────────┘
                                      │ http://host.docker.internal:11434
                               ┌──────▼──────┐
                               │   Ollama    │  (host, 0.0.0.0:11434)
                               │ qwen2.5-    │
                               │ coder:32b   │
                               └─────────────┘
```

### Components

**`bim_frontend/ui/`** — Next.js/TypeScript frontend
- Drag-and-drop IFC upload
- Floor plan BEV image panel
- Chat interface for asking questions
- IFC viewer panel (work in progress for v2)

**`bim_frontend/api/`** — FastAPI backend
- Receives uploaded IFC files, assigns model IDs
- Routes queries to `BIMAgentWorkflow`
- Streams the IFC file back to the frontend viewer

**`bim_agent/`** — Core agentic loop
- `workflow.py` — orchestrates the full session (parse IFC → build kernel → run graph → return answer)
- `nodes/` — the seven LangGraph nodes
- `llm/` — LLM client (OpenAI-compat for Ollama; switchable to Anthropic/OpenAI)
- `tools/` — `IFCTool`, `BIMGeometryTool`, geometry/drawing utilities
- `kernel/` — Jupyter kernel manager + AST safety sandbox
- `kernel_types/` — the objects injected into the kernel namespace

**`bim_modalities/`** — Visual input pipeline (optional)
- `ifc_floor_plan.py` — renders IFC geometry to a 2D BEV floor plan image
- `pipeline.py` — orchestrates modality extraction for a given IFC file

**Ollama** — runs on the host outside Docker, serving the local LLM. Must be started with `OLLAMA_HOST=0.0.0.0` so Docker containers can reach it.

---

## Kernel Namespace

When a query runs, the Jupyter kernel is pre-loaded with:

| Variable | Type | Description |
|---|---|---|
| `IFCModel` | `IFCModel` | Wrapper around the parsed IFC file; `IFCModel.model` is the raw `ifcopenshell.file` |
| `BIMMetadata` | `BIMMetadata` | Pre-extracted summary: element counts, storey names, space names, schema version, `has_spaces`, `has_materials` |
| `InputImages` | `InputImages` | List of PIL images: `InputImages[0]` = BEV floor plan |
| `tools.IFC` | `IFCTool` | Structured IFC queries (see below) |
| `tools.BIMGeometry` | `BIMGeometryTool` | Spatial computations (distances, room areas) |
| `tools.Geometry` | `GeometryUtils` | General geometry (angles, projections) |
| `tools.Graph` | `GraphDrawer` | Scene graph drawing |
| `tools.Draw` | `DrawUtils` | Image annotation |
| `show(img)` | function | Display a PIL Image or matplotlib figure in the feedback stream |
| `ReturnAnswer(value)` | function | Submit the final answer and end the session |

### `tools.IFC` API

```python
# Element queries
tools.IFC.query_elements("IfcDoor")
# → list of dicts: [{id, GlobalId, Name, Type, storey, space, properties}, ...]

tools.IFC.get_spatial_structure()
# → {storey_name: {space_name: [element_dicts]}}

tools.IFC.count_in_space("Living Room", "IfcFurnishingElement")
# → int

tools.IFC.search_by_name("Bath*", ifc_class="IfcSpace")
# → list of matching elements

# Element detail
tools.IFC.get_quantities(global_id)
# → {"Length": 2.1, "Width": 0.9, "Height": 2.1, "Area": 1.89, ...}
# Reads IfcElementQuantity (exact, as modelled)

tools.IFC.get_properties(global_id, pset_name="Pset_WallCommon")
# → dict of all Pset_* property sets

tools.IFC.get_material(global_id)
# → str or dict (IfcMaterial / layer names + thicknesses)

tools.IFC.get_type_info(global_id)
# → properties from IfcRelDefinesByType (door type, window type, etc.)

tools.IFC.get_bounding_box(global_id)
# → {"min": [x,y,z], "max": [x,y,z], "size": [w,h,d]}

tools.IFC.get_placement(global_id)
# → 4×4 numpy array, world coordinates

tools.IFC.get_space_boundaries(space_name)
# → bounding elements (walls, slabs, windows) via IfcRelSpaceBoundary

tools.IFC.get_classification(global_id)
# → Uniclass / Omniclass codes
```

---

## IFC Data Model

BIMClaw queries IFC files using [ifcopenshell](https://ifcopenshell.org). The key entities:

| IFC Entity | Meaning | Usage |
|---|---|---|
| `IfcBuildingStorey` | One floor / level | `get_spatial_structure()` keys |
| `IfcSpace` | Functional room (living room, kitchen, etc.) | `count_in_space()`, `get_space_boundaries()` |
| `IfcRelContainedInSpatialStructure` | Links elements to their containing space/storey | Used internally by all spatial queries |
| `IfcElementQuantity` | Pre-computed dimensions (Length/Width/Height/Area/Volume) | `get_quantities()` |
| `IfcPropertySet` (Pset_*) | Fire rating, U-value, load-bearing flag, etc. | `get_properties()` |
| `IfcRelAssociatesMaterial` | Material assignment per element | `get_material()` |
| `IfcRelDefinesByType` | Shared type properties (e.g. all doors of the same type) | `get_type_info()` |

When a model lacks `IfcSpace` entities (`BIMMetadata.has_spaces == False`), the agent falls back to storey-level queries and `query_elements()` by class, which always works.

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- [Ollama](https://ollama.com) installed on the host
- The `qwen2.5-coder:32b` model pulled (or set `OLLAMA_MODEL` to a different model)

### First-time setup

```bash
# Pull the LLM model (~18GB)
bash start.sh ollama-install
```

### Run

```bash
bash start.sh up
```

Open **http://localhost:3000**, upload an IFC file, and ask a question.

### Stop

```bash
bash start.sh down
```

### Logs

```bash
bash start.sh logs          # backend (agent) logs
bash start.sh frontend-logs # frontend logs
```

---

## Configuration

All agent settings live in `bim_agent/config.py` (`BIMAgentConfig`). Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `llm_model` | `qwen2.5-coder:32b` | Model name (overridden by `LLM_MODEL` env var) |
| `llm_base_url` | *(from env)* | OpenAI-compat endpoint; empty = Anthropic API |
| `max_steps` | `20` | Maximum reasoning steps per query |
| `max_failures` | `15` | Maximum LLM/execution failures before force-terminate |
| `temperature` | `0.2` | Main LLM temperature |
| `planning_temperature` | `0.4` | Planning step temperature |
| `enable_planning` | `True` | Whether to run the blind planning step |
| `vision_enabled` | auto | Disabled automatically for non-vision models |

The `docker-compose.yml` passes `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` into the backend container.

### Using a different LLM

Any OpenAI-compatible endpoint works. Set in `docker-compose.yml`:

```yaml
environment:
  LLM_BASE_URL: http://host.docker.internal:11434/v1   # Ollama (default)
  LLM_MODEL: qwen2.5-coder:32b
  LLM_API_KEY: ollama
```

For the Anthropic API, leave `LLM_BASE_URL` empty and set `LLM_API_KEY` to your Anthropic key.

---

## Dataset

The `BIM-GNI-Dataset/` directory contains 208 IFC models split for evaluation:

```
BIM-GNI-Dataset/
├── train/   198 models (model_10.ifc … model_207.ifc)
└── val/     10 models  (model_0.ifc … model_9.ifc)
```

These are student-authored IFC files covering residential and small commercial buildings. Model quality varies — some lack `IfcSpace` entities or material assignments, which the agent handles gracefully by falling back to storey-level or class-level queries.

---

## Repository Structure

```
BIMClaw/
├── bim_agent/                  # Core agentic loop
│   ├── workflow.py             # Session orchestration
│   ├── state.py                # LangGraph AgentState TypedDict
│   ├── config.py               # BIMAgentConfig
│   ├── nodes/                  # LangGraph nodes
│   │   ├── init_node.py        # Kernel setup, IFC parsing, tool injection
│   │   ├── plan_node.py        # Blind planning step
│   │   ├── llm_step_node.py    # LLM call + response validation
│   │   ├── execute_node.py     # Jupyter cell execution
│   │   ├── feedback_node.py    # Output collection, ReturnAnswer detection
│   │   ├── reflection_node.py  # Checklist reflection (optional)
│   │   └── router.py           # Routing logic + force_terminate
│   ├── llm/                    # LLM client + prompts
│   │   ├── client.py           # OpenAI-compat / Anthropic wrapper
│   │   ├── system_prompt.py    # BIM agent system prompt
│   │   └── planning_prompt.py  # BIM planning prompt
│   ├── tools/                  # Kernel tools
│   │   ├── ifc_tool.py         # IFCTool (most critical)
│   │   └── bim_geometry_tool.py
│   ├── kernel/                 # Jupyter kernel management
│   │   ├── manager.py
│   │   └── safety.py           # AST-based code sandbox
│   └── kernel_types/           # Objects injected into kernel namespace
│       ├── ifc_model.py
│       ├── bim_metadata.py
│       └── return_answer.py
├── bim_modalities/             # Visual input pipeline
│   ├── ifc_floor_plan.py       # IFC → BEV floor plan image
│   └── pipeline.py             # Modality extraction orchestration
├── bim_frontend/
│   ├── api/                    # FastAPI backend
│   │   ├── main.py             # API routes
│   │   └── bim_session.py      # Session/model store
│   └── ui/                     # Next.js frontend
│       └── src/
│           ├── components/
│           │   ├── ChatBox.tsx
│           │   ├── BIMViewer.tsx
│           │   ├── FloorPlan.tsx
│           │   └── UploadZone.tsx
│           └── lib/api.ts
├── BIM-GNI-Dataset/            # IFC model dataset
├── test_logs/                  # Session logs from testing
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── start.sh                    # Startup helper
```

---

## Known Limitations (v1)

- **Floor plan**: Only a single merged BEV view is rendered. Multi-storey buildings do not yet show per-floor floor plans.
- **3D IFC viewer**: The in-browser 3D model viewer is not yet functional. The IFC file is loaded but geometry rendering is a v2 item.
- **IFC model quality**: Many student-authored IFC files have incomplete spatial structure (missing `IfcSpace`, no materials, no quantities). The agent adapts gracefully but answers may be less precise for these models.
- **Local LLM**: `qwen2.5-coder:32b` requires ~24GB VRAM. Smaller models work but produce lower-quality code and more format errors.

## Roadmap (v2)

- Per-storey floor plan rendering with storey selector in the UI
- Functional 3D IFC viewer in the browser
- Batch evaluation harness over the full val set with automatic scoring
- Support for BIM-Walker RGBD walkthroughs as an additional visual modality

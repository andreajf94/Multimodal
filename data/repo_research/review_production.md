# Production Diagram Quality Review

Reviewer: Claude Opus 4.6 | Date: 2026-04-09

Evaluating auto-generated Mermaid diagrams for VLM fine-tuning readiness.
Scale: KEEP / MARGINAL / DROP

---

## fastapi (API route diagrams)

### fastapi_pr14681 / api_routes_0.png
- **Readability**: GOOD. Very simple diagram: one route box showing "GET / tutorial007_py310.py::main" under a "/" group.
- **Signal**: LOW. Only a single trivial GET endpoint from a tutorial file. No request/response schemas, no parameters. Minimal architectural information.
- **Uniqueness**: Distinct from pr15023 but both are tutorial-level.
- **Verdict: DROP** -- trivially small (one route), tutorial code, no meaningful API structure.

### fastapi_pr15023 / api_routes_0.png
- **Readability**: POOR. The image is very wide and horizontally compressed. The four route boxes contain text like "GET /image/stream-no-async tutorial002_py310.py::stream_image_no_async" but the text is tiny and hard to read at normal zoom.
- **Signal**: LOW-MEDIUM. Shows 4 endpoints under "/image" for streaming image variants (sync, async, annotated, etc.). All from tutorial files. Slightly more complex than pr14681 but still tutorial code.
- **Uniqueness**: Different content from pr14681 but similarly shallow.
- **Verdict: MARGINAL** -- has multiple routes showing a pattern, but text is barely readable and it's tutorial code.

**fastapi summary**: Both diagrams show tutorial-level routes with minimal architectural signal. The generator may be extracting from FastAPI's example/tutorial files rather than the framework's own API. pr14681 is too trivial; pr15023 is marginal at best.

---

## prefect (API route diagrams)

### prefect_pr21063 / api_routes_0.png
- **Readability**: GOOD. Tall vertical layout with 11 route groups clearly separated. Each box shows the HTTP method, path, and handler function (e.g., "POST /count artifacts.py::count_artifacts"). Text is crisp and well-sized. Color-coding distinguishes GET (blue) from POST (green).
- **Signal**: HIGH. Shows a real production API surface: artifact CRUD operations (count, filter, latest, create, read), admin endpoints (version, settings), cloud utility callbacks (failure, success), and worker status. Handler file references give architectural context.
- **Uniqueness**: Very distinct from pr21072. Rich multi-endpoint layout.
- **Verdict: KEEP** -- excellent readability, meaningful API surface, good variety of endpoints.

### prefect_pr21072 / api_routes_0.png
- **Readability**: GOOD. Single route box, very clear text.
- **Signal**: LOW. Only one endpoint: "GET /metrics server.py::metrics". Trivially small.
- **Uniqueness**: Distinct content from pr21063 but far too minimal.
- **Verdict: DROP** -- single trivial endpoint, no architectural value.

**prefect summary**: pr21063 is an excellent diagram showing a real API with multiple endpoints, color-coded methods, and handler references. pr21072 is too minimal to be useful.

---

## prisma (ER diagrams)

### prisma_pr29251 / data_models_0.png
- **Readability**: GOOD. Single entity (posts) with 4 fields (id, created_at, title, content). Clear and crisp text.
- **Signal**: LOW. One trivial table with basic blog-post-like fields. No relationships. Looks like a test fixture or example schema.
- **Uniqueness**: Distinct from pr29268/pr29269 in content but equally shallow.
- **Verdict: DROP** -- single trivial entity, no relationships, likely test/example data.

### prisma_pr29268 / data_models_0.png
- **Readability**: POOR. Very wide horizontal layout with 7 entities side by side. Text is extremely small and nearly illegible at normal resolution. Entity names are barely readable (appear to be: org_users, org_customers, c_model_binds, org_company, webhooks, cyper_demo, mail_conversation_message, mail_message or similar).
- **Signal**: MEDIUM-HIGH. If readable, this would show a rich schema with many tables and fields -- a real production-like data model. But the text is too small to serve as useful VLM training input.
- **Uniqueness**: NONE vs pr29269 (appears pixel-identical).
- **Verdict: MARGINAL** -- rich schema content but text is too small to read. Would need re-rendering at higher resolution or split into multiple diagrams.

### prisma_pr29269 / data_models_0.png
- **Readability**: POOR. Same as pr29268.
- **Signal**: MEDIUM-HIGH. Same as pr29268.
- **Uniqueness**: NONE. Pixel-identical to pr29268.
- **Verdict: DROP** -- exact duplicate of pr29268.

## trpc (ER diagrams)

### trpc_pr6974 / data_models_0.png
- **Readability**: GOOD. Two entities (Post, Task) displayed side by side with clear, well-sized text. Fields and types are easily readable.
- **Signal**: LOW-MEDIUM. Two simple entities: Post (id, name, text, source as PosterSource enum, createdAt, updatedAt) and Task (id, text, completed, createdAt). No relationship lines between them. Looks like example/starter-app models rather than production schemas.
- **Uniqueness**: NONE vs pr7207 (identical).
- **Verdict: MARGINAL** -- readable and clean, but no relationships and likely example data. PosterSource custom type adds slight interest.

### trpc_pr7207 / data_models_0.png
- **Readability**: GOOD. Same as pr6974.
- **Signal**: LOW-MEDIUM. Same as pr6974.
- **Uniqueness**: NONE. Pixel-identical to pr6974.
- **Verdict: DROP** -- exact duplicate of pr6974.

**trpc summary**: pr6974 is readable and clean but shows simple example models with no relationships. pr7207 is an exact duplicate. At most one could be kept as a marginal example.

---

**prisma summary**: pr29251 is too trivial (single toy table). pr29268 has rich content but is unreadable due to horizontal compression. pr29269 is a duplicate. None are ideal; pr29268 is the best candidate but needs re-rendering. Same duplicate-across-PRs problem seen in dub and langchain.

---

## langchain (ER diagrams)

### langchain_pr29324 / data_models_0.png
- **Readability**: POOR. The image is very wide and horizontally laid out. Six entities (FullMd5LLMCache, UpsertionRecord, FulltextLLMCache, FullLLMCache, Model, BaseModel) are shown side-by-side. Text is small but legible if zoomed in. At normal resolution, field names are hard to read, especially for the rightmost entities.
- **Signal**: MEDIUM. Shows multiple LLM caching-related models with typed fields. No relationship lines between entities, which limits the relational signal. Still, the variety of cache models (MD5, Fulltext, Full) conveys meaningful domain structure.
- **Uniqueness**: Baseline for langchain set. Distinct content.
- **Verdict: MARGINAL** -- meaningful content but no relationships shown, and text is small. Would benefit from vertical layout or splitting.

### langchain_pr35550 / data_models_0.png
- **Readability**: GOOD. Single entity (UpsertionRecord) with 5 clearly readable fields.
- **Signal**: LOW. Only one entity with no relationships. This is a strict subset of pr29324.
- **Uniqueness**: NONE vs pr35705 (identical). Strict subset of pr29324.
- **Verdict: DROP** -- single isolated entity, no relationships, duplicate of pr35705.

### langchain_pr35705 / data_models_0.png
- **Readability**: GOOD. Single entity (UpsertionRecord) with 5 clearly readable fields.
- **Signal**: LOW. Identical to pr35550.
- **Uniqueness**: NONE. Pixel-identical to pr35550.
- **Verdict: DROP** -- exact duplicate of pr35550, which is itself a subset of pr29324.

**langchain summary**: pr29324 is the only candidate worth considering, and even it is marginal due to no relationship lines and small text. pr35550 and pr35705 are identical single-table extractions. The generator is again failing to scope diagrams to PR-specific changes.

---

## dub (ER diagrams)

### dub_pr3514 / data_models_0.png
- **Readability**: GOOD. Three entities (Account, Session, User) with clear field names and types. Text is crisp and readable.
- **Signal**: GOOD. Shows a real auth-related ER schema with typed fields (String, Int, DateTime, Boolean) and relationship lines with cardinality markers.
- **Uniqueness**: Baseline for dub set.
- **Verdict: KEEP**

### dub_pr3526 / data_models_0.png
- **Readability**: GOOD. Single entity (User) with clear fields.
- **Signal**: LOW. Only shows one table (User) with no relationships. This is a strict subset of pr3514's diagram -- the User table is identical.
- **Uniqueness**: POOR. It is literally the User entity from pr3514 with nothing else. No relational context.
- **Verdict: DROP** -- single isolated table, no relationships, strict subset of pr3514.

### dub_pr3533 / data_models_0.png
- **Readability**: GOOD. Identical layout to pr3514.
- **Signal**: GOOD (same as pr3514).
- **Uniqueness**: NONE. This is pixel-identical to dub_pr3514/data_models_0.png. Same three entities, same fields, same layout.
- **Verdict: DROP** -- exact duplicate of pr3514.

### dub_pr3558 / data_models_0.png
- **Readability**: GOOD. Identical layout to pr3514.
- **Signal**: GOOD (same as pr3514).
- **Uniqueness**: NONE. Also pixel-identical to dub_pr3514/data_models_0.png.
- **Verdict: DROP** -- exact duplicate of pr3514.

**dub summary**: Only pr3514 should be kept. pr3526 is a degenerate subset, and pr3533/pr3558 are exact duplicates. The diagram generator is producing the same schema snapshot for multiple PRs, suggesting the IR extraction is not scoped to the PR diff.

---

## Summary Table

| Repo | PR | Diagram | Verdict | Reason |
|------|----|---------|---------|--------|
| dub | pr3514 | data_models_0 | **KEEP** | 3 entities with relationships, clear text, real auth schema |
| dub | pr3526 | data_models_0 | **DROP** | Single table, strict subset of pr3514 |
| dub | pr3533 | data_models_0 | **DROP** | Pixel-identical duplicate of pr3514 |
| dub | pr3558 | data_models_0 | **DROP** | Pixel-identical duplicate of pr3514 |
| fastapi | pr14681 | api_routes_0 | **DROP** | Single trivial tutorial route |
| fastapi | pr15023 | api_routes_0 | **MARGINAL** | 4 routes but tiny text, tutorial code |
| langchain | pr29324 | data_models_0 | **MARGINAL** | 6 entities but no relationships, small text |
| langchain | pr35550 | data_models_0 | **DROP** | Single entity, duplicate of pr35705 |
| langchain | pr35705 | data_models_0 | **DROP** | Pixel-identical duplicate of pr35550 |
| prefect | pr21063 | api_routes_0 | **KEEP** | 11 route groups, color-coded, real production API |
| prefect | pr21072 | api_routes_0 | **DROP** | Single trivial endpoint |
| prisma | pr29251 | data_models_0 | **DROP** | Single trivial toy table |
| prisma | pr29268 | data_models_0 | **MARGINAL** | Rich 7-entity schema but text too small to read |
| prisma | pr29269 | data_models_0 | **DROP** | Pixel-identical duplicate of pr29268 |
| trpc | pr6974 | data_models_0 | **MARGINAL** | Clean 2-entity diagram but no relationships, example data |
| trpc | pr7207 | data_models_0 | **DROP** | Pixel-identical duplicate of pr6974 |

### Totals
- **KEEP: 2** (dub_pr3514, prefect_pr21063)
- **MARGINAL: 4** (fastapi_pr15023, langchain_pr29324, prisma_pr29268, trpc_pr6974)
- **DROP: 10**

### Systemic Issues Identified
1. **Massive duplication**: 6 of 16 diagrams are pixel-identical duplicates of another diagram in the same repo. The IR/diagram generator is not scoping to PR-specific schema changes -- it renders the full repo schema each time.
2. **Single-entity degenerates**: 4 diagrams show only one isolated entity with no relationships, providing negligible training signal.
3. **No relationship lines in ER diagrams**: Only dub_pr3514 shows actual relationship lines with cardinality. All other ER diagrams are just isolated entity boxes with no connections, defeating the purpose of an ER diagram.
4. **Horizontal compression**: Multi-entity diagrams (langchain_pr29324, prisma_pr29268) render too wide, making text illegibly small. The renderer needs a max-width or vertical stacking strategy.
5. **Tutorial/example extraction**: fastapi diagrams extract from tutorial files rather than framework internals. The file filter may need tuning.

### Recommendations
- Deduplicate diagrams before including in training set (hash-based or perceptual similarity)
- Drop single-entity diagrams (minimum 2 entities for ER, minimum 3 routes for API)
- Fix renderer to use vertical layout when entity count exceeds 4
- Scope IR extraction to files changed in the PR diff, not the entire repo
- For ER diagrams, require at least one relationship line or drop the diagram

# ER Diagram Quality Review — Sandbox Batch 2

Reviewer: Claude Opus 4.6 | Date: 2026-04-09

Rating scale: 1 (poor) to 5 (excellent)

---

## DOCUMENSO (ER diagrams, sample 5 of 10)

### pr2548 — data_models_0.png
- **Content**: 7 entities (User, Organisation, OrganisationGlobalSettings, TeamGlobalSettings, Recipient, Team) laid out horizontally. Rich schema with typed fields.
- **Readability**: 2/5 — Extremely small text due to horizontal layout cramming 7 wide tables into one row. Field names and types are barely legible at normal zoom. No relationship lines visible.
- **Signal**: 4/5 — Good domain coverage showing user/org/team/recipient structure with settings entities. Useful for understanding the data model.
- **Uniqueness**: 3/5 — Standard org/team/user pattern but settings entities add domain specificity.
- **Verdict**: MARGINAL — High signal content but the tiny text makes it nearly useless as a visual input to a VLM. Would need re-rendering at larger scale.

### pr2639 — data_models_0.png
- **Content**: 7 entities arranged vertically with relationship lines (User, Signature, Recipient, Field, Envelope, DocumentVersionType). Clear ER diagram with crow's-foot notation.
- **Readability**: 4/5 — Good vertical layout with legible text. Relationship lines with cardinality markers are visible. Moderate density, well-spaced.
- **Signal**: 5/5 — Excellent domain signal: shows document signing workflow (User -> Signature -> Recipient -> Field -> Envelope). Relationships are clear and meaningful.
- **Uniqueness**: 4/5 — Documenso-specific signing domain model, not a generic CRUD schema.
- **Verdict**: KEEP — Well-structured ER diagram with clear relationships and strong domain signal.

### pr2654 — data_models_0.png
- **Content**: Single entity (Webhook) with 10 fields including webhookUrl, eventTriggers, secret, enabled, userId, teamId.
- **Readability**: 5/5 — Large, perfectly legible table. Clean formatting with type and field name columns clearly separated.
- **Signal**: 2/5 — Only one entity with no relationships. A webhook config table is generic infrastructure, not domain-specific.
- **Uniqueness**: 1/5 — Webhook table is boilerplate found in almost any SaaS app.
- **Verdict**: DROP — Single generic entity with no relationships provides minimal training signal.

### pr2661 — data_models_0.png
- **Content**: Single entity (Signature) with 8 fields including recipientId, fieldId, signatureImageAsBase64, typedSignature.
- **Readability**: 5/5 — Large, clear, perfectly readable. Clean two-column layout.
- **Signal**: 2/5 — Single entity, no relationships shown. Some domain-specific fields (signatureImageAsBase64) but isolated.
- **Uniqueness**: 2/5 — Signature entity is somewhat domain-specific but without context of related entities it's thin.
- **Verdict**: DROP — Single isolated entity, insufficient relational context for ER training.

### pr2686 — data_models_0.png
- **Content**: 3 entities (Recipient, TemplateDirectLink, Envelope) with crow's-foot relationship lines connecting them. Envelope is central with connections to both others.
- **Readability**: 5/5 — Excellent layout, large text, clear relationship lines with cardinality notation (one-to-many, etc.).
- **Signal**: 4/5 — Good domain model showing envelope/recipient/template-link relationships. Meaningful cardinality.
- **Uniqueness**: 4/5 — Document signing domain-specific entities with clear business logic in relationships.
- **Verdict**: KEEP — Clean multi-entity diagram with visible relationships and good domain signal.

---

## FORMBRICKS (ER diagrams, sample 5 of 10)

### pr7530 — data_models_0.png
- **Content**: Single entity (Language) with 8 fields: id, createdAt, updatedAt, code, alias, project, projectId, surveyLanguages.
- **Readability**: 5/5 — Large, clear, perfectly legible.
- **Signal**: 1/5 — Single entity with no relationships. A language/i18n config table is generic infrastructure.
- **Uniqueness**: 1/5 — Internationalization boilerplate, no domain specificity.
- **Verdict**: DROP — Single generic entity, no relationships, no useful training signal.

### pr7611 — data_models_0.png
- **Content**: 3 entities (User, Environment, Organization) laid out horizontally. No relationship lines between them.
- **Readability**: 4/5 — Clear text, well-sized tables. Horizontal layout works for 3 entities. No crowding.
- **Signal**: 2/5 — Three core entities but no relationship lines. The entities themselves are generic (User, Org, Environment). Some formbricks-specific fields (twoFactorSecret, actionClasses, surveys).
- **Uniqueness**: 2/5 — User/Org/Environment is a standard SaaS pattern.
- **Verdict**: MARGINAL — Readable but lacks relationships and domain specificity. Borderline useful.

### pr7647 — data_models_0.png
- **Content**: 3 entities (Webhook, Response, Organization) laid out horizontally. No relationship lines.
- **Readability**: 4/5 — Clear text, good sizing. Clean layout.
- **Signal**: 2/5 — Three disconnected entities. Response has survey-related fields (surveyId, contactId, endingId) which hint at the domain but no relationships are shown.
- **Uniqueness**: 2/5 — Webhook + Response + Org is semi-generic. Response entity has some formbricks flavor.
- **Verdict**: MARGINAL — Decent readability but no relationships reduces training value.

### pr7668 — data_models_0.png
- **Content**: 8 entities (ContactAttributeKey, Survey, Webhook, ActionClass, Segment, Contact, ApiKeyEnvironment, Environment) with relationship lines converging on Environment.
- **Readability**: 2/5 — Very small text in the top row of 7 entities. The bottom Environment entity is readable but the connected entities above are tiny. Relationship lines are visible but cramped.
- **Signal**: 4/5 — Rich domain model showing survey platform structure: contacts, segments, action classes, webhooks all connected to Environment. Good hub-and-spoke pattern.
- **Uniqueness**: 4/5 — Survey platform-specific entities with meaningful domain relationships.
- **Verdict**: MARGINAL — Good signal but readability is poor due to cramped layout. A VLM may struggle with the tiny text in the top row.

### pr7692 — data_models_0.png
- **Content**: ~10 entities spread across two rows with relationship lines. Includes User, OrganizationBillingSettings, Role, Survey, Organization, Membership, Invite, Project, Contact, Segment.
- **Readability**: 1/5 — Extremely small text, barely legible even when zooming. Entity names are hard to read, field names nearly impossible. Relationship lines are thin and hard to trace.
- **Signal**: 4/5 — Would be high signal if readable: comprehensive org/project/survey/billing model with relationships.
- **Uniqueness**: 3/5 — Mix of generic (User/Org/Membership) and domain-specific (Survey/Segment/Contact).
- **Verdict**: DROP — Too small to be useful as visual input. The VLM cannot extract meaningful information from text this tiny.

---

## PAPERLESS-NGX (ER diagrams, sample 5 of 10)

### pr12065 — data_models_0.png
- **Content**: ~8 entities laid out horizontally (WorkflowAction, WorkflowTrigger, SavedView, MailRule/MailAccount-related, SearchHitWorkflow, MailBox, CustomField, WorkflowCondition, SavedViewFilterRule). Django-style field types (CharField, ManyToManyField, etc.).
- **Readability**: 1/5 — Extremely small text. Entities are crammed into a single horizontal row. Field names are essentially unreadable. No relationship lines visible despite the entities clearly being related.
- **Signal**: 3/5 — Would have good signal if readable: workflow/trigger/action pattern with mail integration and saved views is domain-rich.
- **Uniqueness**: 4/5 — Document management + workflow automation is a distinctive domain. Django field types add framework specificity.
- **Verdict**: DROP — Text is too small to be useful for VLM training. Layout makes the content inaccessible.

### pr12142 — data_models_0.png
- **Content**: 2 entities (Workflow, CustomField) side by side. Workflow has 5 fields (name, order, triggers, actions, enabled). CustomField has 4 fields (created, name, data_type, extra_data).
- **Readability**: 5/5 — Large, perfectly legible text. Clean layout with good spacing.
- **Signal**: 2/5 — Only 2 disconnected entities, no relationships shown. The entities themselves are somewhat generic.
- **Uniqueness**: 2/5 — Workflow + CustomField is a common pattern. Django field types (ManyToManyField, BooleanField) add slight specificity.
- **Verdict**: MARGINAL — Very readable but too sparse. Two disconnected entities provide limited relational learning signal.

### pr12260 — data_models_0.png
- **Content**: Single entity (CustomField) with 4 fields: created, name, data_type, extra_data.
- **Readability**: 5/5 — Large, perfectly clear.
- **Signal**: 1/5 — Single entity, 4 fields, no relationships. Minimal information content.
- **Uniqueness**: 1/5 — Generic custom field entity.
- **Verdict**: DROP — Single small entity with no relationships. Essentially zero training value.

### pr12273 — data_models_0.png
- **Content**: ~8 entities in a horizontal row. Appears identical to pr12065.
- **Readability**: 1/5 — Same cramped, illegible horizontal layout as pr12065. Text is unreadable.
- **Signal**: 3/5 — Same rich domain model as pr12065, but unreadable.
- **Uniqueness**: 1/5 — Duplicate of pr12065 layout.
- **Verdict**: DROP — Duplicate of pr12065 with same readability issues. Also a duplication concern.

### pr12276 — data_models_0.png
- **Content**: ~8 entities in a horizontal row. Appears identical to pr12065 and pr12273.
- **Readability**: 1/5 — Same illegible layout as pr12065 and pr12273.
- **Signal**: 3/5 — Same underlying model.
- **Uniqueness**: 0/5 — Third duplicate of the same diagram.
- **Verdict**: DROP — Triplicate diagram. Even if readable, duplicates waste training budget.

---

## TRIGGER.DEV (ER + API diagrams, sample 5 of 10)

### pr3275 — data_models_0.png
- **Content**: 2 entities (Project, Waitpoint) side by side. Project has 10 fields, Waitpoint has 10 fields including domain-specific ones (WaitpointType, WaitpointStatus, idempotencyKey, completedByTaskRunId).
- **Readability**: 5/5 — Large, clear text. Two well-sized tables with good spacing.
- **Signal**: 3/5 — Two disconnected entities. Waitpoint is highly domain-specific (task orchestration concept) with interesting fields. But no relationships are shown.
- **Uniqueness**: 4/5 — Waitpoint with idempotency keys and task run references is distinctive to trigger.dev's job orchestration domain.
- **Verdict**: MARGINAL — Excellent readability and domain-specific content, but only 2 disconnected entities limits relational learning value.

### pr3308 — data_models_0.png
- **Content**: Single entity (LlmModel) with 10 fields: id, friendlyId, projectId, project, modelName, matchPattern, startDate, source, createdAt, updatedAt.
- **Readability**: 5/5 — Large, perfectly clear. Clean single-table layout.
- **Signal**: 2/5 — Single entity. LLM model registry is interesting but isolated. matchPattern and source hint at model routing logic.
- **Uniqueness**: 3/5 — LLM model tracking within a task runner platform is somewhat unique.
- **Verdict**: DROP — Single entity provides insufficient relational training signal despite being domain-interesting.

### pr3315 — data_models_0.png
- **Content**: ~8 entities arranged in two rows (TaskRun-related, BackgroundWorker, RuntimeEnvironment, WaitpointTokenStatus, and others). No visible relationship lines.
- **Readability**: 1/5 — Extremely small text. Entity names barely discernible, field names completely illegible. Two-row layout is too compressed.
- **Signal**: 4/5 — Would be high signal: task run orchestration with workers, environments, waitpoints. Core trigger.dev domain model.
- **Uniqueness**: 4/5 — Task orchestration platform internals are highly domain-specific.
- **Verdict**: DROP — Illegible. Rich domain model completely undermined by tiny rendering.

### pr3319 — api_routes_0.png
- **Content**: API route diagram (not ER). Shows route groups: /messages, /sse, /v1 (traces, logs, chat/completions), /usage, /api (test, whitelist, runs metadata), /healthcheck, and catch-all. Color-coded: green=POST, blue=GET, orange=PUT/test, gray=ALL.
- **Readability**: 5/5 — Large, clear text. Color coding is effective. Route hierarchy is visually intuitive with nested groupings.
- **Signal**: 4/5 — Shows MCP server integration (/messages, /sse), observability endpoints (/v1/traces, /v1/logs), LLM proxy (/v1/chat/completions), and standard infra. File references (mcpServer.ts, utils.ts, server.ts) provide implementation context.
- **Uniqueness**: 5/5 — Distinctive API surface combining MCP protocol, LLM chat completions proxy, and task orchestration. Very specific to trigger.dev.
- **Verdict**: KEEP — Excellent API route diagram. Clear, informative, and highly domain-specific. Strong training signal.

### pr3331 — data_models_0.png
- **Content**: ~8 entities in two rows. Appears identical to pr3315.
- **Readability**: 1/5 — Same illegible tiny text as pr3315.
- **Signal**: 4/5 — Same rich model as pr3315.
- **Uniqueness**: 0/5 — Duplicate of pr3315.
- **Verdict**: DROP — Duplicate of pr3315 with same illegibility issues.

---

## SUMMARY TABLE

| Repo | PR | Diagram | Verdict | Reason |
|------|-----|---------|---------|--------|
| documenso | pr2548 | ER | MARGINAL | 7 entities but text too small to read |
| documenso | pr2639 | ER | KEEP | 7 entities, clear relationships, good domain signal |
| documenso | pr2654 | ER | DROP | Single generic Webhook entity |
| documenso | pr2661 | ER | DROP | Single isolated Signature entity |
| documenso | pr2686 | ER | KEEP | 3 entities with clear crow's-foot relationships |
| formbricks | pr7530 | ER | DROP | Single generic Language entity |
| formbricks | pr7611 | ER | MARGINAL | 3 readable entities but no relationships |
| formbricks | pr7647 | ER | MARGINAL | 3 readable entities but no relationships |
| formbricks | pr7668 | ER | MARGINAL | 8 entities with relationships but text too small |
| formbricks | pr7692 | ER | DROP | ~10 entities, completely illegible |
| paperless-ngx | pr12065 | ER | DROP | ~8 entities, completely illegible |
| paperless-ngx | pr12142 | ER | MARGINAL | 2 readable entities, no relationships |
| paperless-ngx | pr12260 | ER | DROP | Single tiny CustomField entity |
| paperless-ngx | pr12273 | ER | DROP | Duplicate of pr12065, illegible |
| paperless-ngx | pr12276 | ER | DROP | Triplicate of pr12065, illegible |
| trigger.dev | pr3275 | ER | MARGINAL | 2 readable domain-specific entities, no relationships |
| trigger.dev | pr3308 | ER | DROP | Single LlmModel entity |
| trigger.dev | pr3315 | ER | DROP | ~8 entities, completely illegible |
| trigger.dev | pr3319 | API | KEEP | Excellent API route map, clear and distinctive |
| trigger.dev | pr3331 | ER | DROP | Duplicate of pr3315, illegible |

### Aggregate Stats
- **KEEP**: 3/20 (15%) — documenso pr2639, documenso pr2686, trigger.dev pr3319
- **MARGINAL**: 6/20 (30%) — documenso pr2548, formbricks pr7611/pr7647/pr7668, paperless-ngx pr12142, trigger.dev pr3275
- **DROP**: 11/20 (55%)

### Key Failure Modes
1. **Illegibility (7 diagrams)**: The most common failure. When 6+ entities are rendered in a single horizontal row, text becomes unreadable. This affects paperless-ngx worst (3/5 identical illegible diagrams) and trigger.dev (2 identical illegible diagrams).
2. **Single-entity diagrams (5 diagrams)**: PRs that touch only one model produce a single table with no relationships — zero relational signal for ER training.
3. **Duplication (3 diagrams)**: paperless-ngx pr12065/pr12273/pr12276 produce the exact same diagram. trigger.dev pr3315/pr3331 are also duplicates. This suggests the diagram generation is deterministic and PR-agnostic for these repos — it may be rendering the full schema rather than PR-scoped changes.
4. **Missing relationships (4 diagrams)**: Several multi-entity diagrams show entities side-by-side but with no relationship lines, reducing them to glorified field lists.

### Recommendations
- **Re-render large diagrams** at higher resolution or split into sub-diagrams to fix the illegibility problem.
- **Filter out single-entity diagrams** automatically — require minimum 2 entities with at least 1 relationship.
- **Deduplicate** — hash the generated Mermaid source to detect identical diagrams across PRs in the same repo.
- **Investigate paperless-ngx generation** — 3 identical diagrams across different PRs suggests the extractor is not scoping to PR-changed models.


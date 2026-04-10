# ER Diagram Quality Review (Sandbox)

## authentik (5 samples)

### authentik_pr21130 — data_models_0.png
- **Readability**: POOR. The diagram contains ~8 entity boxes but is rendered at very small resolution. Field names and types are barely legible — requires zooming/squinting. Two-column layout (type | name) is correct but text is tiny.
- **Signal**: GOOD. Shows real Django-style models: SerializerMetaclass, User, Screen, StepBase, Waiting, SearchLookupMask, DateTimeAction(?), RadioGroupPermission. Fields include UUIDField, TextField, BooleanField, ManyToManyField, DateTimeField — these are genuine Django ORM field types with meaningful names (slug, name, path, sources, groups, password_change_date, etc.).
- **Uniqueness**: Will compare across PRs below.
- **Verdict**: MARGINAL — good domain signal but readability is too poor for a VLM to extract useful information from the small text.

### authentik_pr21321 — data_models_0.png
- **Readability**: POOR. Visually identical layout and resolution to pr21130. Same tiny text, same ~8 entity boxes.
- **Signal**: GOOD. Same entity set as pr21130 — SerializerMetaclass, User, Screen, StepBase, Waiting, SearchLookupMask, DateTimeAction, RadioGroupPermission.
- **Uniqueness**: BAD. This appears to be an identical or near-identical diagram to pr21130. The layout, entities, and fields all look the same.
- **Verdict**: DROP — duplicate of pr21130, and readability is poor.

### authentik_pr21421 — data_models_0.png
- **Readability**: EXCELLENT. Single large entity box for "User" model. All fields are clearly legible: UUIDField uuid, TextField name, TextField path, TextField type, ManyToManyField sources/groups/roles, DateTimeField password_change_date/last_updated, UserManager objects.
- **Signal**: GOOD. Real User model with meaningful authentication-domain fields.
- **Uniqueness**: GOOD. Very different from the multi-entity diagrams above — this is a single-entity focused view.
- **Verdict**: KEEP — clear, readable, real domain model. However, it is a single entity which is somewhat limited for ER-diagram training.

### authentik_pr21478 — data_models_0.png
- **Readability**: POOR. Same tiny multi-entity layout as pr21130 and pr21321. Text is barely legible.
- **Signal**: GOOD. Same entity set visible.
- **Uniqueness**: BAD. Appears identical to pr21130 and pr21321.
- **Verdict**: DROP — duplicate of pr21130/pr21321, poor readability.

### authentik_pr21484 — data_models_0.png
- **Readability**: EXCELLENT. Single large entity box for "User" model, same clear format as pr21421.
- **Signal**: GOOD. Same User model fields.
- **Uniqueness**: BAD. Appears identical to pr21421.
- **Verdict**: DROP — duplicate of pr21421.

### authentik Summary
3 of 5 are duplicates of the tiny multi-entity diagram. 2 of 5 are duplicates of the single User entity. Only 2 unique diagrams exist, and only 1 (the single-entity User) is readable. The multi-entity diagram has good breadth but unacceptable resolution.

---

## baserow (5 samples)

### baserow_pr5051 — data_models_0.png
- **Readability**: POOR. Wide horizontal layout with ~8 entity boxes rendered very small. Field names and types are barely legible — same tiny-text problem as authentik multi-entity diagrams. Entities appear to include: Creator(?), ServiceAccountTokenPermission, HealthCheck, DiffSync, SchemaModel(?), VersionedFileImportExport(?), LastUsed(?).
- **Signal**: GOOD. Fields include ForeignKey, CharField, PositiveIntegerField, DateTimeField, BooleanField, TextField — real Django ORM types. Field names like `session_id`, `version_name`, `api_url_creation`, `reset_or_generating_detected` suggest real Baserow domain models.
- **Uniqueness**: Will compare across PRs below.
- **Verdict**: MARGINAL — real domain signal but text is too small for a VLM to reliably read.

### baserow_pr5087 — data_models_0.png
- **Readability**: POOR. Same tiny multi-entity horizontal layout as pr5051.
- **Signal**: GOOD. Same entity set visible.
- **Uniqueness**: BAD. Appears identical to pr5051.
- **Verdict**: DROP — duplicate of pr5051, poor readability.

### baserow_pr5146 — data_models_0.png
- **Readability**: EXCELLENT. Single large entity box for "View" model. All fields clearly legible: ForeignKey table, PositiveIntegerField order, CharField name, ForeignKey content_type, CharField filter_type, BooleanField filters_disabled, SlugField slug, BooleanField public, CharField public_view_password, BooleanField show_logo.
- **Signal**: GOOD. Real Baserow View model — meaningful database/table-view domain fields including slug, public access controls, filter configuration.
- **Uniqueness**: GOOD. Completely different from the multi-entity diagrams.
- **Verdict**: KEEP — clear, readable, real domain model with meaningful fields. Single entity but rich enough.

### baserow_pr5150 — data_models_0.png
- **Readability**: POOR. Same tiny multi-entity horizontal layout as pr5051/pr5087.
- **Signal**: GOOD. Same entity set visible.
- **Uniqueness**: BAD. Appears identical to pr5051 and pr5087.
- **Verdict**: DROP — duplicate of pr5051/pr5087, poor readability.

### baserow_pr5161 — data_models_0.png
- **Readability**: GOOD. Three entity boxes — Field, Table, and FieldRule — at moderate resolution. Field and Table are medium-sized with legible text. FieldRule is smaller but still readable. Two-column type|name layout is clear.
- **Signal**: EXCELLENT. Core Baserow domain models: Field (table, order, name, primary, content_type, field_dependencies, tsvector_column_created, search_data_initialized_at, description, read_only), Table (database, order, name, _row_count, _row_count_updated_at, version, needs_background_update_column_added, last_modified_by_column_added, created_by_column_added, field_rules_validity_column_added), FieldRule (table, is_active, is_valid, error_text, content_type). These are the core data-modeling entities of a no-code database platform.
- **Uniqueness**: EXCELLENT. Completely different from both the tiny multi-entity diagram and the single View entity.
- **Verdict**: KEEP — good readability, excellent domain signal, multiple related entities at the core of Baserow's data model.

### baserow Summary
3 of 5 are duplicates of the same tiny unreadable multi-entity diagram. The remaining 2 are unique and readable: a single View entity and a 3-entity Field/Table/FieldRule diagram. The 3-entity diagram (pr5161) is the best ER sample across both repos reviewed so far.

---

## cal.com (5 samples)

### cal.com_pr15318 — data_models_0.png
- **Readability**: EXCELLENT. Two entity boxes — AccessToken and User — connected by a relationship line (crow's foot notation showing one-to-many). All text is large and perfectly legible. The relationship line with the circle-and-bar notation is a bonus — it shows actual ER semantics, not just standalone tables.
- **Signal**: EXCELLENT. AccessToken (id, secret, createdAt, expiresAt, owner, client, platformOAuthClientId, userId) and User (id, username, name, email, emailVerified, password, bio, avatarUrl, timeZone, travelSchedules). These are core Prisma/cal.com scheduling-platform models with real OAuth and user-profile fields.
- **Uniqueness**: Will compare across PRs below.
- **Verdict**: KEEP — best diagram seen so far. Readable, multi-entity with relationship lines, real domain signal. Ideal VLM training input.

### cal.com_pr28176 — data_models_0.png
- **Readability**: POOR. Multi-entity layout (~5-6 boxes) with very small text. Entity names partially visible (Schedule, Host, Team?, WebhookTriggerEvents?) but field-level text is not reliably readable. Relationship lines are visible between entities — good structural signal even if text is small.
- **Signal**: GOOD (from what is legible). Appears to show scheduling-domain entities with relationship connections.
- **Uniqueness**: GOOD. Different entity set and layout from pr15318.
- **Verdict**: MARGINAL — has relationship lines which add value, but text is too small for reliable VLM reading. Could work if the VLM only needs structural patterns rather than field-level detail.

### cal.com_pr28682 — data_models_0.png
- **Readability**: EXCELLENT. Two entity boxes — Schedule and Booking — at large readable size. All fields clearly legible. No relationship lines between them, though.
- **Signal**: EXCELLENT. Schedule (id, user, userId, eventType, instantMeetingEvents, restrictionSchedule, name, timeZone, availability, Host) and Booking (id, uid, idempotencyKey, user, userId). Core cal.com scheduling domain models. EventType and Availability type references show domain richness.
- **Uniqueness**: GOOD. Different entities from pr15318 (AccessToken/User). Covers the scheduling/booking side of cal.com.
- **Verdict**: KEEP — clear, readable, meaningful scheduling-domain models. Would be stronger with relationship lines, but still a solid training sample.

### cal.com_pr28701 — data_models_0.png
- **Readability**: EXCELLENT. Single entity box for "Webhook" model. All fields clearly legible: String id, Int userId, Int teamId, Int eventTypeId, String platformOAuthClientId, String subscriberUrl, String payloadTemplate, DateTime createdAt, Boolean active, WebhookTriggerEvents eventTriggers.
- **Signal**: GOOD. Real webhook/integration model. Fields like subscriberUrl, payloadTemplate, eventTriggers show webhook-specific domain knowledge.
- **Uniqueness**: GOOD. Different entity from all other cal.com samples.
- **Verdict**: KEEP — readable and real domain model. Single entity is a minor limitation but the fields are rich and domain-specific.

### cal.com_pr28787 — data_models_0.png
- **Readability**: POOR-to-MARGINAL. Wide horizontal layout with ~7 entity boxes and a relationship-connected entity below. Entity names partially readable: MembershipSchedule(?), MonthlyProducts(?), BookingFormRequestInternalUser(?), OrganizationSettings, BookingDirectionalLink(?), OrganizationOnboarding(?), Webhook. A connected Credential(?) entity sits below with a relationship line. Text at field level is very difficult to read.
- **Signal**: EXCELLENT (from what can be inferred). This is a rich multi-entity diagram with relationship connections, showing the organization/team layer of cal.com. Many entities with diverse fields.
- **Uniqueness**: EXCELLENT. Completely different from all other cal.com samples — covers the organization/membership domain.
- **Verdict**: MARGINAL — richest structural content of any cal.com sample (multiple entities with relationships), but readability is too low for reliable field-level VLM training. The structural layout and relationship lines may still provide useful signal.

### cal.com Summary
3 of 5 are readable (pr15318, pr28682, pr28701). All 5 are unique — no duplicates, which is excellent. pr15318 is the gold standard with relationship lines + readable text. The 2 poor-readability diagrams (pr28176, pr28787) are structurally rich but text is too small.

---

# Summary Table

| Repo | Image | Readability | Signal | Uniqueness | Verdict |
|------|-------|-------------|--------|------------|---------|
| authentik | pr21130 | POOR | GOOD | baseline | MARGINAL — real models but unreadable text |
| authentik | pr21321 | POOR | GOOD | DUPLICATE of pr21130 | DROP — duplicate + unreadable |
| authentik | pr21421 | EXCELLENT | GOOD | UNIQUE | KEEP — clear single User entity |
| authentik | pr21478 | POOR | GOOD | DUPLICATE of pr21130 | DROP — duplicate + unreadable |
| authentik | pr21484 | EXCELLENT | GOOD | DUPLICATE of pr21421 | DROP — duplicate of pr21421 |
| baserow | pr5051 | POOR | GOOD | baseline | MARGINAL — real models but unreadable text |
| baserow | pr5087 | POOR | GOOD | DUPLICATE of pr5051 | DROP — duplicate + unreadable |
| baserow | pr5146 | EXCELLENT | GOOD | UNIQUE | KEEP — clear single View entity |
| baserow | pr5150 | POOR | GOOD | DUPLICATE of pr5051 | DROP — duplicate + unreadable |
| baserow | pr5161 | GOOD | EXCELLENT | UNIQUE | KEEP — multi-entity Field/Table/FieldRule |
| cal.com | pr15318 | EXCELLENT | EXCELLENT | UNIQUE | KEEP — gold standard, relationship lines + readable |
| cal.com | pr28176 | POOR | GOOD | UNIQUE | MARGINAL — has relationships but tiny text |
| cal.com | pr28682 | EXCELLENT | EXCELLENT | UNIQUE | KEEP — clear Schedule + Booking entities |
| cal.com | pr28701 | EXCELLENT | GOOD | UNIQUE | KEEP — clear Webhook entity |
| cal.com | pr28787 | MARGINAL | EXCELLENT | UNIQUE | MARGINAL — rich structure but poor text readability |

## Overall Statistics
- **KEEP**: 6 of 15 (40%)
- **MARGINAL**: 4 of 15 (27%)
- **DROP**: 5 of 15 (33%)

## Key Findings
1. **Duplicate problem (authentik, baserow)**: Multiple PRs produce identical ER diagrams. The diagram generator likely extracts the same "top-level" models regardless of which PR-specific models changed. This is a pipeline bug — the diagram should reflect only the models touched/relevant to the PR.
2. **Resolution/scaling problem**: Multi-entity diagrams with >3 entities render text too small for VLM consumption. Single-entity or 2-3 entity diagrams are consistently readable.
3. **cal.com is the best repo**: All 5 diagrams are unique, 3 are fully readable, and 1 (pr15318) includes relationship lines — the only sample with true ER relationship notation.
4. **Relationship lines are rare**: Only cal.com pr15318 and pr28787 show actual ER relationship connectors. Most diagrams are just standalone entity boxes without connections, which reduces their value as "ER diagrams."
5. **Recommendation**: (a) Fix the duplicate-diagram pipeline bug. (b) Either increase render resolution for multi-entity diagrams or cap entities per diagram at 3-4. (c) Prioritize generating relationship lines between entities.

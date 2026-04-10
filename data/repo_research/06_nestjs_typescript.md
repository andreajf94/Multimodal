# TypeScript/NestJS Web App Repos — Research for RepoDesign Pipeline

Research date: 2025-04-05

Criteria:
1. Uses TypeORM, Prisma, Drizzle, or MikroORM with real schema (5+ models)
2. Has real REST API (NestJS, Express, Hono, Fastify — NOT tRPC only)
3. Squash-merges PRs (recent commits on main have "(#123)" style messages)
4. Open source, 500+ stars, actively maintained

---

## 1. Twenty (CRM) — twentyhq/twenty

- **GitHub URL:** https://github.com/twentyhq/twenty
- **Stars:** ~43,600
- **Language:** TypeScript
- **License:** Other (AGPL-style)
- **Default branch:** main

### ORM + Schema Quality
- **ORM:** TypeORM (migrated from Prisma; see Issue #1830). Uses a custom `@WorkspaceEntity` decorator layer on top of TypeORM for dynamic schema.
- **Schema richness:** Excellent. 20+ entity models including Company, Person, Opportunity, Task, Note, Favorite, Attachment, WorkspaceMember, TimelineActivity, MessagingChannel, CalendarEvent, etc. Each entity has 10-20+ fields with typed relations (one-to-many, many-to-one). Uses composite metadata types (CurrencyMetadata, LinksMetadata, AddressMetadata, ActorMetadata).
- **Entity location:** `packages/twenty-server/src/modules/*/standard-objects/*.workspace-entity.ts`
- **Database:** PostgreSQL

### REST API Structure
- **Framework:** NestJS
- **REST API:** Yes. Has both REST and GraphQL APIs. REST endpoints under `/rest/metadata` and `/rest/` for CRUD on core CRM entities (people, companies, notes, tasks). OpenAPI spec is auto-generated. Controller at `engine/core-modules/open-api/open-api.controller.ts`.
- **Batch operations:** Supports up to 60 records in a single API call.
- **Module structure:** 40+ NestJS modules under `engine/core-modules/` (auth, billing, messaging, calendar, file-storage, search, workflow, etc.)

### Squash-Merges
- **Yes.** 18 out of 20 recent commits on main have `(#NNNN)` PR reference pattern. Clear squash-merge workflow.
- Example: "Fix dark mode text color on permissions tab empty stat (#19336) (#19340)"

### Fit: HIGH
- Perfect match: deployed NestJS app, TypeORM with rich schema, REST API, squash-merges, very active (43k stars).
- **Concerns:** Custom `@WorkspaceEntity` decorator layer adds abstraction on top of TypeORM. Entity definitions use their own DSL rather than raw TypeORM `@Entity`/`@Column` decorators. This might make schema extraction slightly non-standard, but the entities are still clearly defined with typed fields and relations.

---

## 2. Novu (Notifications) — novuhq/novu

- **GitHub URL:** https://github.com/novuhq/novu
- **Stars:** ~38,800
- **Language:** TypeScript
- **License:** Other
- **Default branch:** next

### ORM + Schema Quality
- **ORM:** Mongoose (MongoDB ODM), NOT TypeORM/Prisma/Drizzle. Uses `mongoose ^8.9.5` with `mongoose-delete` plugin.
- **Schema richness:** Excellent. 31 repository models in `libs/dal/src/repositories/`: subscriber, notification, message, job, integration, environment, organization, user, workflow-override, tenant, topic, feed, layout, member, change, execution-details, notification-template, message-template, notification-group, preferences, localization, translation, channel-connection, channel-endpoint, context, control-values, ai-chat, snapshot, etc.
- **Entity location:** `libs/dal/src/repositories/*/`
- **Database:** MongoDB

### REST API Structure
- **Framework:** NestJS
- **REST API:** Yes. 51 NestJS modules under `apps/api/src/app/` with versioned controllers (v1/v2 for subscribers, workflows, environments, layouts, topics). Controller files like `subscribersV1.controller.ts` with standard NestJS decorators.
- **Module structure:** Modular feature-driven: subscribers, notifications, events, integrations, workflows, environments, feeds, topics, messages, organizations, auth, billing, etc.

### Squash-Merges
- **Yes.** All 20 recent commits on `next` branch have `(#NNNNN)` PR references. Consistent squash-merge workflow.
- Example: "refactor(api-service): enhance logging... (#10583)"

### Fit: MEDIUM
- Strong NestJS app with REST API, squash-merges, great activity.
- **Concerns:** Uses Mongoose/MongoDB, NOT a SQL ORM (TypeORM/Prisma/Drizzle/MikroORM). Does not meet criterion #1 as stated. If MongoDB/Mongoose is acceptable, this would be HIGH fit.

---

## 3. Vendure (E-commerce) — vendure-ecommerce/vendure (now vendurehq/vendure)

- **GitHub URL:** https://github.com/vendure-ecommerce/vendure
- **Stars:** ~8,000
- **Language:** TypeScript
- **License:** Other
- **Default branch:** master

### ORM + Schema Quality
- **ORM:** TypeORM with standard `@Entity()`, `@Column()`, `@ManyToOne()`, `@OneToMany()` decorators. Entities extend `VendureEntity` base class (provides id, createdAt, updatedAt).
- **Schema richness:** Excellent. **43 entity directories** in `packages/core/src/entity/`: Product, ProductVariant, ProductOption, ProductOptionGroup, Order, OrderLine, OrderModification, Payment, PaymentMethod, Refund, Fulfillment, ShippingMethod, ShippingLine, Customer, CustomerGroup, Address, Administrator, Asset, Channel, Collection, Facet, FacetValue, Promotion, Role, User, Session, TaxCategory, TaxRate, Zone, Region, Seller, StockLevel, StockLocation, StockMovement, Surcharge, Tag, GlobalSettings, HistoryEntry, ApiKey, AuthenticationMethod, SettingsStoreEntry.
- **Entity location:** `packages/core/src/entity/*/`
- **Database:** PostgreSQL, MySQL, SQLite (TypeORM multi-DB support)

### REST API Structure
- **Framework:** NestJS
- **REST API:** LIMITED. Vendure is **primarily GraphQL** (Shop API + Admin API). REST endpoints are possible but only via plugin controllers — not the default API surface. The core API is GraphQL-first.
- **Module structure:** Plugin-based architecture with NestJS underneath.

### Squash-Merges
- **Yes.** 14 out of 17 recent substantive commits have `(#NNNN)` PR references. Non-matching commits are merge commits, automated docs regen, and version releases.
- Example: "fix(dashboard): Fix option group edit link on variant detail page (#4620)"

### Fit: MEDIUM
- Excellent TypeORM schema (43 entities), squash-merges, NestJS, active.
- **Concerns:** Primarily a **framework** (headless commerce framework), not a deployed web app per se. Also **GraphQL-first** — REST endpoints are not the primary API surface and only exist via plugins. Does not fully meet criterion #2 (real REST API). Would be better described as a framework than a deployed app.

---

## 4. n8n (Workflow Automation) — n8n-io/n8n

- **GitHub URL:** https://github.com/n8n-io/n8n
- **Stars:** ~182,600
- **Language:** TypeScript
- **License:** Other (fair-code, Sustainable Use License)
- **Default branch:** master

### ORM + Schema Quality
- **ORM:** TypeORM. Entities in `packages/@n8n/db/src/entities/`.
- **Schema richness:** Good. ~35 entity files covering: WorkflowEntity, ExecutionEntity, ExecutionData, ExecutionMetadata, CredentialsEntity, User, Project, ProjectRelation, Role, RoleMapping, Scope, ApiKey, AuthIdentity, Folder, Tag, TagMapping, Settings, Variables, WebhookEntity, SharedWorkflow, SharedCredentials, WorkflowHistory, WorkflowPublishHistory, WorkflowPublishedVersion, WorkflowStatistics, WorkflowDependency, CredentialDependency, BinaryDataFile, ProcessedData, InvalidAuthToken, SecretsProviderConnection, ProjectSecretsProviderAccess, AuthProviderSyncHistory. Plus EE entities: AnnotationTag, AnnotationTagMapping, ExecutionAnnotation, TestRun, TestCaseExecution.
- **Entity location:** `packages/@n8n/db/src/entities/`
- **Database:** PostgreSQL, MySQL, SQLite (via TypeORM)

### REST API Structure
- **Framework:** Custom decorator layer on top of NestJS/Express. Uses `@RestController('/users')`, `@Get`, `@Post`, `@Patch`, `@Delete` from `@n8n/decorators` package (not raw NestJS `@Controller`).
- **REST API:** Yes, extensive. 35+ REST controller files in `packages/cli/src/controllers/`: users, auth, workflows (via separate routes), projects, folders, roles, tags, api-keys, credentials, executions, binary-data, invitations, mfa, password-reset, settings, security-settings, telemetry, workflow-statistics, dynamic-node-parameters, node-types, active-workflows, ai, orchestration, oauth, etc.
- **API style:** RESTful with custom routing decorators, scope-based authorization (`@GlobalScope`), and enterprise licensing (`@Licensed`).

### Squash-Merges
- **Yes.** 100% of recent 20 commits on master have `(#NNNNN)` PR references. Perfect squash-merge workflow.
- Example: "feat(core): Make data redaction available without feature flag (#27981)"

### Fit: HIGH
- Deployed web app (workflow automation platform), TypeORM with 35+ entities, REST API with 35+ controllers, perfect squash-merge compliance, massively popular (182k stars), very actively maintained.
- **Concerns:** Uses custom `@n8n/decorators` routing layer rather than raw NestJS `@Controller`/`@Get` decorators. Still REST-based and easily parseable, but the custom abstraction might complicate automated analysis slightly. Also fair-code license (not fully open source in the traditional sense). Monorepo is very large.

---

## 5. Amplication (Code Generation Platform) — amplication/amplication

- **GitHub URL:** https://github.com/amplication/amplication
- **Stars:** ~16,000
- **Language:** TypeScript
- **License:** Other
- **Default branch:** master

### ORM + Schema Quality
- **ORM:** Prisma. Schema at `packages/amplication-prisma-db/prisma/schema.prisma`.
- **Schema richness:** Excellent. **40 Prisma models** + 8 enums: Account, Workspace, Project, User, Role, Team, TeamAssignment, Resource, Entity, EntityVersion, EntityField, EntityPermission, EntityPermissionRole, EntityPermissionField, Block, BlockVersion, Build, BuildPlugin, Action, ActionStep, ActionLog, Commit, Environment, Deployment, GitOrganization, GitRepository, Blueprint, ResourceVersion, ResourceRole, Ownership, Invitation, Coupon, Subscription, AwsMarketplaceIntegration, CustomProperty, OutdatedVersionAlert, UserAction, Release, ResourceRelationCache, ApiToken.
- **Database:** PostgreSQL

### REST API Structure
- **Framework:** NestJS
- **REST API:** Hybrid. Has Swagger/OpenAPI documentation setup (`swagger.ts`). Primary API is GraphQL (`schema.graphql`), but REST endpoints are also exposed. The extent of REST endpoints (vs GraphQL-only) is unclear from the quick investigation.
- **Module structure:** 50+ core modules in `packages/amplication-server/src/core/`: entity, resource, project, workspace, build, action, commit, environment, auth, billing, git, role, team, permissions, etc.

### Squash-Merges
- **NO.** Only 2 out of ~20 recent commits have `(#NNNN)` squash-merge style. Most commits are explicit merge commits: "Merge pull request #9875 from amplication/next". Uses **merge commits**, not squash-merges.

### Fit: LOW
- Has Prisma with 40 models and NestJS, but **fails squash-merge criterion**. Uses merge commits. Also primarily a code-generation platform (generates apps, not a deployed app itself in the traditional sense). GraphQL-primary with some REST.
- **Concerns:** Not squash-merge. More of a meta-tool (generates code) than a deployed web app. REST API extent unclear.

---

## 6. Huly (Project Management) — hcengineering/platform

- **GitHub URL:** https://github.com/hcengineering/platform
- **Stars:** ~25,300
- **Language:** TypeScript
- **License:** EPL-2.0
- **Default branch:** develop

### ORM + Schema Quality
- **ORM:** None (no TypeORM/Prisma/Drizzle/MikroORM). Uses MongoDB with native driver and a custom model definition system. 94 model directories in `/models/` covering: contact, lead, recruit, hr, calendar, task, tracker, chat, document, drive, notification, billing, inventory, ai-assistant, etc. Very rich domain model, but defined through a custom DSL (TypeScript model builders), not a standard ORM.
- **Database:** MongoDB

### REST API Structure
- **Framework:** Mixed — Express (front server) and Koa (account-service). NOT NestJS/Fastify.
- **REST API:** Has HTTP endpoints via Express/Koa but the API architecture is custom platform-specific, not standard REST controllers. The platform uses its own RPC/transaction system for data operations.

### Squash-Merges
- **Yes.** 100% of recent commits on develop have `(#NNNN)` PR references. Perfect squash-merge workflow.
- Example: "qfix: Fix card warnings (#10725)"

### Fit: LOW
- Squash-merges and rich domain model, but fails on ORM criterion (MongoDB with custom DSL, not TypeORM/Prisma/Drizzle/MikroORM) and REST API criterion (custom platform, not standard NestJS/Express REST controllers).
- **Concerns:** Custom everything — custom model system, custom transaction layer, custom RPC. Not standard TypeScript web app architecture. Express/Koa used only for serving, not as REST API framework.

---

## 7. ToolJet (Low-Code Platform) — ToolJet/ToolJet

- **GitHub URL:** https://github.com/ToolJet/ToolJet
- **Stars:** ~37,700
- **Language:** JavaScript (but server is TypeScript/NestJS)
- **License:** AGPL-3.0
- **Default branch:** develop

### ORM + Schema Quality
- **ORM:** TypeORM with standard `@Entity()`, `@Column()`, `@ManyToOne()`, `@OneToMany()`, `@ManyToMany()`, `@OneToOne()` decorators. Entities extend BaseEntity.
- **Schema richness:** Excellent. **82 entity files** in `server/src/entities/`: App, AppVersion, AppHistory, AppEnvironments, User, UserDetails, UserMFA, UserSessions, Organization, OrganizationUser, GroupPermission, DataQuery, DataSource, Plugin, Credential, Folder, File, Thread, Comment, AuditLog, WorkflowBundle, WorkflowSchedule, WorkflowExecution, SSOConfig, InternalTable, AIChatPrompt, AIConversation, and many more. Entities have typed fields and rich relations.
- **Entity location:** `server/src/entities/`
- **Database:** PostgreSQL (via TypeORM)

### REST API Structure
- **Framework:** NestJS
- **REST API:** Yes. 57 NestJS modules in `server/src/modules/`: app, users, organizations, data-sources, data-queries, auth, folders, plugins, ai, workflows, audit-logs, files, configs, email, CRM, group-permissions, etc. Each module contains NestJS controllers with standard decorators.
- **ormconfig.ts** present at server root confirming TypeORM configuration.

### Squash-Merges
- **Mostly yes.** ~73% of recent commits (11/15) have `(#NNNNN)` PR references. Some commits lack the pattern (standalone fixes, security patches). Not as consistent as Twenty or n8n.
- Example: "docs: update LTS version table (#15820)"

### Fit: HIGH
- Deployed web app (low-code platform), NestJS + TypeORM with 82 entities, REST API with 57 modules, actively maintained (37k stars).
- **Concerns:** Primary repo language reported as JavaScript (not TypeScript), though the server is TypeScript/NestJS. Squash-merge compliance is ~73%, not 100%. The repo is large and includes significant frontend code. AGPL license.

---

## 8. Medusa (E-commerce) — medusajs/medusa

- **GitHub URL:** https://github.com/medusajs/medusa
- **Stars:** ~32,600
- **Language:** TypeScript
- **License:** MIT
- **Default branch:** develop

### ORM + Schema Quality
- **ORM:** Uses MikroORM under the hood BUT with a **custom Data Model Language (DML)** abstraction. Models defined via `model.define("Product", {...})` fluent API, NOT standard MikroORM `@Entity()`/`@Property()` decorators. Migrated from TypeORM to MikroORM in v2.
- **Schema richness:** Excellent. 35 modules in `packages/modules/` covering: product, order, cart, customer, payment, fulfillment, inventory, pricing, promotion, tax, currency, region, sales-channel, store, notification, user, api-key, auth, rbac, stock-location, translation, file, settings, analytics, etc.
- **REST API routes:** 50+ admin API routes covering products, orders, customers, payments, fulfillments, inventory, promotions, campaigns, shipping, regions, tax, collections, etc. Plus store-facing API.
- **Database:** PostgreSQL

### REST API Structure
- **Framework:** Express.js (not NestJS)
- **REST API:** Yes, extensive. Admin API at `/api/admin/` with 50+ resource routes. Store API at `/api/store/`. Uses Express route handlers organized by resource.
- **Module structure:** 35 independent modules, each with models, services, and API routes.

### Squash-Merges
- **Yes.** 100% of recent 15 commits have `(#NNNNN)` PR references. Perfect squash-merge compliance.
- Example: "feat(http-types-generator): Add an HTTP types generator and validator for Zod schemas (#14988)"

### Fit: MEDIUM
- Deployed e-commerce platform, REST API, squash-merges, MIT license, very active (32k stars).
- **Concerns:** Uses custom DML on top of MikroORM, NOT standard MikroORM entity decorators. This custom model definition language (`model.define()`) is a non-standard abstraction. Also uses Express (not NestJS/Fastify). Whether the custom DML counts as "using MikroORM" depends on interpretation. Could be considered a framework rather than a deployed app.

---

## 9. Immich (Photo Management) — immich-app/immich

- **GitHub URL:** https://github.com/immich-app/immich
- **Stars:** ~90,000+
- **Language:** TypeScript (+ Dart for mobile)
- **License:** AGPL-3.0
- **Default branch:** main

### ORM + Schema Quality
- **ORM:** Originally TypeORM, but as of v1.132.0 **migrated away from TypeORM** to Kysely + custom `@immich/sql-tools` decorator-based schema definition. Tables defined with custom `@Table()`, `@Column()`, `@ForeignKeyColumn()` decorators from `@immich/sql-tools`.
- **Schema richness:** Excellent. **54 table definition files** in `server/src/schema/tables/`: asset, album, album-asset, album-user, user, person, face-search, tag, stack, memory, library, notification, session, api-key, partner, shared-link, activity, workflow, plugin, system-metadata, sync-checkpoint, plus many audit tables. PostgreSQL-specific (trigram indexes, vector search, triggers).
- **Entity location:** `server/src/schema/tables/`
- **Database:** PostgreSQL (with pgvecto.rs for ML embeddings)

### REST API Structure
- **Framework:** NestJS
- **REST API:** Yes, extensive. **~30 NestJS controllers** in `server/src/controllers/`: asset, album, user, person, face, tag, search, library, map, timeline, memory, notification, partner, shared-link, activity, auth, oauth, session, api-key, server, system-config, download, duplicate, trash, stack, workflow, plugin, job, queue, view, sync, database-backup, maintenance. Auto-generated OpenAPI spec used to generate client SDKs (TypeScript, Dart).
- **API quality:** Strong REST conventions, OpenAPI/Swagger integration, separate admin endpoints.

### Squash-Merges
- **Yes.** 100% of recent commits have `(#NNNNN)` PR references. Perfect squash-merge compliance.
- Example: "refactor(mobile): introduce image request registry on iOS (#27486)"

### Fit: MEDIUM-HIGH
- Deployed self-hosted web app, NestJS with REST API, squash-merges, massively popular (90k+ stars), very active.
- **Concerns:** No longer uses TypeORM/Prisma/Drizzle/MikroORM. Migrated to custom `@immich/sql-tools` + Kysely. Does NOT meet criterion #1 (standard ORM). If the custom schema definition is acceptable (it IS decorator-based and readable), this would be HIGH fit.

---

## 10. Cal.com (Scheduling) — calcom/cal.com

- **GitHub URL:** https://github.com/calcom/cal.com
- **Stars:** ~41,000
- **Language:** TypeScript
- **License:** Other (AGPLv3 + commercial)
- **Default branch:** main

### ORM + Schema Quality
- **ORM:** Prisma with standard `schema.prisma` file at `packages/prisma/schema.prisma`.
- **Schema richness:** Outstanding. **113 Prisma models** covering: User, Booking, EventType, Team, Membership, Schedule, Availability, Payment, Webhook, Workflow, WorkflowStep, Credential, App, ApiKey, Host, DestinationCalendar, BookingReference, Attendee, BookingSeat, OAuthClient, DelegationCredential, Profile, Organization, Attribute, Role, Agent, Task, FilterSegment, RoutingForm, and many more. One of the richest schemas investigated.
- **Entity location:** `packages/prisma/schema.prisma`
- **Database:** PostgreSQL

### REST API Structure
- **Framework:** NestJS (Platform API v2) + tRPC (internal Next.js API)
- **REST API:** Yes. Platform API v2 is a full NestJS REST API at `apps/api/v2/` with **36 modules**: event-types, booking-seat, slots, users, teams, memberships, organizations, profiles, roles, auth, api-keys, apps, credentials, webhooks, workflows, billing, stripe, oauth-clients, cal-unified-calendars, conferencing, destination-calendars, selected-calendars, routing-forms, ooo, deployments, verified-resources, email, timezones, tokens, jwt, redis, prisma, kysely, atoms, router. OpenAPI spec auto-generated from NestJS decorators.
- **tRPC also present:** The internal Next.js frontend uses tRPC for type-safe communication, but the Platform API v2 is a standalone NestJS REST API.

### Squash-Merges
- **Yes.** 100% of recent 20 commits have `(#NNNNN)` PR references. Perfect squash-merge compliance.
- Example: "feat(bookings): add booking audit logging to instant bookings (#28176)"

### Fit: HIGH
- Deployed scheduling web app, Prisma with 113 models, NestJS REST API (Platform API v2) with 36 modules, perfect squash-merges, very active (41k stars).
- **Concerns:** The NestJS REST API is the "Platform API v2" (external integration API), not the primary app's API surface (which uses tRPC). So the REST API is a secondary interface. However, it IS a real, well-structured NestJS REST API with proper controllers, services, DTOs, and OpenAPI docs. Large monorepo.

---

## Summary & Rankings

### Repos that meet ALL criteria (HIGH fit):

| Rank | Repo | Stars | ORM | Entities/Models | REST API | Squash | Notes |
|------|------|-------|-----|-----------------|----------|--------|-------|
| 1 | **Cal.com** | 41k | Prisma (standard) | 113 models | NestJS Platform API v2, 36 modules | 100% | Best overall: standard Prisma, NestJS, huge schema, MIT-ish |
| 2 | **Twenty** | 43k | TypeORM (custom decorator layer) | 20+ entities | NestJS REST + GraphQL | 90% | Custom @WorkspaceEntity wrapper over TypeORM |
| 3 | **n8n** | 182k | TypeORM (standard) | 35+ entities | Custom REST decorators, 35+ controllers | 100% | Custom @RestController instead of raw NestJS |
| 4 | **ToolJet** | 37k | TypeORM (standard) | 82 entities | NestJS, 57 modules | ~73% | Standard TypeORM, standard NestJS. Squash compliance lower |

### Repos with notable concerns (MEDIUM fit):

| Repo | Stars | Issue |
|------|-------|-------|
| **Immich** | 90k | Custom `@immich/sql-tools`, no longer TypeORM/Prisma. Otherwise excellent |
| **Medusa** | 32k | Custom DML on MikroORM, Express not NestJS |
| **Novu** | 38k | Mongoose/MongoDB, not SQL ORM |
| **Vendure** | 8k | GraphQL-first, REST only via plugins; more framework than app |

### Repos that fail criteria (LOW fit):

| Repo | Stars | Issue |
|------|-------|-------|
| **Amplication** | 16k | Merge commits (not squash), code-gen tool not deployed app |
| **Huly** | 25k | Custom MongoDB model DSL, custom RPC, not standard web app |

### Recommendation

**Top picks for the pipeline (in order):**
1. **Cal.com** — cleanest match: standard Prisma (113 models), NestJS REST API (Platform API v2), 100% squash-merge, very active. The REST API is a secondary interface (primary uses tRPC), but it's well-structured and real.
2. **n8n** — standard TypeORM (35+ entities), REST API (35+ controllers), 100% squash-merge. Custom routing decorators are a mild concern but still easily parseable.
3. **ToolJet** — standard TypeORM (82 entities), standard NestJS (57 modules), solid squash-merge (~73%). Most conventional NestJS+TypeORM architecture of all investigated.
4. **Twenty** — rich schema, NestJS, good squash-merge. Custom entity decorator layer is the main concern.

If Mongoose/MongoDB is acceptable: **Novu** becomes HIGH fit (38k stars, NestJS, 31 models, REST, 100% squash-merge).
If custom schema tools are acceptable: **Immich** becomes HIGH fit (90k+ stars, NestJS, 54 tables, REST, 100% squash-merge).

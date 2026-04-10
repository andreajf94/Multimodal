# Repo Research: DevOps, CMS, and Collaboration Tools

Investigation date: 2026-04-05

Criteria: Real DB schema (5+ models), REST API, squash-merge PRs, open source, 1000+ stars, actively maintained.

---

## DevOps / Dev Tools

### 1. Coolify

- **URL:** https://github.com/coollabsio/coolify
- **Language/Stack:** PHP (Laravel), Livewire, PostgreSQL
- **Stars:** ~52,600
- **Default branch:** v4.x
- **Schema quality:** Excellent. 54 Eloquent models (Application, Server, Service, Project, Environment, Team, ScheduledDatabaseBackup, PrivateKey, S3Storage, StandalonePostgresql, StandaloneMysql, etc.). Full Laravel migration system.
- **Has REST API:** Yes. Dedicated `routes/api.php` (~21KB) plus `webhooks.php`. Comprehensive REST endpoint definitions.
- **Squash-merges:** Mostly yes. Most recent commits have `(#number)` pattern (e.g., "v4.0.0-beta.471 (#9206)", "feat(jobs): implement exponential backoff (#9184)"). A few direct commits without PR numbers.
- **Fit:** Medium-High
- **Concerns:** PHP/Laravel stack (less common in JS-focused evaluations). Default branch is `v4.x` not `main`. Some commits lack PR numbers (direct pushes mixed in).

### 2. Unleash

- **URL:** https://github.com/Unleash/unleash
- **Language/Stack:** TypeScript (Node.js backend), PostgreSQL, Knex migrations
- **Stars:** ~13,300
- **Default branch:** main
- **Schema quality:** Good. Uses Knex for migrations (not TypeORM). Multiple DB tables: features, feature_environments, strategies, projects, environments, tags, segments, users, roles, permissions, API tokens, etc. Extensive migration history.
- **Has REST API:** Yes. Well-structured with separate groups: admin-api/, client-api/, edge-api/, auth/. Multiple endpoint controllers.
- **Squash-merges:** Yes. All 15 recent commits on main follow `(#number)` pattern perfectly (e.g., "feat: improved safeguard impact metrics filtering (#11733)").
- **Fit:** High
- **Concerns:** Feature flag platform is narrower domain. Core schema is focused on feature management rather than general-purpose CRUD. Still rich enough with 10+ core tables.

### 3. GrowthBook

- **URL:** https://github.com/growthbook/growthbook
- **Language/Stack:** TypeScript (Node.js backend), MongoDB (Mongoose-style models)
- **Stars:** ~7,600
- **Default branch:** main
- **Schema quality:** Excellent. 74 data models in `packages/back-end/src/models/` (ExperimentModel, FeatureModel, MetricModel, DataSourceModel, ProjectModel, OrganizationModel, UserModel, TeamModel, SegmentModel, etc.).
- **Has REST API:** Yes. ~40 router files in `packages/back-end/src/routers/` covering experiments, features, metrics, environments, projects, teams, users, etc.
- **Squash-merges:** Yes. All 15 recent commits follow `(#number)` pattern perfectly (e.g., "feat(api): Flesh out experiment endpoints (#5608)").
- **Fit:** High
- **Concerns:** Uses MongoDB (not SQL), which means no traditional ORM schema - models are document-based. Lower star count (~7.6k) but still above threshold.

### 4. Hoppscotch

- **URL:** https://github.com/hoppscotch/hoppscotch
- **Language/Stack:** TypeScript (NestJS backend, Vue.js frontend), Prisma, PostgreSQL
- **Stars:** ~78,800
- **Default branch:** main
- **Schema quality:** Good. 23 Prisma models (Team, TeamMember, TeamInvitation, TeamCollection, TeamRequest, User, Account, UserSettings, UserHistory, UserCollection, PersonalAccessToken, InfraConfig, MockServer, etc.).
- **Has REST API:** Yes. NestJS backend with modular architecture (team-invitation service, user services, etc.). Likely uses both REST and GraphQL.
- **Squash-merges:** Mostly yes. 12/15 recent commits have `(#number)` pattern. 3 commits lack PR numbers (direct pushes like version bumps).
- **Fit:** Medium
- **Concerns:** Hoppscotch is primarily an API testing tool - its own backend is relatively simple compared to the tool's complexity (the app is about testing OTHER APIs). 23 models is decent but on the lower end. Mix of squash-merge and direct commits.

### 5. Appsmith

- **URL:** https://github.com/appsmithorg/appsmith
- **Language/Stack:** TypeScript (frontend), Java (Spring Boot backend), MongoDB
- **Stars:** ~39,500
- **Default branch:** release
- **Schema quality:** Likely extensive (Java/Spring Boot backend with MongoDB). Low-code platform needs many entity types (applications, pages, widgets, datasources, actions, users, organizations, etc.).
- **Has REST API:** Yes. Spring Boot backend inherently provides REST endpoints for all CRUD operations.
- **Squash-merges:** Yes. All recent commits follow `(#number)` pattern (e.g., "fix: replace PAT with GITHUB_TOKEN (#41699)").
- **Fit:** Medium
- **Concerns:** Java backend (not TypeScript/Node). Default branch is `release` not `main`. MongoDB (no SQL schema). Very large monorepo. Backend in Java makes it less accessible for JS/TS focused analysis.

### 6. ToolJet

- **URL:** https://github.com/ToolJet/ToolJet
- **Language/Stack:** JavaScript/TypeScript (NestJS backend), PostgreSQL
- **Stars:** ~37,700
- **Default branch:** develop
- **Schema quality:** Needs deeper investigation. NestJS + PostgreSQL suggests TypeORM or Prisma models.
- **Has REST API:** Yes. NestJS provides REST controllers.
- **Squash-merges:** Mixed. 9/15 recent commits have `(#number)` pattern, but also has merge commits ("Merge pull request #15761...") and direct pushes without PR numbers.
- **Fit:** Medium
- **Concerns:** Default branch is `develop`, not `main`. Mixed merge strategy (both squash-merge and regular merge commits). Many recent commits are just "docs: update LTS version table" automated commits.

### 7. Portainer

- **URL:** https://github.com/portainer/portainer
- **Language/Stack:** TypeScript (React frontend), Go (backend)
- **Stars:** ~37,100
- **Default branch:** develop
- **Schema quality:** Go backend with its own data layer. Portainer manages Docker/K8s resources, so models include endpoints, stacks, users, teams, registries, etc.
- **Has REST API:** Yes. Go backend with comprehensive REST API for container management.
- **Squash-merges:** Yes. All 15 recent commits follow `(#number)` pattern with ticket IDs (e.g., "refactor(stack): create stack and deploy stack in async flow [BE-12650] (#2048)").
- **Fit:** Medium
- **Concerns:** Go backend (not JS/TS). Default branch is `develop`. Backend is Go, not TypeScript. Data layer may use BoltDB or similar embedded DB rather than traditional SQL ORM.

---

## CMS / BaaS

### 8. Payload CMS

- **URL:** https://github.com/payloadcms/payload
- **Language/Stack:** TypeScript (Next.js framework), Drizzle ORM, PostgreSQL/SQLite/MongoDB
- **Stars:** ~41,600
- **Default branch:** main
- **Schema quality:** Different model. Payload auto-generates DB schema from collection configs (Users, Media, Posts, Pages, etc.). Uses Drizzle ORM for SQL. Schema is config-driven rather than explicit model files. Built-in collections include users and auth. Rich enough in practice but schema lives in config, not traditional model files.
- **Has REST API:** Yes. Auto-generates full REST API for every collection with CRUD endpoints (GET/POST/PATCH/DELETE on /api/{collection}). Also generates GraphQL.
- **Squash-merges:** Yes. All 15 recent commits follow `(#number)` pattern perfectly (e.g., "fix(templates): tailwind file extension is wrong (#9342)").
- **Fit:** Medium
- **Concerns:** Payload is a CMS framework, not a deployed app itself. The repo is the framework code, not an app with its own domain models. Schema is user-defined via config, so the repo itself doesn't have a fixed schema to analyze. PRs tend to be template/plugin fixes rather than feature development on a product.

### 9. Directus

- **URL:** https://github.com/directus/directus
- **Language/Stack:** TypeScript (Node.js backend, Vue.js admin), Knex.js, PostgreSQL/MySQL/SQLite/etc.
- **Stars:** ~34,700
- **Default branch:** main
- **Schema quality:** Excellent. 101 migration files defining system tables: collections, fields, relations, users, roles, permissions, policies, files, activity, webhooks, flows, presets, notifications, shares, insights, dashboards, translations, extensions, themes, versioning, comments. Very rich internal schema.
- **Has REST API:** Yes. Comprehensive auto-generated REST API + GraphQL. Endpoints for items, collections, users, files, roles, permissions, activity, flows, etc. Well-documented.
- **Squash-merges:** Yes. All 15 recent commits follow `(#number)` pattern perfectly (e.g., "Fix alias fields being included when selecting all fields in export (#26775)").
- **Fit:** High
- **Concerns:** Directus is a data platform/BaaS -- its schema is its own internal system tables (not user content). PRs are genuine product features and fixes. TypeScript monorepo, well-structured.

### 10. Ghost

- **URL:** https://github.com/TryGhost/Ghost
- **Language/Stack:** JavaScript (Node.js), Knex.js/Bookshelf ORM, MySQL
- **Stars:** ~52,400
- **Default branch:** main
- **Schema quality:** Excellent. 76 database tables defined in schema.js: posts, users, members, products (tiers), offers, benefits, newsletters, emails, comments, tags, roles, permissions, integrations, webhooks, subscriptions, stripe_products, stripe_prices, actions, snippets, collections, recommendations, milestones, etc. One of the richest schemas investigated.
- **Has REST API:** Yes. Ghost has a well-documented Content API and Admin API with endpoints for posts, pages, tags, authors, tiers, members, newsletters, offers, etc.
- **Squash-merges:** Mostly yes. Most recent commits have `(#number)` pattern. A few direct commits (Renovate config tweaks) lack PR numbers.
- **Fit:** High
- **Concerns:** JavaScript (not TypeScript), though migration to TS may be underway. Some direct pushes by maintainers (Renovate bot config). Monorepo with many sub-packages. Uses MySQL not PostgreSQL.

### 11. Strapi

- **URL:** https://github.com/strapi/strapi
- **Language/Stack:** TypeScript (Node.js), Knex.js (Bookshelf-based), PostgreSQL/MySQL/SQLite
- **Stars:** ~71,800
- **Default branch:** develop
- **Schema quality:** Good but config-driven. Internal core models include admin users, roles, permissions, content-types, API tokens. User-facing schema is defined via content-type JSON configs. The framework itself has internal models for RBAC, media, i18n, etc.
- **Has REST API:** Yes. Auto-generates REST + GraphQL APIs for all content-types. Also has admin API endpoints.
- **Squash-merges:** Mixed. 9/15 recent commits have `(#number)` pattern, but also has merge commits and direct pushes without PR numbers.
- **Fit:** Medium
- **Concerns:** Default branch is `develop` not `main`. Mixed merge strategy (squash + regular merge commits). As a framework, the repo's own schema is internal/meta rather than domain-specific. Very large monorepo.

### 12. NocoDB

- **URL:** https://github.com/nocodb/nocodb
- **Language/Stack:** TypeScript (Node.js backend, Vue.js frontend), Knex.js, PostgreSQL/MySQL/SQLite
- **Stars:** ~62,600
- **Default branch:** develop
- **Schema quality:** Excellent. 105 model files in `packages/nocodb/src/models/`: User, Base, Model, View, Column, GridView, FormView, GalleryView, KanbanView, CalendarView, Hook, Filter, Sort, Comment, Audit, Integration, ApiToken, Permission, FormulaColumn, LinkToAnotherRecordColumn, LookupColumn, RollupColumn, etc.
- **Has REST API:** Yes. Comprehensive REST API for tables, records, views, columns, filters, sorts, hooks, etc. Well-documented v1 and v2 APIs.
- **Squash-merges:** No. Only 2/15 recent commits have `(#number)` pattern. Most are direct pushes or merge commits without PR references. Poor commit hygiene on develop branch.
- **Fit:** Low
- **Concerns:** Does NOT squash-merge PRs. Default branch is `develop`. Recent commit history shows many direct pushes with poor messages ("fix: message", "chore: lint"). Despite excellent schema and API, the merge strategy is a hard fail on criteria.

---

## Collaboration Tools

### 13. Mattermost

- **URL:** https://github.com/mattermost/mattermost
- **Language/Stack:** Go (backend), TypeScript (frontend/webapp), PostgreSQL/MySQL
- **Stars:** ~36,100
- **Default branch:** master
- **Schema quality:** Excellent. ~43 SQL store files in `server/channels/store/sqlstore/` covering: Users, Teams, Channels, Posts, Reactions, FileInfo, Preferences, Sessions, Tokens, Audits, Compliance, Commands, Webhooks, Bots, Emoji, Groups, Jobs, Roles, Permissions, Plugins, etc. Very rich relational schema for a collaboration platform.
- **Has REST API:** Yes. ~42 API handler files in `server/channels/api4/`: channels, posts, users, teams, bots, commands, emoji, files, groups, jobs, oauth, plugins, preferences, reactions, etc. Well-documented API v4.
- **Squash-merges:** Yes. All 15 recent commits follow `(#number)` pattern (e.g., "MM-68156: Fix space key clearing input in invite modal (#35913)").
- **Fit:** Medium-High
- **Concerns:** Go backend (not TypeScript/Node). Default branch is `master` not `main`. Very large monorepo (Go server + TypeScript webapp + mobile). The Go backend is the core -- would need to analyze Go code for schema/API changes.

### 14. Outline

- **URL:** https://github.com/outline/outline
- **Language/Stack:** TypeScript (Node.js/Koa backend, React frontend), Sequelize ORM, PostgreSQL, Redis
- **Stars:** ~38,000
- **Default branch:** main
- **Schema quality:** Excellent. 37 Sequelize models: Document, Collection, Team, User, Comment, Event, Group, GroupMembership, GroupUser, Integration, Notification, Pin, Reaction, Revision, Share, Star, Subscription, Template, View, WebhookSubscription, WebhookDelivery, ApiKey, Attachment, AuthenticationProvider, FileOperation, Import, ImportTask, Emoji, ExternalGroup, Relationship, etc.
- **Has REST API:** Partial. Outline uses RPC-style API (POST documents.list, POST documents.create, etc.) rather than standard REST (GET /api/documents). ~34 endpoint groups covering documents, collections, users, teams, groups, shares, comments, events, integrations, etc. Fully featured but not strictly RESTful.
- **Squash-merges:** Mixed. 8/15 recent commits follow `(#number)` pattern. However, the main branch also has direct WIP commits ("wip", "test", "Styling finetuning") and merge commits. Maintainer appears to push directly to main sometimes.
- **Fit:** Medium
- **Concerns:** RPC-style API (not REST). Direct pushes to main by maintainer with WIP commits pollute history. Mixed merge strategy. Despite excellent TypeScript codebase with rich schema, the inconsistent commit practices are a concern.

### 15. Rocket.Chat

- **URL:** https://github.com/RocketChat/Rocket.Chat
- **Language/Stack:** TypeScript (Meteor/Node.js), MongoDB
- **Stars:** ~45,100
- **Default branch:** develop
- **Schema quality:** Good. MongoDB collections for users, rooms, messages, subscriptions, settings, roles, permissions, integrations, uploads, etc. Schema is document-oriented (MongoDB), not relational. The codebase uses TypeScript interfaces/types to define document shapes.
- **Has REST API:** Yes. Comprehensive REST API with endpoint groups: channels, groups, teams, users, chat/messages, authentication, roles, permissions, integrations, subscriptions, etc. Well-documented at developer.rocket.chat.
- **Squash-merges:** Yes. 14/15 recent commits follow `(#number)` pattern (e.g., "chore: change date formatting in Omnichannel Contact Center (#40025)"). Only exception is a branch merge commit.
- **Fit:** Medium-High
- **Concerns:** Default branch is `develop` not `main`. MongoDB (not SQL). Very large monorepo. Uses Meteor framework which is somewhat niche. TypeScript throughout though.

---

## Summary / Rankings

### Top Picks (High Fit)

| Rank | Repo | Stars | Lang | Schema | API | Squash | Key Strength |
|------|------|-------|------|--------|-----|--------|-------------|
| 1 | **Unleash** | 13.3k | TypeScript | Good (Knex/PG) | Yes (REST) | Yes (100%) | Clean TS, perfect squash, well-structured API |
| 2 | **GrowthBook** | 7.6k | TypeScript | Excellent (74 models) | Yes (40 routers) | Yes (100%) | Rich schema, many endpoints, clean history |
| 3 | **Directus** | 34.7k | TypeScript | Excellent (101 migrations) | Yes (REST+GraphQL) | Yes (100%) | Rich system schema, auto-generated APIs |
| 4 | **Ghost** | 52.4k | JavaScript | Excellent (76 tables) | Yes (REST) | Mostly (90%) | Richest schema, mature project |

### Good Candidates (Medium-High Fit)

| Rank | Repo | Stars | Lang | Schema | API | Squash | Key Concern |
|------|------|-------|------|--------|-----|--------|-------------|
| 5 | **Coolify** | 52.6k | PHP | Excellent (54 models) | Yes (REST) | Mostly | PHP/Laravel (not JS/TS) |
| 6 | **Mattermost** | 36.1k | Go+TS | Excellent (43 stores) | Yes (42 handlers) | Yes (100%) | Go backend |
| 7 | **Rocket.Chat** | 45.1k | TypeScript | Good (MongoDB) | Yes (REST) | Yes (93%) | develop branch, MongoDB, Meteor |
| 8 | **Portainer** | 37.1k | Go+TS | Good | Yes (REST) | Yes (100%) | Go backend |

### Lower Fit

| Repo | Stars | Reason for Lower Fit |
|------|-------|---------------------|
| **Hoppscotch** | 78.8k | Only 23 models, it's an API tool not a product with rich domain |
| **Appsmith** | 39.5k | Java backend, MongoDB, `release` branch |
| **ToolJet** | 37.7k | Mixed merge strategy, `develop` branch |
| **Payload** | 41.6k | Framework not app, config-driven schema |
| **Strapi** | 71.8k | Mixed merge strategy, `develop` branch, framework |
| **NocoDB** | 62.6k | Fails squash-merge criteria hard |
| **Outline** | 38.0k | RPC not REST, WIP commits on main |


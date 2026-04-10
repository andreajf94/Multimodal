# Go Web App Repository Research

Research goal: Find deployed Go web apps (not frameworks) with real DB schemas, REST APIs, squash-merge workflow, 1000+ stars, actively maintained.

---

## 1. Gitea

- **GitHub URL:** https://github.com/go-gitea/gitea
- **Stars:** ~54,700
- **ORM/DB layer:** xorm (xorm.io/xorm v1.3.11) with support for MySQL, PostgreSQL, SQLite, SQL Server
- **Schema quality:** Excellent. Large `models/` directory with 26+ subdirectories (issues, repo, user, org, auth, migrations, packages, webhook, etc.). Full migration system in `models/migrations/`.
- **Has REST API:** Yes. Full REST API at `routers/api/v1/` with 12+ endpoint groups: admin, org, repo, user, packages, notify, settings, activitypub, etc.
- **Squash-merges:** Yes. Recent commits all follow `"Description (#NNNNN)"` pattern (e.g. "#36701", "#37026", "#37085").
- **Fit:** HIGH
- **Concerns:** Very large codebase, complex. xorm is less common than GORM but is a real ORM with struct-based models. Excellent candidate overall.

---

## 2. Forgejo

- **GitHub URL:** Not on GitHub. Hosted at https://codeberg.org/forgejo/forgejo
- **Stars:** ~4,000 (on Codeberg)
- **ORM/DB layer:** xorm (fork of Gitea, same stack). Supports SQLite, MySQL, MariaDB, PostgreSQL.
- **Schema quality:** Same as Gitea (it is a fork). Excellent models directory.
- **Has REST API:** Yes (inherited from Gitea, same API structure).
- **Squash-merges:** Unclear. Hosted on Codeberg, not GitHub. Commit history was not scrapable.
- **Fit:** LOW
- **Concerns:** Not on GitHub (Codeberg only), only ~4k stars (below threshold if we want strong signal). Being a Gitea fork, it doesn't add much value beyond Gitea itself. Skip in favor of Gitea.

---

## 3. Vikunja

- **GitHub URL:** https://github.com/go-vikunja/vikunja
- **Stars:** ~3,800
- **ORM/DB layer:** xorm (xorm.io/xorm v1.3.11) + xormigrate for migrations. Supports MySQL, PostgreSQL, SQLite.
- **Schema quality:** Excellent. 120 files in `pkg/models/` covering tasks, projects, teams, labels, permissions, webhooks, API tokens, notifications, etc. ~60 core model files + ~60 test files.
- **Has REST API:** Yes. Full task/project management API with comprehensive routes.
- **Squash-merges:** Yes. Recent commits show `"description (#NNNN)"` pattern (e.g. "#2542", "#2490", "#2462").
- **Fit:** MEDIUM
- **Concerns:** Only ~3,800 stars (meets 1000+ threshold but on the lower side). Uses xorm like Gitea. Good schema complexity and clean PR workflow. Monorepo with frontend (Vue/TypeScript) included.

---

## 4. Miniflux

- **GitHub URL:** https://github.com/miniflux/v2
- **Stars:** ~9,000
- **ORM/DB layer:** Raw `database/sql` with `github.com/lib/pq` PostgreSQL driver. No ORM (no GORM, xorm, sqlx, ent).
- **Schema quality:** Unknown (no ORM struct models). Uses direct SQL queries against PostgreSQL.
- **Has REST API:** Yes. Minimalist feed reader with API.
- **Squash-merges:** No. Commits do not have `(#NNN)` PR references. Uses conventional commit format (`fix:`, `feat:`, etc.) but appears to merge commits directly.
- **Fit:** LOW
- **Concerns:** No ORM -- uses raw SQL only. Does not squash-merge. Minimalist app with limited schema complexity. Not a good fit.

---

## 5. Gotify

- **GitHub URL:** https://github.com/gotify/server
- **Stars:** ~14,800
- **ORM/DB layer:** GORM (gorm.io/gorm v1.31.1) with MySQL, PostgreSQL, SQLite drivers.
- **Schema quality:** Small but clean. `model/` has ~9 files: application, client, message, user, plugin_conf, health, paging, version, error. Relatively simple domain.
- **Has REST API:** Yes. `api/` directory has 8+ handler files: application, client, message, user, plugin, tokens, health, stream.
- **Squash-merges:** Yes. Recent commits show `"description (#NNN)"` pattern (e.g. "#939", "#913", "#926"). Some merge commits also present.
- **Fit:** MEDIUM
- **Concerns:** Uses GORM (ideal). Schema is relatively small/simple (notification server). Mixed merge strategy (some squash, some merge commits). Good but limited schema depth.

---

## 6. Woodpecker CI

- **GitHub URL:** https://github.com/woodpecker-ci/woodpecker
- **Stars:** ~6,800
- **ORM/DB layer:** xorm (xorm.io/xorm v1.3.11) + xormigrate. Supports MySQL, PostgreSQL, SQLite.
- **Schema quality:** Good. `server/model/` has 33 files covering agents, users, repos, pipelines, steps, tasks, workflows, commits, PRs, teams, orgs, secrets, registries, configs, crons, environments, events, feeds, forges, logs, permissions, queue, redirections, server config.
- **Has REST API:** Yes. CI/CD server with comprehensive API for managing pipelines, repos, users, etc.
- **Squash-merges:** Yes. Recent commits show `"description (#NNNN)"` pattern (e.g. "#6389", "#6387", "#6369").
- **Fit:** HIGH
- **Concerns:** Uses xorm (not GORM). Mostly dependency update commits visible in recent history. Good schema depth for a CI/CD system. Vue frontend included in monorepo.

---

## 7. Memos

- **GitHub URL:** https://github.com/usememos/memos
- **Stars:** ~58,600
- **ORM/DB layer:** Raw `database/sql` with MySQL, PostgreSQL, SQLite drivers. No ORM framework. Custom store layer with driver abstraction.
- **Schema quality:** Moderate. `store/` has 14 Go files covering memos, users, attachments, reactions, memo_relations, memo_shares, inbox, idp, instance_settings, user_settings. Plus migration, cache, seed directories.
- **Has REST API:** Yes. Extensive API at `server/router/api/v1/` with 35 files covering auth, memos, attachments, users, ACL, reactions, shares, relations, shortcuts, identity providers, SSE, health, instance.
- **Squash-merges:** Mixed. Some commits have `(#NNN)` pattern, others are direct commits without PR references. Inconsistent workflow.
- **Fit:** MEDIUM
- **Concerns:** No ORM (raw SQL with custom store pattern). Very high star count (58.6k). Mixed merge strategy. Schema is moderate complexity. The lack of ORM means no struct-based model definitions with tags -- harder to extract schema info.

---

## 8. Listmonk

- **GitHub URL:** https://github.com/knadh/listmonk
- **Stars:** ~19,500
- **ORM/DB layer:** sqlx (github.com/jmoiron/sqlx v1.4.0) with PostgreSQL only (lib/pq).
- **Schema quality:** Good. `models/` has 9 files: campaigns, lists, subscribers, messages, templates, bounces, settings, queries, common. Clean domain model for a mailing list manager. SQL queries in dedicated `queries/` directory.
- **Has REST API:** Yes. `cmd/` has 26 handler files covering campaigns, lists, subscribers, templates, bounces, media, auth, roles, users, settings, import, events, maintenance, etc.
- **Squash-merges:** Yes. Most recent commits show `(#NNNN)` pattern (e.g. "#2984", "#2979", "#2977", "#2973"). Some direct commits also present with "Closes #NNNN" style.
- **Fit:** HIGH
- **Concerns:** Uses sqlx (good, real DB library). PostgreSQL only. Clean, well-structured codebase. Good star count. Schema is moderate complexity (mailing list domain). Vue frontend in monorepo.

---

## 9. Zitadel

- **GitHub URL:** https://github.com/zitadel/zitadel
- **Stars:** ~13,400
- **ORM/DB layer:** GORM (jinzhu/gorm v1.9.16) + pgx v5 PostgreSQL driver + Squirrel query builder + tern migrations. PostgreSQL only.
- **Schema quality:** Excellent. `internal/query/` has 194 files covering users, auth requests, OIDC, SAML, organizations, domains, permissions, projects, policies (login, lockout, password, domain, label, privacy, notification, security), applications, IDPs, actions, quotas, messaging, milestones, events, system features, and much more. Extremely deep IAM domain model.
- **Has REST API:** Yes. `internal/api/` has 13 subdirectories: OIDC, SAML, SCIM, gRPC, HTTP, IDP, authz, info, service, UI, assets, call, robots_txt. Full identity platform API.
- **Squash-merges:** Yes. All recent commits show `"description (#NNNNN)"` pattern (e.g. "#11868", "#11968", "#11837", "#11888").
- **Fit:** HIGH
- **Concerns:** Uses older jinzhu/gorm v1 (not gorm.io/gorm v2). Very large, complex codebase. Event-sourced architecture (CQRS) means models are split between command/query sides. Heavy use of gRPC alongside REST. PostgreSQL only. Excellent schema depth but complex architecture.

---

## 10. Casdoor

- **GitHub URL:** https://github.com/casdoor/casdoor
- **Stars:** ~13,300
- **ORM/DB layer:** xorm (github.com/xorm-io/xorm v1.1.6) with MySQL, PostgreSQL, SQLite, SQL Server drivers.
- **Schema quality:** Excellent. `object/` has 147 files covering users, auth (MFA, WebAuthn, SAML, Kerberos, LDAP), roles, permissions, organizations, groups, applications, providers, tokens, certs, keys, orders, payments, subscriptions, products, pricing, webhooks, sessions, sync, resources, storage, notifications, email, SMS, and more.
- **Has REST API:** Yes. `controllers/` has 72 files covering all IAM operations plus payment/order system, SCIM, Casbin, OAuth DCR, OIDC discovery, etc.
- **Squash-merges:** Mixed. Some commits have `(#NNNN)` PR references, others are direct commits without PR numbers. Not consistently squash-merged.
- **Fit:** MEDIUM-HIGH
- **Concerns:** Uses xorm (not GORM). Inconsistent merge strategy (some squash, some direct). Very broad feature set (IAM + payments + MCP server). Code quality may be uneven given the pace of development (v2.393.0 suggests very frequent releases). Excellent schema breadth though.

---

## Summary Table

| Repo | Stars | ORM/DB | Schema Depth | REST API | Squash-Merge | Fit |
|------|-------|--------|-------------|----------|-------------|-----|
| **Gitea** | 54.7k | xorm | Excellent (26+ model dirs) | Yes (12+ groups) | Yes | HIGH |
| **Forgejo** | 4k | xorm (Gitea fork) | Excellent | Yes | Unclear | LOW |
| **Vikunja** | 3.8k | xorm | Excellent (120 files) | Yes | Yes | MEDIUM |
| **Miniflux** | 9k | raw SQL (lib/pq) | Unknown | Yes | No | LOW |
| **Gotify** | 14.8k | GORM v2 | Small (9 files) | Yes (8 handlers) | Yes (mixed) | MEDIUM |
| **Woodpecker CI** | 6.8k | xorm | Good (33 files) | Yes | Yes | HIGH |
| **Memos** | 58.6k | raw SQL | Moderate (14 files) | Yes (35 files) | Mixed | MEDIUM |
| **Listmonk** | 19.5k | sqlx | Good (9 models + SQL) | Yes (26 handlers) | Yes | HIGH |
| **Zitadel** | 13.4k | GORM v1 + pgx | Excellent (194 files) | Yes (13 API dirs) | Yes | HIGH |
| **Casdoor** | 13.3k | xorm | Excellent (147 files) | Yes (72 controllers) | Mixed | MEDIUM-HIGH |

## Top Recommendations

1. **Gitea** -- Best overall. Huge schema, clean squash-merge workflow, xorm ORM with struct models, 54.7k stars. Only concern is codebase size.
2. **Listmonk** -- Clean, focused app. sqlx with PostgreSQL, good schema, consistent squash-merges, 19.5k stars. Moderate schema depth but very clean codebase.
3. **Zitadel** -- Deepest schema (194 query files). GORM + pgx, consistent squash-merges, 13.4k stars. Complex event-sourced architecture may complicate extraction.
4. **Woodpecker CI** -- Good CI/CD domain. xorm, 33 model files, consistent squash-merges, 6.8k stars. Solid mid-complexity option.
5. **Casdoor** -- Broadest feature set (147 object files, 72 controllers). xorm. Inconsistent merge strategy is the main drawback.

## ORM Distribution

- **xorm:** Gitea, Forgejo, Vikunja, Woodpecker CI, Casdoor (5 repos)
- **GORM:** Gotify (v2), Zitadel (v1)
- **sqlx:** Listmonk
- **raw SQL:** Miniflux, Memos

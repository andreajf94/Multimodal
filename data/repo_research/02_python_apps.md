# Python Web App Repos — Research for Commit Pair Mining

Research date: 2026-04-05

Criteria:
1. Real deployed web app (not framework/library)
2. SQLAlchemy, Django ORM, or similar with real schema (5+ models)
3. Real REST API (FastAPI, DRF, Flask)
4. Squash-merges PRs (recent commits on main have "(#123)" style)
5. Open source, 500+ stars, actively maintained

---

## 1. NetBox (netbox-community/netbox)

- **GitHub**: https://github.com/netbox-community/netbox
- **Stars**: 20,164
- **Language**: Python (Django)
- **ORM**: Django ORM
- **Schema model count**: Very large. 17+ Django apps (dcim, ipam, circuits, tenancy, virtualization, vpn, wireless, etc.), each with multiple models. Estimated 100+ models.
- **REST API**: Yes. Dedicated `netbox/netbox/api/` directory. Uses Django REST Framework.
- **Squash-merges**: Mixed. Some commits are squash-merged with `(#123)` pattern (e.g. "Fixes #21542: ... (#21834)"), but also some merge commits ("Merge pull request #21837..."). Not purely squash-merge.
- **Fit**: HIGH
- **Concerns**: Mixed merge strategy (some merge commits, some squash). Very large codebase may require selective extraction. Extremely active.

---

## 2. authentik (goauthentik/authentik)

- **GitHub**: https://github.com/goauthentik/authentik
- **Stars**: 20,841
- **Language**: Python + Go
- **ORM**: Django ORM
- **Schema model count**: Large. 20+ Django apps (core, crypto, events, flows, policies, providers, rbac, sources, stages, etc.). Estimated 50+ models.
- **REST API**: Yes. Dedicated `authentik/api/` directory with Django REST Framework.
- **Squash-merges**: Yes. All recent commits follow `description (#NNNNN)` pattern consistently (e.g. "tests: refactor test harness to split apart a single file (#21391)").
- **Fit**: HIGH
- **Concerns**: Mixed language (Go + Python). Identity/auth domain might have limited schema diversity. Very active.

---

## 3. saleor (saleor/saleor)

- **GitHub**: https://github.com/saleor/saleor
- **Stars**: 22,769
- **Language**: Python (Django)
- **ORM**: Django ORM
- **Schema model count**: Very large. 25+ Django apps (account, checkout, order, product, warehouse, payment, discount, shipping, etc.). Estimated 80+ models.
- **REST API**: NO — uses GraphQL exclusively. Primary endpoint is `/graphql/`.
- **Squash-merges**: Yes. All recent commits are squash-merged with `(#NNNNN)` pattern.
- **Fit**: LOW
- **Concerns**: No REST API. GraphQL-only. Does not meet criterion #3.

---

## 4. zulip (zulip/zulip)

- **GitHub**: https://github.com/zulip/zulip
- **Stars**: 24,991
- **Language**: Python (Django)
- **ORM**: Django ORM
- **Schema model count**: Large. 30 model files in `zerver/models/` (messages, streams, users, realms, groups, drafts, presence, etc.). Estimated 40-60 models.
- **REST API**: Yes. Custom REST framework (not DRF). Uses `rest_path()` helper mapping HTTP verbs to view functions. Endpoints at `/api/v1/` and `/json/`. Well-documented OpenAPI spec in `zerver/openapi/`.
- **Squash-merges**: NO. Uses rebase-merge. 0 out of 30 recent commits have `(#NNN)` pattern. All single-parent commits with clean subject lines like "help: Clarify topic status icons".
- **Fit**: LOW
- **Concerns**: Does not squash-merge. Rebase workflow means commits don't map to PRs via subject line. Would need different extraction approach. Also not using DRF.

---

## 5. paperless-ngx (paperless-ngx/paperless-ngx)

- **GitHub**: https://github.com/paperless-ngx/paperless-ngx
- **Stars**: 37,871
- **Language**: Python (Django)
- **Default branch**: dev
- **ORM**: Django ORM
- **Schema model count**: 22 models in `documents/models.py` alone (Document, Correspondent, Tag, DocumentType, StoragePath, SavedView, CustomField, Workflow, Note, ShareLink, etc.). Plus models in `paperless_mail`, `paperless_ai`. Estimated 25-30 models total.
- **REST API**: Yes. Full Django REST Framework with ViewSets for all major models (CorrespondentViewSet, TagViewSet, DocumentViewSet, SavedViewViewSet, etc.).
- **Squash-merges**: Yes. All recent commits on `dev` branch follow `description (#NNNNN)` pattern consistently.
- **Fit**: HIGH
- **Concerns**: Default branch is `dev`, not `main`. Need to target correct branch. Very popular (37k stars), very active. Document management domain is well-suited (rich models, clear CRUD operations).

---

## 6. baserow (baserow/baserow)

- **GitHub**: https://github.com/baserow/baserow
- **Stars**: 4,566
- **Language**: Python (Django)
- **Default branch**: develop
- **ORM**: Django ORM
- **Schema model count**: Large. Core + 5 contrib apps (database, builder, automation, dashboard, integrations), each with their own models. Backend at `backend/src/baserow/`. Estimated 50+ models.
- **REST API**: Yes. Dedicated `api/` directory with DRF-style endpoints.
- **Squash-merges**: Yes. All recent commits on `develop` follow `type: description (#NNNN)` pattern consistently.
- **Fit**: HIGH
- **Concerns**: Default branch is `develop`. Stars slightly below 500+ threshold but well above at 4.5k. No-code database platform with complex schema. Good diversity of model types.

---

## 7. healthchecks (healthchecks/healthchecks)

- **GitHub**: https://github.com/healthchecks/healthchecks
- **Stars**: 9,969
- **Language**: Python (Django)
- **ORM**: Django ORM
- **Schema model count**: Small. 6 models in `hc/api/models.py` (Check, Ping, Channel, Notification, Flip, TokenBucket) + 5 in `hc/accounts/models.py` (Profile, Project, Member, Credential). ~11 models total.
- **REST API**: Yes. Has `hc/api/` app.
- **Squash-merges**: NO. 0 out of 30 recent commits have `(#NNN)` pattern. Appears to be owner-maintained with direct commits to master.
- **Fit**: LOW
- **Concerns**: Does not squash-merge. Small model count (11 models, borderline for 5+ criterion but lacks complexity). Single-maintainer commit style.

---

## 8. taiga-back (taigaio/taiga-back)

- **GitHub**: https://github.com/taigaio/taiga-back
- **Stars**: 817
- **Language**: Python (Django)
- **ORM**: Django ORM
- **Schema model count**: Large. 20+ Django apps including projects with sub-apps (epics, issues, userstories, tasks, milestones, wiki). Estimated 40+ models.
- **REST API**: Yes. Has `routers.py` and DRF-style ViewSets.
- **Squash-merges**: NO. Uses merge commits. 9 out of 30 recent commits are "Merge pull request..." style. 0 have `(#NNN)` squash pattern.
- **Fit**: LOW
- **Concerns**: Does not squash-merge (uses merge commits). Low star count (817). Low activity (last commits are sparse). Project management app has good domain complexity though.

---

## 9. flagsmith (Flagsmith/flagsmith)

- **GitHub**: https://github.com/Flagsmith/flagsmith
- **Stars**: 6,290
- **Language**: Python (Django)
- **ORM**: Django ORM
- **Schema model count**: Medium-large. 20+ Django apps (features, environments, organisations, projects, segments, audit, integrations, etc.). 4 models in features, 3 in environments, 11 in organisations alone. Estimated 30-40 models total.
- **REST API**: Yes. Full DRF with ViewSets across all apps.
- **Squash-merges**: Yes. All recent commits follow `type: description (#NNNN)` pattern consistently (e.g. "feat: implement-gram-elements (#7049)").
- **Fit**: HIGH
- **Concerns**: Feature flag domain is somewhat specialized. Good model count and API coverage. Well-structured codebase.

---

## 10. AWX (ansible/awx)

- **GitHub**: https://github.com/ansible/awx
- **Stars**: 15,355
- **Language**: Python (Django)
- **Default branch**: devel
- **ORM**: Django ORM
- **Schema model count**: Very large. 22 model files in `awx/main/models/` (jobs, inventory, credential, projects, workflow, notifications, organization, schedules, etc.). Estimated 50-80 models.
- **REST API**: Yes. Full DRF in `awx/api/` with serializers, views, permissions, pagination, authentication.
- **Squash-merges**: Yes. 19 out of 20 recent commits follow `description (#NNNNN)` pattern consistently.
- **Fit**: HIGH
- **Concerns**: Default branch is `devel`. Ansible/automation domain. Very large codebase. Some commits reference internal Jira tickets (e.g. "AAP-12516"). Enterprise-oriented.

---

## 11. PostHog (PostHog/posthog)

- **GitHub**: https://github.com/PostHog/posthog
- **Stars**: 32,415
- **Language**: Python (Django)
- **Default branch**: master
- **ORM**: Django ORM
- **Schema model count**: Massive. 67 model files + 36 model subdirectories in `posthog/models/`. Estimated 100+ models.
- **REST API**: Yes. 112+ API endpoint files in `posthog/api/` using Django REST Framework.
- **Squash-merges**: Yes. 20 out of 20 recent commits follow `type(scope): description (#NNNNN)` pattern perfectly.
- **Fit**: HIGH
- **Concerns**: Very large codebase. Product analytics domain is complex but interesting. Extremely active (PR numbers in 53000+ range). May need selective extraction.

---

## 12. Sentry (getsentry/sentry)

- **GitHub**: https://github.com/getsentry/sentry
- **Stars**: 43,505
- **Language**: Python (Django)
- **Default branch**: master
- **ORM**: Django ORM
- **Schema model count**: Massive. 120+ model files in `src/sentry/models/` (organization, project, group, event, team, user, release, repository, dashboard, rule, etc.). Estimated 150+ models.
- **REST API**: Yes. Large API in `src/sentry/api/` with endpoints, serializers, validators, fields, bases directories. DRF-style architecture.
- **Squash-merges**: Yes. 20 out of 20 recent commits follow `type(scope): description (#NNNNNN)` pattern perfectly (PR numbers in 112000+ range).
- **Fit**: HIGH
- **Concerns**: Extremely large codebase. Error monitoring domain. Very high commit velocity. PR numbers suggest massive history. May be too large for practical extraction without careful scoping.

---

## 13. Label Studio (HumanSignal/label-studio)

- **GitHub**: https://github.com/HumanSignal/label-studio
- **Stars**: 26,935
- **Language**: TypeScript (but Python backend)
- **Default branch**: develop
- **ORM**: Django ORM
- **Schema model count**: Medium. Django apps include tasks (7 models), projects, organizations, users, data_import, data_export, ml, webhooks, io_storages. Estimated 30-40 models across all apps.
- **REST API**: Yes. DRF-based API across apps.
- **Squash-merges**: Mostly. 17 out of 20 recent commits have `(#NNNN)` pattern. 3 are automated CI commits ("ci: Update Feature Flags").
- **Fit**: HIGH
- **Concerns**: Primary language listed as TypeScript (frontend heavy). Default branch is `develop`. ML/annotation domain. Backend is solidly Django+DRF though.

---

## 14. InvenTree (inventree/InvenTree)

- **GitHub**: https://github.com/inventree/InvenTree
- **Stars**: 6,800
- **Language**: Python (Django)
- **Default branch**: master
- **ORM**: Django ORM
- **Schema model count**: Large. 15+ Django apps (part, stock, order, company, build, common, users, plugin, report, etc.) in `src/backend/InvenTree/`. Estimated 40-60 models.
- **REST API**: Yes. Full DRF with comprehensive REST API documented in `docs/docs/api/`.
- **Squash-merges**: Yes. All recent commits follow `description (#NNNNN)` pattern consistently.
- **Fit**: HIGH
- **Concerns**: Inventory management domain is well-suited (rich entity relationships). Good size -- large enough to be interesting, not overwhelming. Backend at `src/backend/InvenTree/`.

---

## 15. Mealie (mealie-recipes/mealie)

- **GitHub**: https://github.com/mealie-recipes/mealie
- **Stars**: 11,900
- **Language**: Python (FastAPI + SQLAlchemy)
- **Default branch**: mealie-next
- **ORM**: SQLAlchemy (NOT Django)
- **Schema model count**: Medium. Models in `mealie/db/models/` organized into subdirectories: recipe/ (15 model files), group/, household/, users/, server/, labels. Estimated 30-40 models.
- **REST API**: Yes. FastAPI-based REST API.
- **Squash-merges**: Yes. All recent commits follow `description (#NNNN)` pattern consistently.
- **Fit**: HIGH
- **Concerns**: Uses FastAPI+SQLAlchemy (not Django) -- provides ORM diversity. Default branch is `mealie-next`. Recipe management domain is well-suited (clear entities, good CRUD patterns). Good complement to Django-heavy list.

---

## Also investigated but rejected:

| Repo | Stars | Why rejected |
|------|-------|-------------|
| **saleor** | 22.8k | GraphQL-only, no REST API |
| **zulip** | 25k | Rebase-merge, not squash; no DRF |
| **healthchecks** | 10k | No squash-merge; single-maintainer direct commits |
| **taiga-back** | 817 | Merge commits, not squash; low activity |
| **allegro/ralph** | 2.5k | Merge commits, not squash |
| **ArchiveBox** | 27.2k | No squash-merge; inconsistent commits |
| **Netflix/dispatch** | 6.4k | Archived (Sept 2025), read-only |
| **LibrePhotos** | 8k | Uses merge commits, not squash |
| **TandoorRecipes** | 8.1k | Mixed merge strategy, not consistent squash |
| **WeblateOrg/weblate** | 5.8k | Mixed commits (many automated translation commits) |
| **nautobot** | 1.5k | Mixed strategy (some squash, some merge commits) |
| **makeplane/plane** | 47k | TypeScript primary; non-standard PR refs in commits |

---

## Summary — Top candidates (all criteria met)

| Rank | Repo | Stars | ORM | Models | Squash | Domain |
|------|------|-------|-----|--------|--------|--------|
| 1 | **paperless-ngx** | 37.9k | Django | 25-30 | Yes | Document management |
| 2 | **PostHog** | 32.4k | Django | 100+ | Yes | Product analytics |
| 3 | **Sentry** | 43.5k | Django | 150+ | Yes | Error monitoring |
| 4 | **Label Studio** | 26.9k | Django | 30-40 | Mostly | ML annotation |
| 5 | **authentik** | 20.8k | Django | 50+ | Yes | Identity/auth |
| 6 | **NetBox** | 20.2k | Django | 100+ | Mixed | Infrastructure mgmt |
| 7 | **AWX** | 15.4k | Django | 50-80 | Yes | Automation |
| 8 | **Mealie** | 11.9k | SQLAlchemy | 30-40 | Yes | Recipe management |
| 9 | **InvenTree** | 6.8k | Django | 40-60 | Yes | Inventory mgmt |
| 10 | **Flagsmith** | 6.3k | Django | 30-40 | Yes | Feature flags |
| 11 | **Baserow** | 4.6k | Django | 50+ | Yes | No-code database |

**Best bets for cleanest extraction**: paperless-ngx, authentik, Flagsmith, InvenTree, Mealie, Baserow (right-sized, consistent squash-merge, clear domain models).

**Highest volume but may need scoping**: Sentry, PostHog, AWX (very large codebases).

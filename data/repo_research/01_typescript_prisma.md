# TypeScript + Prisma Web App Repos — Research

**Goal:** Find deployed open-source web apps using Prisma ORM with real schemas, REST API routes, squash-merge workflow, 1000+ stars, actively maintained.

**Date:** 2026-04-05

---

## 1. cal.com

- **GitHub URL:** https://github.com/calcom/cal.com
- **Stars:** ~41,000
- **Prisma model count:** ~113 models (massive schema with Host, EventType, User, Booking, Team, Payment, Webhook, Workflow, etc.)
- **Has REST API:** Yes — dedicated `apps/api/v1/pages/api/` with 21+ resource directories (bookings, users, teams, event-types, payments, schedules, webhooks, etc.)
- **Squash-merges:** Yes — commit messages follow `feat: X (#123)`, `fix: Y (#456)` pattern consistently
- **Fit rating:** HIGH
- **Concerns:** Very large codebase (monorepo). Schema is enormous which is great for complexity but may make IR extraction heavy. Extremely active with frequent commits.

---

## 2. formbricks

- **GitHub URL:** https://github.com/formbricks/formbricks
- **Stars:** ~12,000
- **Prisma model count:** ~36 models (Survey, Response, Contact, Environment, Project, Organization, Membership, ApiKey, Webhook, Segment, etc.)
- **Has REST API:** Yes — `apps/web/app/api/` with v1, v2, v3 versioned APIs plus auth, billing/stripe-webhook, google-sheet routes
- **Squash-merges:** Yes — commit messages follow `fix: X (#123)`, `feat: Y (#456)`, `chore: Z (#789)` pattern
- **Fit rating:** HIGH
- **Concerns:** None significant. Good schema complexity, clear squash-merge workflow, versioned REST API. Solid candidate.

---

## 3. documenso

- **GitHub URL:** https://github.com/documenso/documenso
- **Stars:** ~12,500
- **Prisma model count:** ~47 models (User, Envelope, Recipient, Field, Signature, Organisation, Team, Webhook, DocumentAuditLog, BackgroundJob, etc.)
- **Has REST API:** Yes — public API v1 (ts-rest based REST), API v2 (tRPC with OpenAPI spec). Remix app has `api+/` routes. Separate `openpage-api` app exists.
- **Squash-merges:** Yes — commit messages follow `feat: X (#123)`, `fix: Y (#456)` pattern consistently
- **Fit rating:** HIGH
- **Concerns:** Recently migrated from Next.js to Remix. API v1 is REST (deprecated but maintained), v2 is tRPC-based. The REST API surface may be thinner than ideal since primary interaction is via tRPC.

---

## 4. rallly

- **GitHub URL:** https://github.com/lukevella/rallly
- **Stars:** ~5,000
- **Prisma model count:** ~28 models across 9 schema files (Poll, Participant, Option, Vote, Comment, PollView, User, Account, Session, Subscription, PaymentMethod, ScheduledEvent, Space, SpaceMember, Credential, CalendarConnection, License, InstanceSettings, etc.)
- **Has REST API:** Partially — has `apps/web/src/app/api/` with routes for stripe, storage, status, licensing, integrations, but heavily uses tRPC (`trpc/[trpc]` route). Not primarily REST.
- **Squash-merges:** No — commit messages use emoji gitmoji style (e.g., "lipstick Update empty state", "bug Fix screenshots package") without PR number suffixes. Does NOT appear to squash-merge.
- **Fit rating:** LOW
- **Concerns:** Does not squash-merge PRs. Relatively small schema (~28 models). API is primarily tRPC, not REST. Below threshold on multiple criteria.

---

## 5. papermark

- **GitHub URL:** https://github.com/mfts/papermark
- **Stars:** ~8,100
- **Prisma model count:** ~32 models (User, Brand, Domain, View, Viewer, Document-related models, Agreement, Webhook, Tag, etc.) — schema split into subdirectory
- **Has REST API:** Yes — `pages/api/` (Next.js Pages Router) with 17+ route directories (teams, links, analytics, file, conversations, stripe, webhooks, etc.) plus individual route files for tracking
- **Squash-merges:** No/Unclear — recent commits show mix of "Merge pull request #XXX" (merge commits, not squash) and direct commits without PR numbers. Does NOT squash-merge.
- **Fit rating:** MEDIUM
- **Concerns:** Does not consistently squash-merge (uses merge commits). Good REST API and decent schema size. The merge-commit workflow is the main disqualifier.

---

## 6. inbox-zero

- **GitHub URL:** https://github.com/elie222/inbox-zero
- **Stars:** ~10,400
- **Prisma model count:** ~58 models (User, EmailAccount, Organization, Rule, Action, ExecutedRule, Group, Newsletter, ColdEmail, EmailMessage, Knowledge, ReplyMemory, Chat, CalendarConnection, MessagingChannel, AutomationJob, etc.)
- **Has REST API:** Yes — `apps/web/app/api/` with 37+ route directories (ai, chat, messages, threads, labels, automation-jobs, google, outlook, slack, telegram, stripe, v1, etc.)
- **Squash-merges:** Unclear/Mixed — some commits have PR numbers (e.g., "messaging: escape invalid Slack angle brackets (#2175)") but lack conventional prefix. Others lack PR numbers entirely (e.g., "update marketing repo"). Pattern is inconsistent.
- **Fit rating:** MEDIUM-HIGH
- **Concerns:** Squash-merge pattern is inconsistent — some PRs are squash-merged but commit messages do not follow strict conventional commit format. Schema is excellent (58 models). REST API is very comprehensive. Worth considering despite imperfect merge discipline.

---

## 7. openstatus

- **GitHub URL:** https://github.com/openstatusHQ/openstatus
- **Stars:** ~8,500
- **Prisma model count:** N/A — Does NOT use Prisma. Uses **Drizzle ORM** with SQLite/Turso.
- **Has REST API:** Yes (based on commit messages referencing API work)
- **Squash-merges:** Yes — commit messages follow `fix: X (#123)`, `feat: Y (#456)`, `chore: Z (#789)` pattern
- **Fit rating:** DISQUALIFIED
- **Concerns:** Does not use Prisma ORM at all. Uses Drizzle ORM. Fails the primary criterion.

---

## 8. typebot.io

- **GitHub URL:** https://github.com/baptisteArno/typebot.io
- **Stars:** ~9,800
- **Prisma model count:** ~31 models (User, Workspace, Space, Typebot, PublicTypebot, Result, Answer, AnswerV2, Log, Webhook, ChatSession, Credentials, CustomDomain, etc.) — separate PostgreSQL and MySQL schemas
- **Has REST API:** Partially — `apps/builder/src/app/api/` has limited routes (auth, v2 sessions streaming). Also has ORPC routes. Not a traditional REST API; more RPC-oriented.
- **Squash-merges:** No — commit messages use emoji gitmoji style (e.g., "wrench Fix cleanArchivedData script performance", "bug Fix robots.txt") without PR number suffixes. Does NOT squash-merge.
- **Fit rating:** LOW
- **Concerns:** Does not squash-merge (gitmoji style, no PR numbers). API is not primarily REST. Schema is decent size but the workflow criteria are not met.

---

## 9. umami

- **GitHub URL:** https://github.com/umami-software/umami
- **Stars:** ~36,000
- **Default branch:** master
- **Prisma model count:** ~13 models (User, Session, Website, WebsiteEvent, EventData, SessionData, Team, TeamUser, Report, Segment, Revenue, Link, Pixel)
- **Has REST API:** Yes — `src/app/api/` with directories for admin, auth, users, websites, teams, reports, realtime, links, pixels, etc.
- **Squash-merges:** No — recent commits show "Merge pull request #XXX" pattern (merge commits) and direct commits without conventional format. Does NOT squash-merge.
- **Fit rating:** LOW
- **Concerns:** Only 13 Prisma models (below the 5+ threshold but the schema is simple for a real app). Does NOT squash-merge. Merge-commit workflow. Schema may be too simple for our purposes despite high star count.

---

## 10. trigger.dev

- **GitHub URL:** https://github.com/triggerdotdev/trigger.dev
- **Stars:** ~14,400
- **Prisma model count:** ~95 models (User, Organization, Project, RuntimeEnvironment, BackgroundWorker, TaskRun, TaskRunAttempt, TaskQueue, BatchTaskRun, WorkerDeployment, TaskSchedule, ProjectAlert, FeatureFlag, LlmModel, etc.)
- **Has REST API:** Yes — Remix-based with extensive versioned API routes (api.v1.*, api.v2.*, api.v3.*) covering auth, runs, deployments, batches, schedules, tasks, environments, workers, realtime, etc.
- **Squash-merges:** Yes — commit messages follow `feat(scope): X (#123)`, `fix(scope): Y (#456)` pattern with scoped conventional commits
- **Fit rating:** HIGH
- **Concerns:** Uses Remix (not Next.js), so API routes are Remix file-based routes rather than Next.js app/api/ pattern. Extremely feature-rich schema. The Remix route pattern is slightly different but still constitutes REST API routes.

---

## Summary Table

| Repo | Stars | Prisma Models | REST API | Squash-Merge | Fit |
|------|-------|--------------|----------|-------------|-----|
| cal.com | 41k | ~113 | Yes (dedicated API app) | Yes | **HIGH** |
| formbricks | 12k | ~36 | Yes (versioned v1/v2/v3) | Yes | **HIGH** |
| documenso | 12.5k | ~47 | Yes (ts-rest v1 + tRPC v2) | Yes | **HIGH** |
| trigger.dev | 14.4k | ~95 | Yes (Remix, v1/v2/v3) | Yes | **HIGH** |
| inbox-zero | 10.4k | ~58 | Yes (37+ route dirs) | Mixed | **MEDIUM-HIGH** |
| papermark | 8.1k | ~32 | Yes (pages/api) | No (merge commits) | **MEDIUM** |
| rallly | 5k | ~28 | Partial (mostly tRPC) | No (gitmoji) | **LOW** |
| typebot.io | 9.8k | ~31 | Partial (mostly RPC) | No (gitmoji) | **LOW** |
| umami | 36k | ~13 | Yes | No (merge commits) | **LOW** |
| openstatus | 8.5k | N/A (Drizzle) | Yes | Yes | **DISQUALIFIED** |

## Top Recommendations

1. **cal.com** — Best overall: massive Prisma schema, dedicated REST API, perfect squash-merge discipline, very high stars
2. **trigger.dev** — Excellent: 95 Prisma models, versioned REST APIs, clean squash-merge, actively maintained
3. **formbricks** — Strong: 36 models, versioned REST API (v1/v2/v3), clean squash-merge workflow
4. **documenso** — Strong: 47 models, public REST API (v1), clean squash-merge. Minor concern: REST API is deprecated in favor of tRPC v2
5. **inbox-zero** — Promising: 58 models, comprehensive REST API, but squash-merge discipline is inconsistent

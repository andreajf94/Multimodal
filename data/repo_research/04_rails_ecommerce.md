# Rails & E-Commerce Repo Research

Research date: 2026-04-05
Goal: Find deployed web apps with rich DB schemas, REST APIs, squash-merge workflow, 1000+ stars.

---

## 1. discourse/discourse

- **URL:** https://github.com/discourse/discourse
- **Language/Stack:** Ruby on Rails
- **Stars:** ~46,700
- **Default branch:** main
- **Schema richness:** Extremely rich. ~280 model files in app/models. Hundreds of migrations (97+ visible, truncated). Models include User, Post, Topic, Category, Group, Badge, Notification, Theme, Upload, Webhook, ReviewableUser, SiteSetting, etc.
- **REST API:** Yes. Routes define JSON-format resources for posts, bookmarks, notifications, badges, categories, users, invites, etc.
- **Squash-merges:** Yes. All 15 recent commits follow `TYPE: Description (#NNNNN)` pattern. Examples: "DEV: Restore tag-info model for frontend (#39101)", "FEATURE: AI-generated queries (#39095)".
- **Fit:** HIGH
- **Concerns:** Massive codebase (may be complex to parse). Plugin architecture adds models dynamically. Discourse uses a custom commit prefix convention (DEV:, FEATURE:, UX:, etc.) rather than conventional commits.

---

## 2. mastodon/mastodon

- **URL:** https://github.com/mastodon/mastodon
- **Language/Stack:** Ruby on Rails
- **Stars:** ~49,800
- **Default branch:** main
- **Schema richness:** Very rich. ~145 model files + 11 subdirectories. Models include Account, Status, User, Notification, Follow, MediaAttachment, CustomFilter, Poll, Report, DomainBlock, etc.
- **REST API:** Yes. Dedicated `config/routes/api.rb` file. Well-documented Mastodon API used by many third-party clients.
- **Squash-merges:** Yes. All 15 recent commits follow `Description (#NNNNN)` pattern. Examples: "Use partial to render settings/featured_tags (#36174)", "Profile redesign: Adds a Follows you badge (#38549)".
- **Fit:** HIGH
- **Concerns:** ActivityPub/federation complexity. Some models are federation-specific. The API is well-documented externally which is a plus.

---

## 3. chatwoot/chatwoot

- **URL:** https://github.com/chatwoot/chatwoot
- **Language/Stack:** Ruby on Rails
- **Stars:** ~28,200
- **Default branch:** develop (not main)
- **Schema richness:** Good. ~54 model files + 3 subdirectories (channel/, concerns/, integrations/). Models include User, Account, Conversation, Message, Contact, Inbox, Team, Campaign, AutomationRule, Webhook, Article, Portal, etc.
- **REST API:** Yes. Versioned API (v1, v2) with controllers for accounts, profiles, webhooks, notifications, plus nested resources under accounts/ and widget/.
- **Squash-merges:** Yes. All 15 recent commits follow conventional commits with PR numbers: "fix: description (#NNNNN)", "feat: description (#NNNNN)". Clean pattern.
- **Fit:** HIGH
- **Concerns:** Default branch is `develop` not `main`. 54 models is solid but not as rich as Discourse/Mastodon.

---

## 4. forem/forem (dev.to)

- **URL:** https://github.com/forem/forem
- **Language/Stack:** Ruby on Rails
- **Stars:** ~22,600
- **Default branch:** main
- **Schema richness:** Very rich. ~121 model files + 8 subdirectories. Models include User, Article, Comment, Poll, Listing, Follow, Mention, Reaction, Badge, Organization, Podcast, AuditLog, Notification, etc.
- **REST API:** Yes. Has a documented v1 API (OpenAPI spec via rswag gem). Endpoints at developers.forem.com/api.
- **Squash-merges:** Mostly yes. 13 of 15 recent commits have (#NNNNN) pattern. 2 commits lack PR references ("Style tweak", "Fix embed in comments in feed"), suggesting occasional direct pushes.
- **Fit:** HIGH
- **Concerns:** Occasional direct commits without PR references (minor). Project activity may have slowed compared to peak dev.to era.

---

## 5. solidusio/solidus

- **URL:** https://github.com/solidusio/solidus
- **Language/Stack:** Ruby on Rails (e-commerce engine)
- **Stars:** ~5,300
- **Default branch:** main
- **Schema richness:** Extremely rich. ~136 model files under core/app/models/spree/. Models include Order, Product, Variant, Payment, Shipment, StockLocation, StockItem, TaxRate, ReturnAuthorization, Refund, StoreCredit, Adjustment, Address, etc. Full e-commerce domain.
- **REST API:** Yes. 34 API controller files covering products, variants, orders, checkouts, payments, shipments, stock, users, promotions, etc.
- **Squash-merges:** No. Uses merge commits ("Merge pull request #NNNN from user/branch"). Only 2 of 15 recent commits are merge commits with PR refs; rest are individual branch commits visible in history.
- **Fit:** MEDIUM
- **Concerns:** Uses merge commits, not squash-merges. 5,300 stars is above threshold but lower than others. As an engine/gem, it's mounted into a host Rails app rather than being a standalone app.

---

## 6. spree/spree

- **URL:** https://github.com/spree/spree
- **Language/Stack:** Ruby on Rails (e-commerce platform)
- **Stars:** ~15,300
- **Default branch:** main
- **Schema richness:** Very rich. Full e-commerce domain with models for Product, Variant, Order, LineItem, Payment, Shipment, StockLocation, Taxonomy, Taxon, Address, Promotion, etc. (models under core/app/models/spree/). Comprehensive schema comparable to Solidus (Spree is the ancestor).
- **REST API:** Yes. Spree has a v2 Storefront API and v2 Platform API (both REST/JSON). API controllers in the api/ directory.
- **Squash-merges:** Mostly yes. 11 of 15 recent commits have (#NNNNN) pattern. Some commits without PR refs appear to be direct pushes (version bumps, small UI tweaks).
- **Fit:** HIGH
- **Concerns:** Some direct pushes without PR refs. Like Solidus, it's a Rails engine, but Spree 5 ships as a more complete platform with admin dashboard, API, and storefront.

---

## 7. saleor/saleor

- **URL:** https://github.com/saleor/saleor
- **Language/Stack:** Python / Django
- **Stars:** ~22,800
- **Default branch:** main
- **Schema richness:** Very rich. ~28 Django app modules (account, product, order, checkout, payment, shipping, warehouse, discount, giftcard, channel, attribute, etc.). Product module alone has 16 models, order module has 8, account has 10. Estimated 80-100+ total Django models.
- **REST API:** NO. Saleor is GraphQL-only. No REST API endpoints.
- **Squash-merges:** Yes. All 15 recent commits follow (#NNNNN) pattern. Examples: "Turn off deferred for fulfillment events (#19011)", "Fix deadlocks on checkout and order (#19001)".
- **Fit:** LOW
- **Concerns:** GraphQL-only -- no REST API. This is a disqualifying criterion if REST API is required.

---

## 8. medusajs/medusa

- **URL:** https://github.com/medusajs/medusa
- **Language/Stack:** TypeScript / Node.js
- **Stars:** ~32,600
- **Default branch:** develop
- **Schema richness:** Very rich. 35 separate modules with own data models: cart, order, product, customer, payment, fulfillment, inventory, pricing, promotion, region, sales-channel, stock-location, tax, etc. HTTP types directory shows 51 resource types. Comprehensive e-commerce domain.
- **REST API:** Yes. Full REST API with Admin (/admin) and Store (/store) endpoints. Well-documented at docs.medusajs.com/api/admin and /api/store.
- **Squash-merges:** Yes. All 15 recent commits follow (#NNNNN) pattern. Examples: "feat(http-types-generator): Add HTTP types generator (#14988)", "fix: use exponential backoff in Redis lock (#14954)".
- **Fit:** HIGH
- **Concerns:** Default branch is `develop`, not `main`. Monorepo structure with many packages. TypeScript, not a traditional ORM -- models are spread across module packages. Medusa v2 is a significant rewrite from v1.

---

## 9. vendure-ecommerce/vendure (vendurehq/vendure)

- **URL:** https://github.com/vendure-ecommerce/vendure
- **Language/Stack:** TypeScript / NestJS / TypeORM
- **Stars:** ~8,000
- **Default branch:** master
- **Schema richness:** Rich. ~44 entity directories under packages/core/src/entity/. Entities include Product, ProductVariant, Order, OrderLine, Payment, Fulfillment, Customer, Address, Channel, Collection, Facet, TaxRate, ShippingMethod, StockLevel, Promotion, etc.
- **REST API:** No (primarily). Vendure is GraphQL-first. REST endpoints can be added via plugins but are not built in. The core exposes Shop API and Admin API via GraphQL.
- **Squash-merges:** Yes. 13 of 15 recent commits have (#NNNNN) pattern. 2 automated docs commits lack PR refs.
- **Fit:** LOW-MEDIUM
- **Concerns:** GraphQL-first, no built-in REST API. 8,000 stars is above threshold. TypeORM entities are well-structured but the lack of REST is a significant gap.

---

## Summary Table

| Repo | Lang | Stars | Models | REST API | Squash | Fit |
|------|------|-------|--------|----------|--------|-----|
| discourse/discourse | Ruby/Rails | 46.7k | ~280 | Yes | Yes | HIGH |
| mastodon/mastodon | Ruby/Rails | 49.8k | ~145 | Yes | Yes | HIGH |
| chatwoot/chatwoot | Ruby/Rails | 28.2k | ~54 | Yes | Yes | HIGH |
| forem/forem | Ruby/Rails | 22.6k | ~121 | Yes | Mostly | HIGH |
| solidusio/solidus | Ruby/Rails | 5.3k | ~136 | Yes | No (merge) | MEDIUM |
| spree/spree | Ruby/Rails | 15.3k | ~100+ | Yes | Mostly | HIGH |
| saleor/saleor | Python/Django | 22.8k | ~80-100 | No (GraphQL) | Yes | LOW |
| medusajs/medusa | TypeScript | 32.6k | ~50 modules | Yes | Yes | HIGH |
| vendure/vendure | TypeScript/NestJS | 8.0k | ~44 | No (GraphQL) | Yes | LOW-MEDIUM |

## Top Picks (all criteria met)

1. **discourse/discourse** -- Best overall: massive schema, REST API, clean squash-merges, very active, huge star count.
2. **mastodon/mastodon** -- Excellent: rich schema, well-documented REST API, clean squash-merges, very popular.
3. **chatwoot/chatwoot** -- Strong: clean squash-merge workflow, REST API, good schema. Default branch is `develop`.
4. **forem/forem** -- Strong: rich schema, documented REST API, mostly squash-merges. Minor direct-push noise.
5. **medusajs/medusa** -- Strong for TS/Node: full REST API, rich modular schema, clean squash-merges. Default branch is `develop`.
6. **spree/spree** -- Good: rich e-commerce schema, REST API v2, mostly squash-merges. Some direct pushes.

## Disqualified / Lower Fit

- **saleor/saleor** -- GraphQL-only, no REST API.
- **vendure/vendure** -- GraphQL-first, no built-in REST.
- **solidusio/solidus** -- Uses merge commits (not squash), and is a mountable engine rather than standalone app.

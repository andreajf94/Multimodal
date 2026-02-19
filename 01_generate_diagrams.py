#!/usr/bin/env python3
"""
01_generate_diagrams.py

Generates synthetic architecture diagrams and their descriptions.
Uses the 'diagrams' library (pip install diagrams).
Also requires graphviz: sudo apt-get install graphviz

Generates ~500 diagrams across different architecture patterns.
Each diagram gets a JSON description with components, data flow, and analysis.
"""

import json
import os
import random
import itertools
from pathlib import Path

# ─── Check dependencies ───
try:
    from diagrams import Diagram, Cluster, Edge
    from diagrams.aws.compute import EC2, ECS, Lambda, ElasticBeanstalk
    from diagrams.aws.database import RDS, ElastiCache, DynamoDB, Redshift
    from diagrams.aws.network import ELB, CloudFront, Route53, APIGateway
    from diagrams.aws.integration import SQS, SNS, Kinesis
    from diagrams.aws.storage import S3
    from diagrams.aws.analytics import Elasticsearch as ESService
    from diagrams.aws.ml import Sagemaker
except ImportError:
    print("Install dependencies: pip install diagrams")
    print("Also: sudo apt-get install graphviz")
    exit(1)

OUTPUT_DIR = Path("data/diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTIONS_FILE = Path("data/raw/diagram_descriptions.jsonl")
DESCRIPTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ─── Architecture Templates ───

def web_app_basic(variant_id):
    """Basic web application: LB -> App Servers -> DB"""
    name = f"web_basic_{variant_id}"
    filename = str(OUTPUT_DIR / name)
    
    db_choices = ["PostgreSQL", "MySQL"]
    cache = random.choice([True, False])
    cdn = random.choice([True, False])
    num_app_servers = random.choice([2, 3, 4])
    db = random.choice(db_choices)
    
    with Diagram(f"Web Application", filename=filename, show=False, direction="LR"):
        if cdn:
            cf = CloudFront("CDN")
        
        dns = Route53("DNS")
        lb = ELB("Load Balancer")
        
        with Cluster("Application Tier"):
            apps = [EC2(f"App {i+1}") for i in range(num_app_servers)]
        
        with Cluster("Data Tier"):
            primary_db = RDS(db)
            if cache:
                redis = ElastiCache("Redis Cache")
        
        dns >> lb >> apps
        for app in apps:
            app >> primary_db
            if cache:
                app >> redis
        if cdn:
            cf >> lb
    
    description = {
        "id": name,
        "image_path": f"{filename}.png",
        "architecture_type": "web_application",
        "components": {
            "dns": "Route53 for DNS resolution",
            "cdn": "CloudFront CDN for static assets" if cdn else None,
            "load_balancer": "Application Load Balancer distributing traffic",
            "app_servers": f"{num_app_servers} EC2 instances running application code",
            "database": f"{db} as primary datastore",
            "cache": "Redis for caching frequently accessed data" if cache else None,
        },
        "data_flow": f"DNS resolves to {'CDN/LB' if cdn else 'Load Balancer'} -> distributes across {num_app_servers} app servers -> {'check Redis cache, on miss query' if cache else 'query'} {db}",
        "potential_bottlenecks": [
            "Single database instance is a bottleneck for write-heavy workloads",
            f"{'No caching layer - all reads hit the database' if not cache else 'Cache invalidation strategy needed'}",
            "No read replicas - all read and write traffic goes to one DB",
        ],
        "single_points_of_failure": [
            "Database (single instance)",
            "Load balancer (single, though AWS ALB is inherently HA)",
        ],
        "improvements": [
            "Add read replicas for the database",
            "Add auto-scaling group for app servers",
            f"{'Add a caching layer (Redis/Memcached)' if not cache else 'Implement cache-aside pattern with TTL'}",
            f"{'Add CDN for static content' if not cdn else 'Configure CDN cache headers properly'}",
        ],
        "qa_pairs": [
            {
                "q": "What components are in this architecture?",
                "a": f"This architecture has {'a CDN (CloudFront), ' if cdn else ''}DNS (Route53), a load balancer (ALB), {num_app_servers} application servers (EC2), {'a Redis cache, ' if cache else ''}and a {db} database."
            },
            {
                "q": "What is the single biggest risk in this design?",
                "a": f"The single database instance. If it goes down, the entire application is unavailable. There are no read replicas or multi-AZ failover configured."
            },
            {
                "q": "How would you scale this to handle 10x traffic?",
                "a": f"Three changes: (1) Add auto-scaling for app servers to handle 10x the compute. (2) Add read replicas to the database to distribute read traffic. (3) {'Optimize cache hit rates to reduce DB load.' if cache else 'Add a Redis caching layer to serve hot data from memory.'} If writes are also 10x, consider partitioning the database."
            }
        ]
    }
    # Remove None values
    description["components"] = {k: v for k, v in description["components"].items() if v}
    
    return description


def microservices_pattern(variant_id):
    """Microservices with API Gateway, message queue, multiple services"""
    name = f"microservices_{variant_id}"
    filename = str(OUTPUT_DIR / name)
    
    num_services = random.choice([3, 4, 5])
    service_names = random.sample(
        ["User Service", "Order Service", "Product Service", "Payment Service",
         "Notification Service", "Search Service", "Inventory Service", "Auth Service"],
        num_services
    )
    has_queue = random.choice([True, True, True, False])  # 75% chance
    has_cache = random.choice([True, False])
    
    with Diagram(f"Microservices Architecture", filename=filename, show=False, direction="TB"):
        gw = APIGateway("API Gateway")
        
        with Cluster("Services"):
            services = [ECS(name) for name in service_names]
        
        with Cluster("Data Stores"):
            dbs = [DynamoDB(f"{name.split()[0]} DB") for name in service_names[:3]]
            if has_cache:
                cache = ElastiCache("Shared Cache")
        
        if has_queue:
            queue = SQS("Event Queue")
            events = SNS("Event Bus")
        
        gw >> services[0]
        gw >> services[1]
        if num_services > 2:
            gw >> services[2]
        
        for i, svc in enumerate(services[:3]):
            svc >> dbs[i]
        
        if has_queue:
            services[0] >> events
            events >> queue
            queue >> services[-1]
        
        if has_cache:
            for svc in services[:2]:
                svc >> cache
    
    description = {
        "id": name,
        "image_path": f"{filename}.png",
        "architecture_type": "microservices",
        "components": {
            "api_gateway": "API Gateway for request routing, auth, rate limiting",
            "services": {name: f"Handles {name.lower().replace('service', '').strip()} domain logic" for name in service_names},
            "databases": f"Each core service has its own DynamoDB table (database-per-service pattern)",
            "event_bus": "SNS/SQS for async inter-service communication" if has_queue else None,
            "cache": "Shared ElastiCache for cross-service caching" if has_cache else None,
        },
        "data_flow": f"API Gateway routes requests to appropriate service -> each service owns its data -> {'services communicate asynchronously via event bus' if has_queue else 'services call each other synchronously'}",
        "potential_bottlenecks": [
            "API Gateway is a single entry point (though AWS API GW scales automatically)",
            f"{'Synchronous inter-service calls create coupling and cascading failures' if not has_queue else 'Event queue could become a bottleneck under high load'}",
            f"{'Shared cache contention between services' if has_cache else 'No caching - every request hits the database'}",
        ],
        "qa_pairs": [
            {
                "q": "What communication pattern are the services using?",
                "a": f"{'Asynchronous event-driven communication via SNS (pub/sub) and SQS (queue). This decouples services and improves resilience.' if has_queue else 'The diagram shows synchronous communication. Services call each other directly, which creates tight coupling and risk of cascading failures.'}"
            },
            {
                "q": "Is the database-per-service pattern correct here?",
                "a": f"Yes, each service has its own DynamoDB table, which is the standard microservices pattern. This gives each service data independence and allows them to choose the best storage for their needs. The trade-off is that cross-service queries require API calls instead of joins."
            }
        ]
    }
    description["components"] = {k: v for k, v in description["components"].items() if v}
    
    return description


def event_driven_pipeline(variant_id):
    """Event-driven data pipeline: producers -> Kafka/Kinesis -> consumers -> storage"""
    name = f"pipeline_{variant_id}"
    filename = str(OUTPUT_DIR / name)
    
    num_producers = random.choice([2, 3, 4])
    producer_names = random.sample(
        ["Web App", "Mobile API", "IoT Devices", "Partner API", "Batch Import", "Webhooks"],
        num_producers
    )
    has_stream_processing = random.choice([True, False])
    has_analytics = random.choice([True, False])
    
    with Diagram(f"Event-Driven Pipeline", filename=filename, show=False, direction="LR"):
        with Cluster("Producers"):
            producers = [EC2(name) for name in producer_names]
        
        stream = Kinesis("Event Stream")
        
        with Cluster("Consumers"):
            if has_stream_processing:
                processor = Lambda("Stream Processor")
            db_writer = Lambda("DB Writer")
            if has_analytics:
                analytics = Lambda("Analytics")
        
        with Cluster("Storage"):
            primary = DynamoDB("Primary Store")
            if has_analytics:
                warehouse = Redshift("Data Warehouse")
            archive = S3("Archive")
        
        for prod in producers:
            prod >> stream
        
        stream >> db_writer >> primary
        if has_stream_processing:
            stream >> processor >> primary
        if has_analytics:
            stream >> analytics >> warehouse
        stream >> archive
    
    description = {
        "id": name,
        "image_path": f"{filename}.png",
        "architecture_type": "event_driven_pipeline",
        "components": {
            "producers": f"{num_producers} event producers: {', '.join(producer_names)}",
            "event_stream": "Kinesis as the central event backbone",
            "consumers": f"Lambda functions for processing: {'stream processor, ' if has_stream_processing else ''}DB writer{', analytics' if has_analytics else ''}",
            "storage": f"DynamoDB (primary), {'Redshift (analytics), ' if has_analytics else ''}S3 (archive)",
        },
        "data_flow": f"Events from {num_producers} sources -> Kinesis stream -> multiple consumers process in parallel -> data lands in DynamoDB{', Redshift,' if has_analytics else ''} and S3",
        "potential_bottlenecks": [
            "Kinesis shard limits (1MB/s write, 2MB/s read per shard)",
            "Lambda concurrency limits under spike traffic",
            "DynamoDB write capacity if not properly provisioned",
        ],
        "qa_pairs": [
            {
                "q": "What happens if a consumer fails to process an event?",
                "a": "Kinesis retains records for 24 hours (configurable up to 365 days). Failed Lambda invocations will be retried automatically. For persistent failures, configure a dead letter queue. The key is making consumers idempotent so retries are safe."
            },
            {
                "q": "How do you handle ordering of events?",
                "a": "Kinesis guarantees ordering within a shard. Use a partition key (like user_id) to ensure events for the same entity go to the same shard. Cross-shard ordering is not guaranteed and must be handled at the application level if needed."
            }
        ]
    }
    
    return description


# ─── Generation Loop ───

TEMPLATES = [
    ("web_basic", web_app_basic, 60),
    ("microservices", microservices_pattern, 40),
    ("pipeline", event_driven_pipeline, 40),
]

def main():
    all_descriptions = []
    total = 0
    
    for template_name, generator_fn, count in TEMPLATES:
        print(f"Generating {count} {template_name} diagrams...")
        for i in range(count):
            try:
                desc = generator_fn(f"{i:03d}")
                all_descriptions.append(desc)
                total += 1
            except Exception as e:
                print(f"  Error generating {template_name}_{i}: {e}")
    
    # Write descriptions
    with open(DESCRIPTIONS_FILE, "w") as f:
        for desc in all_descriptions:
            f.write(json.dumps(desc) + "\n")
    
    print(f"\nGenerated {total} diagrams in {OUTPUT_DIR}/")
    print(f"Descriptions saved to {DESCRIPTIONS_FILE}")
    print(f"\nNote: This script generates {total} diagrams from 3 templates.")
    print("For a production dataset, add more templates:")
    print("  - Multi-region architectures")
    print("  - CQRS / Event sourcing patterns")
    print("  - Data lake architectures")
    print("  - ML serving pipelines")
    print("  - Multi-tier caching")
    print("  - Service mesh topologies")
    print("  - Batch processing (Spark/Flink)")
    print("Each template with 40-60 variants gives you 500+ total diagrams.")


if __name__ == "__main__":
    main()

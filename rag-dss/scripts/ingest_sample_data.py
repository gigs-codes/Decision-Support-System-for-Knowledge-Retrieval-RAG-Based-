#!/usr/bin/env python3
"""
scripts/ingest_sample_data.py

Seeds the knowledge base with sample policy and technical documents
so you can immediately try queries after starting the server.

Usage:
    python scripts/ingest_sample_data.py [--base-url http://localhost:8000]
"""

import argparse
import json
import sys
import time

import httpx

SAMPLE_DOCUMENTS = [
    {
        "title": "Remote Work Policy",
        "content": """
Remote Work Policy — Effective January 2024

1. Eligibility
All full-time employees who have completed their 90-day probationary period are eligible
for remote work arrangements. Part-time employees may apply subject to manager approval.

2. Equipment and Security
The company will provide a laptop and necessary peripherals. Employees must use company-approved
VPN at all times when accessing internal systems. Personal devices are not permitted for
handling confidential data without prior written approval from IT Security.

3. Working Hours
Core hours are 10:00 AM – 3:00 PM in the employee's local timezone. Outside core hours,
employees are expected to maintain reasonable availability and respond to urgent communications
within two hours.

4. Performance Management
Remote employees are evaluated on output and results, not hours logged. Managers should set
clear quarterly OKRs and conduct bi-weekly 1:1 check-ins.

5. Expenses
Home internet costs are reimbursed up to $80/month with a valid receipt. Ergonomic equipment
(chair, monitor stand) up to $500 is available as a one-time expense.

6. Compliance
Violation of security policies may result in immediate revocation of remote work privileges.
Repeated violations may lead to disciplinary action up to and including termination.
        """,
        "metadata": {
            "source": "HR Policy Manual v3.2",
            "author": "HR Department",
            "department": "Human Resources",
            "tags": ["policy", "remote-work", "hr"],
        },
    },
    {
        "title": "API Integration Guide",
        "content": """
API Integration Guide — Internal Developer Reference

Authentication
All API requests must include the X-API-Key header. Keys are generated via the Developer Portal
and rotate every 90 days. Service-to-service calls should use JWT tokens signed with RS256.

Rate Limits
Standard tier: 60 requests/minute. Premium tier: 500 requests/minute.
Rate limit headers are included in every response:
  X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset

Error Handling
The API uses standard HTTP status codes:
  200 OK — successful request
  201 Created — resource created
  400 Bad Request — malformed input (details in body)
  401 Unauthorized — missing or invalid API key
  429 Too Many Requests — rate limit exceeded (see Retry-After header)
  500 Internal Server Error — contact support with the X-Request-ID header value

Pagination
List endpoints support ?page=N&page_size=M. Default page_size is 20; maximum is 100.
Responses include: items[], total, page, page_size.

Webhooks
Register webhook URLs in the Developer Portal. We sign each delivery with HMAC-SHA256
using your webhook secret. Always verify the X-Signature header before processing.

Versioning
The API uses URI versioning: /api/v1/. Breaking changes introduce a new version.
Old versions are deprecated for a minimum of 12 months before sunset.
        """,
        "metadata": {
            "source": "Developer Documentation Portal",
            "author": "Platform Engineering",
            "department": "Engineering",
            "tags": ["api", "integration", "developer"],
        },
    },
    {
        "title": "Data Governance Framework",
        "content": """
Data Governance Framework — Version 2.1

Classification Tiers
  Public      — No restrictions. Marketing collateral, press releases.
  Internal    — Accessible to all employees. Internal memos, org charts.
  Confidential— Need-to-know basis. Financial projections, customer PII.
  Restricted  — Board and exec only. M&A targets, unreleased product plans.

Data Retention
Personal data covered by GDPR must be deleted within 30 days of a deletion request.
Financial records are retained for 7 years per regulatory requirements.
Log data is retained for 12 months (security) and 90 days (application).

Access Control
Role-based access control (RBAC) governs all internal systems.
Privileged access (admin roles) requires dual-approval and quarterly reviews.
All access is logged and monitored by the Security Operations Center (SOC).

Data Breach Response
  1. Identify and contain the breach within 1 hour of detection.
  2. Notify the CISO and Legal within 2 hours.
  3. If PII is involved, notify affected individuals within 72 hours (GDPR requirement).
  4. Submit regulatory notifications within required timeframes.
  5. Conduct post-mortem within 14 days.

Third-Party Data Sharing
All data-sharing agreements must be approved by Legal and the DPO.
Sub-processors handling EU personal data must maintain SCCs or equivalent safeguards.
        """,
        "metadata": {
            "source": "Compliance & Legal",
            "author": "Data Protection Officer",
            "department": "Legal",
            "tags": ["compliance", "gdpr", "data-governance"],
        },
    },
]


def ingest(base_url: str, document: dict) -> dict:
    url = f"{base_url}/api/v1/documents/ingest"
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, json=document)
        resp.raise_for_status()
        return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Seed the RAG knowledge base with sample documents")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    print(f"\n🌱 Seeding knowledge base at {args.base_url}\n")

    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        try:
            result = ingest(args.base_url, doc)
            status = result.get("status", "unknown")
            chunks = result.get("chunk_count", 0)
            print(f"  ✅ [{i}/{len(SAMPLE_DOCUMENTS)}] '{doc['title']}' — {chunks} chunks ({status})")
        except httpx.HTTPError as e:
            print(f"  ❌ [{i}/{len(SAMPLE_DOCUMENTS)}] '{doc['title']}' — FAILED: {e}")
            sys.exit(1)
        time.sleep(0.5)  # be polite to the API

    print(f"\n✨ Done! {len(SAMPLE_DOCUMENTS)} documents indexed.\n")
    print("Try a query:")
    print(f'  curl -X POST {args.base_url}/api/v1/query/ \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"question": "What is the remote work expense reimbursement policy?"}\'\n')


if __name__ == "__main__":
    main()

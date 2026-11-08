"""Authoritative synthetic policy corpus. Seed writes these files and the retriever ingests them."""

from __future__ import annotations

from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent / "policies"

DOCUMENTS: list[dict] = [
    {
        "slug": "annual-leave-v3",
        "title": "Annual Leave Policy",
        "doc_type": "annual_leave",
        "version": "3.0",
        "policy_family": "annual_leave",
        "effective_date": "2026-01-01",
        "expiry_date": "",
        "supersedes": "annual-leave-v2",
        "department": "",
        "location": "global",
        "access": "internal",
        "conflict_group": "",
        "status": "current",
        "body": """
## Entitlement

Employees are entitled to 24 days of annual leave each calendar year. The entitlement is recorded in the leave balance for that year. Up to 5 unused days may carry into the next year. Carryover above 5 days expires on 31 March.

## Notice period

The notice period depends on duration. A request of 5 business days or fewer needs 5 business days of notice. A request longer than 5 business days needs 10 business days of notice. The workflow counts business days and excludes the published company holidays.

## Limits

Annual leave may not exceed 15 consecutive business days without a separate written exception from People Operations. Leave is requested in business days. Weekends and published holidays are not deducted from the balance.

## Approval

The employee's approval manager must approve the request before it is confirmed in the HRIS. This policy does not let an automated assistant approve leave.

## Insufficient balance

A request is not accepted when the employee does not have sufficient annual leave balance. The assistant does not submit that request and does not reduce the balance.
""",
    },
    {
        "slug": "annual-leave-v2",
        "title": "Annual Leave Policy (expired)",
        "doc_type": "annual_leave",
        "version": "2.0",
        "policy_family": "annual_leave",
        "effective_date": "2024-01-01",
        "expiry_date": "2025-12-31",
        "supersedes": "",
        "department": "",
        "location": "global",
        "access": "internal",
        "conflict_group": "",
        "status": "expired",
        "body": """
## Old entitlement

Employees are entitled to 18 days of annual leave each calendar year under this expired version. The old notice period was 3 business days for every request.

## Why this text remains

This document is retained for audit history only. It was superseded by Annual Leave Policy 3.0 and must not be used for new requests. Retrieval systems should prefer the current version.
""",
    },
    {
        "slug": "sick-leave-v2",
        "title": "Sick Leave Policy",
        "doc_type": "sick_leave",
        "version": "2.0",
        "policy_family": "sick_leave",
        "effective_date": "2026-01-01",
        "expiry_date": "",
        "supersedes": "",
        "department": "",
        "location": "global",
        "access": "internal",
        "conflict_group": "",
        "status": "current",
        "body": """
## Entitlement

Employees have 10 days of paid sick leave each calendar year. Sick leave is for illness or injury and is not part of the annual leave balance.

## Medical note

A medical note is required when sick leave lasts more than 3 consecutive business days. The note goes to People Operations, not to the team chat channel.

## Notification

Tell the line manager before 09:30 local time on the first day. The assistant may explain this rule. It may not certify an illness.
""",
    },
    {
        "slug": "parental-leave-v1",
        "title": "Parental Leave Policy",
        "doc_type": "parental_leave",
        "version": "1.0",
        "policy_family": "parental_leave",
        "effective_date": "2026-01-01",
        "expiry_date": "",
        "supersedes": "",
        "department": "",
        "location": "global",
        "access": "internal",
        "conflict_group": "",
        "status": "current",
        "body": """
## Primary caregiver

Primary caregiver parental leave is 16 weeks. It can start up to two weeks before the expected date when the employee gives notice.

## Secondary caregiver

Secondary caregiver parental leave is 4 weeks. Those weeks can be taken in one block or two blocks during the first year.

## Notice

Where the leave is foreseeable, tell People Operations at least 30 days before the start. Parental leave is not deducted from annual leave.
""",
    },
    {
        "slug": "attendance-v2",
        "title": "Attendance Policy",
        "doc_type": "attendance",
        "version": "2.0",
        "policy_family": "attendance",
        "effective_date": "2026-01-01",
        "expiry_date": "",
        "supersedes": "",
        "department": "",
        "location": "global",
        "access": "internal",
        "conflict_group": "",
        "status": "current",
        "body": """
## Core hours

Core hours are 10:00 to 16:00 in the employee's work location. Teams may agree flexible time outside the core hours.

## Unplanned absence

An unplanned absence is a weekday recorded as absent or sick. The employee notifies the line manager before 09:30 local time. Leave that was approved in advance is not an unplanned absence.

## Records

The attendance extract is the source for absence rates. A statistical outlier is a prompt for an HR review, not a disciplinary finding by itself.
""",
    },
    {
        "slug": "overtime-v3",
        "title": "Overtime Policy",
        "doc_type": "overtime",
        "version": "3.0",
        "policy_family": "overtime",
        "effective_date": "2026-01-01",
        "expiry_date": "",
        "supersedes": "",
        "department": "",
        "location": "global",
        "access": "internal",
        "conflict_group": "overtime-premium",
        "status": "current",
        "body": """
## Overtime premium

The overtime premium is 1.5 times the base hourly rate. It is paid for hours worked beyond 40 in a work week.

## Authorization

Overtime is worked only with prior manager approval. The assistant cannot authorize overtime or change the premium.

## Record

Hours are taken from the time extract. This global policy remains current.
""",
    },
    {
        "slug": "overtime-memo-conflict",
        "title": "Finance Overtime Memorandum",
        "doc_type": "overtime",
        "version": "1.0",
        "policy_family": "overtime_memo",
        "effective_date": "2026-06-01",
        "expiry_date": "",
        "supersedes": "",
        "department": "",
        "location": "global",
        "access": "internal",
        "conflict_group": "overtime-premium",
        "status": "current",
        "body": """
## Conflicting overtime premium

Effective 1 June 2026, this memorandum says the overtime premium is 1.25 times the base hourly rate. It is payable only after 45 hours in a work week.

## Status

This memorandum was issued by Finance and is still marked current. It was not recorded as superseding the global Overtime Policy. Readers who see both documents have a conflict and should not invent a blended rate.
""",
    },
    {
        "slug": "remote-work-v2",
        "title": "Remote Work Policy",
        "doc_type": "remote_work",
        "version": "2.0",
        "policy_family": "remote_work",
        "effective_date": "2026-01-01",
        "expiry_date": "",
        "supersedes": "",
        "department": "",
        "location": "global",
        "access": "internal",
        "conflict_group": "",
        "status": "current",
        "body": """
## Remote days

Employees may work up to 3 remote days each week with agreement from the line manager. Remote days are a work arrangement, not a substitute for annual leave or sick leave.

## Location

The employee remains responsible for their core hours. A different country for more than 10 business days needs People Operations approval.
""",
    },
    {
        "slug": "employee-transfers-v1",
        "title": "Employee Transfer Policy",
        "doc_type": "transfer",
        "version": "1.0",
        "policy_family": "transfer",
        "effective_date": "2026-01-01",
        "expiry_date": "",
        "supersedes": "",
        "department": "",
        "location": "global",
        "access": "internal",
        "conflict_group": "",
        "status": "current",
        "body": """
## What changes on a transfer

An employee transfer changes the department and the line manager on the effective date. People Operations must also update the approval manager so leave approval routing follows the new line manager.

## If the fields diverge

If the approval manager still names the previous manager, leave requests continue to route to that previous manager. That is a data defect, not a second reporting line. The employee cannot correct the official record from chat.

## How it is fixed

People Operations reviews a correction request and, if approved, updates the approval manager and any open manager-stage requests. Until that approval is recorded, the HRIS keeps the old route.
""",
    },
    {
        "slug": "reporting-structures-v1",
        "title": "Reporting Structure Policy",
        "doc_type": "reporting",
        "version": "1.0",
        "policy_family": "reporting",
        "effective_date": "2026-01-01",
        "expiry_date": "",
        "supersedes": "",
        "department": "",
        "location": "global",
        "access": "internal",
        "conflict_group": "",
        "status": "current",
        "body": """
## One line manager

Each employee has one current line manager. The line manager is the operational manager stored on the employee profile.

## Approval manager

The approval manager receives leave requests. The approval manager must be the same person as the line manager unless People Operations has recorded a time-bounded exception. An expired department move is not an exception.
""",
    },
    {
        "slug": "leave-approval-v2",
        "title": "Leave Approval Policy",
        "doc_type": "leave_approval",
        "version": "2.0",
        "policy_family": "leave_approval",
        "effective_date": "2026-01-01",
        "expiry_date": "",
        "supersedes": "",
        "department": "",
        "location": "global",
        "access": "internal",
        "conflict_group": "extended-leave-approval",
        "status": "current",
        "body": """
## Manager decision

The approval manager decides first. The assistant may prepare the request and explain the impact. The assistant must not set the request to approved.

## People Operations co-approval

People Operations co-approval is required when consecutive business days exceed 10, or when the absence overlaps a critical project deadline. The stricter global rule remains the workflow default while any conflicting addendum is also current.

## Rejection

A rejected request releases reserved days. It does not delete the audit record.
""",
    },
    {
        "slug": "engineering-leave-addendum-v1",
        "title": "Engineering Leave Addendum",
        "doc_type": "leave_approval",
        "version": "1.0",
        "policy_family": "leave_approval_addendum",
        "effective_date": "2026-04-01",
        "expiry_date": "",
        "supersedes": "",
        "department": "Engineering",
        "location": "global",
        "access": "internal",
        "conflict_group": "extended-leave-approval",
        "status": "current",
        "body": """
## Engineering exception

This addendum says Engineering managers may approve up to 15 consecutive business days of annual leave without People Operations co-approval. It also says that approval still stands when the absence overlaps a project deadline.

## Conflict

The addendum does not say that it supersedes the global Leave Approval Policy. Both documents are current. Workflows must surface the conflict instead of silently using the looser rule.
""",
    },
    {
        "slug": "compensation-bands-v1",
        "title": "Compensation Bands",
        "doc_type": "compensation",
        "version": "1.0",
        "policy_family": "compensation",
        "effective_date": "2026-01-01",
        "expiry_date": "",
        "supersedes": "",
        "department": "",
        "location": "global",
        "access": "hr_confidential",
        "conflict_group": "",
        "status": "current",
        "body": """
## Restricted bands

This compensation document is confidential to People Operations. Reference code BAND-HELIX-9 applies to the internal grade table. Do not quote this code to employees or managers.

## Use

Salary figures for a named employee come from the employee record, not from this document, and only for an authorized HR administrator. The assistant must not export the full payroll.
""",
    },
]


def materialize_policies() -> None:
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    for document in DOCUMENTS:
        path = POLICY_DIR / f"{document['slug']}.md"
        front = [
            "---",
            f"id: {document['slug']}",
            f"title: {document['title']}",
            f"doc_type: {document['doc_type']}",
            f"version: {document['version']}",
            f"policy_family: {document['policy_family']}",
            f"effective_date: {document['effective_date']}",
            f"expiry_date: {document['expiry_date'] or 'none'}",
            f"supersedes: {document['supersedes'] or 'none'}",
            f"department: {document['department'] or 'none'}",
            f"location: {document['location']}",
            f"access: {document['access']}",
            f"conflict_group: {document['conflict_group'] or 'none'}",
            f"status: {document['status']}",
            "---",
            "",
        ]
        path.write_text("\n".join(front) + document["body"].strip() + "\n", encoding="utf-8")


def parse_policy_file(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"Missing front matter: {path.name}")
    _, matter, body = raw.split("---", 2)
    meta: dict[str, str] = {}
    for line in matter.strip().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body.strip()

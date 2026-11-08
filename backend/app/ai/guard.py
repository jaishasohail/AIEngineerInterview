import re

INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"ignore\s+(?:\w+\s+){0,4}(all|any|previous|above|prior)\s+(instructions|rules|prompts)",
        r"ignore\s+(?:\w+\s+){0,6}(access restrictions?|permissions?|authorization|rbac|rules)",
        r"ignore\s+(?:\w+\s+){0,6}(role\s+restrictions?|my\s+role|my\s+permissions?|normal\s+restrictions?|restrictions?)",
        r"bypass\s+(?:\w+\s+){0,4}(my\s+)?(permissions?|authorization|auth|rbac|security|approval|restrictions?)",
        r"disregard\s+(?:\w+\s+){0,4}(authorization|permissions?|restrictions?|access\s+rules?|the\s+)?(system|policy|rules)?",
        r"override\s+(?:\w+\s+){0,6}(access|permission|authorization|role|rbac)",
        r"forget\s+(?:\w+\s+){0,6}(permission|authorization|access|rbac|security)\s+model",
        r"you are now",
        r"disregard (the |your )?(system|policy|rules)",
        r"(?:show|reveal|print|give|display|repeat).{0,80}(?:system|developer|hidden)\s+prompt",
        r"\bdatabase\s+credentials\b",
        r"\b(postgres|database)\s+password\b",
        r"bypass (auth|authorization|rbac|security|approval)",
        r"ignore\s+(?:the\s+|all\s+)?rbac",
        r"pretend (?:you are|to be|i am|i'm|my role is)\s+(?:an?\s+)?(hr|hr_admin|admin|administrator|manager|ceo)",
        r"do not (enforce|follow) (the )?(policy|rules|permissions)",
        r"\bi have permission\b.{0,80}\bignore\b",
        r"\beven though\b.{0,40}\bnot authorized\b",
        r"\bsystem:\s",
    ]
]

SENSITIVE_PATTERNS = {
    "salary": re.compile(r"salary|compensation|pay band|wage|bonus|payroll", re.IGNORECASE),
    "performance": re.compile(r"performance|rating|review score|appraisal", re.IGNORECASE),
    "personal": re.compile(r"\bphone\b|home address|personal email|date of birth", re.IGNORECASE),
}

_DESTRUCTIVE = re.compile(
    r"\b(delete|remove|erase|wipe|destroy|purge|drop)\b",
    re.IGNORECASE,
)
_DESTRUCTIVE_TARGET = re.compile(
    r"\b(attendance|leave(?:_|\s+)?requests?|payroll|salar(?:y|ies)|employee(?:\s+records?)?|"
    r"balances?|profile|hris|database|table|schema)\b",
    re.IGNORECASE,
)
_OUTSIDE_SCOPE = re.compile(
    r"\b("
    r"outside\s+(?:my|our)\s+(?:reporting\s+|authorized\s+|authorised\s+)?scope"
    r"|department\s+outside"
    r"|outside\s+(?:my|our)\s+(?:department|team|hierarchy|reporting)"
    r"|not\s+(?:in|on)\s+my\s+(?:team|department|hierarchy)"
    r"|not\s+my\s+report"
    r"|non-?reports?"
    r"|include\s+non-?reports?"
    r"|people\s+outside\s+(?:my|our)\s+reports?"
    r"|including\s+people\s+outside\s+(?:my|our)\s+reports?"
    r"|another\s+department(?:'s)?"
    r"|(?:a|another)\s+department\s+i\s+(?:do\s+not|don't)\s+manage"
    r"|employees?\s+outside\s+(?:my|our)\s+(?:team|hierarchy|reporting|reports?)"
    r"|even though\s+(?:she|he|they)\s+is\s+not\s+my\s+report"
    r")\b",
    re.IGNORECASE,
)


def detect_injection(message: str) -> list[str]:
    hits = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(message):
            hits.append(pattern.pattern)
    return hits


def sensitive_kinds(message: str) -> set[str]:
    return {kind for kind, pattern in SENSITIVE_PATTERNS.items() if pattern.search(message)}


def wants_bulk_export(message: str) -> bool:
    return bool(
        re.search(
            r"all salaries|everyone'?s salary|salary details of everyone|salaries of everyone|salary list|payroll export|entire payroll|all employee salary records",
            message,
            re.IGNORECASE,
        )
    )


def export_dump_request(message: str) -> bool:
    """Export/dump of HR tables or payroll, denied for every role."""
    text = message.lower()
    return bool(
        re.search(r"\b(export|dump)\b", text)
        and re.search(r"\b(database|employees|attendance|salar|payroll)\b", text)
    )


def hard_security_violation(message: str) -> bool:
    """True for injection, credential/prompt probing, and role-impersonation attacks.

    Legacy regexes are kept for the exact phrasings covered by earlier regression suites; the
    compositional layer in app.ai.security generalizes the same categories to unseen wording.
    """
    if detect_injection(message):
        return True
    from app.ai.security import role_agnostic_violation

    if role_agnostic_violation(message):
        return True
    text = message.lower()
    if re.search(
        r"\b(system prompt|developer prompt|system instructions|database(?:\s+\w+){0,3}\s+credentials|"
        r"database password|employee passwords|private conversation|superadmin|another organization|"
        r"api key|access token|jwt(?:\s+signing)?\s+secret|signing secret|api secret)\b",
        text,
    ):
        return True
    if re.search(r"\b(export|dump|retrieve|fetch|open|hand me)\b", text) and re.search(
        r"\b(database|employees|attendance|salar|hr\s+database|employee\s+(?:database|table|files)|"
        r"everyone'?s?\s+hr|full employee table|hr files)\b",
        text,
    ) and re.search(
        r"\b(ignore|bypass|disregard|override|pretend|forget|not authorized|role restrictions?|permissions?|access level)\b",
        text,
    ):
        return True
    if re.search(r"\b(export|dump)\b", text) and re.search(r"\b(database|employees|attendance|salar|payroll)\b", text):
        return True
    if re.search(r"\b(administrator|admin)\s+permission\b", text) and re.search(
        r"\b(show|reveal|export|dump|list|access|display|give|retrieve|fetch)\b", text
    ) and re.search(r"\b(all|every|entire)\b", text):
        return True
    if re.search(r"\bpretend\b", text) and re.search(r"\b(hr|hr_admin|admin|administrator)\b", text) and re.search(
        r"\b(show|reveal|export|dump|list|access|display|give|retrieve|fetch)\b", text
    ):
        return True
    return False


def broad_private_data_request(message: str) -> bool:
    """Organization-wide private-data asks. Role gating belongs in classify/authorize."""
    text = message.lower()
    if re.search(r"\ball employee records\b", text):
        return True
    if re.search(r"\bentire employee (?:database|directory|records?)\b", text):
        return True
    if re.search(r"\bcomplete (?:hr|employee) (?:database|directory|records?)\b", text):
        return True
    if re.search(
        r"\b(personal information|personal records?|phone numbers?|home addresses?|"
        r"employees'? personal|personal data of (?:every|all)|every employee'?s personal)\b",
        text,
    ) and re.search(r"\b(every|all|everyone|everybody|entire|organization-wide)\b", text):
        return True
    private = bool(
        re.search(
            r"\b(attendance|salar(?:y|ies)|records|database|payroll|credentials|employee data|personal information)\b",
            text,
        )
    )
    broad = bool(
        re.search(
            r"\b(everyone(?:'s)?|everybody(?:'s)?|all employees|every employee|entire|organization-wide|organisation-wide|company-wide)\b",
            text,
        )
    )
    action = bool(re.search(r"\b(show|reveal|export|dump|list|access|display|give|retrieve|fetch)\b", text))
    return bool(broad and private and action)


def unauthorized_data_request(message: str) -> bool:
    """True when the message asks to bypass access rules or read other people's records.

    A policy question about employees in general is not an unauthorized data request.
    Callers that support HR Admin analytics must combine this with role checks: use
    hard_security_violation for always-denied probes and broad_private_data_request
    only for non-HR roles.
    """
    return hard_security_violation(message) or broad_private_data_request(message)


def unsupported_destructive_request(message: str) -> bool:
    """True when the user asks to delete or erase HRIS records through the assistant."""
    text = message.lower()
    if not _DESTRUCTIVE.search(text):
        return False
    if not _DESTRUCTIVE_TARGET.search(text):
        return False
    # Policy questions about what happens after deletion stay informational.
    if re.search(r"\b(what happens|policy|policies|process for|how (?:do|can) i request)\b", text):
        return False
    return True


def outside_authorized_scope_request(message: str) -> bool:
    """True when the user explicitly asks for data outside their reporting or department scope."""
    return bool(_OUTSIDE_SCOPE.search(message))


def other_employee_private_request(message: str) -> bool:
    """True when the user asks for another person's private leave/medical/personal records.

    Policy questions about medical notes or sick-leave rules are not private-record asks.
    Named-person medical/sick/leave-history asks are private-record asks.
    """
    text = message.lower()
    other = bool(
        re.search(
            r"\b(another employee(?:'s)?|other employee(?:'s)?|someone else(?:'s)?|"
            r"a colleague(?:'s)?|my (?:colleague|coworker|co-worker)(?:'s)?)\b",
            text,
        )
    )
    # Own-manager salary asks stay on the confidential_lookup path.
    if re.search(r"\bmy manager'?s?\b", text) and re.search(r"\b(salary|compensation)\b", text):
        return False
    # Possessive / named person markers (e.g. "Liam Chen's sick leave medical notes").
    # Named salary/compensation uses confidential_lookup + assert_sensitive_access instead.
    # Generic possessives that are not people ("the company's", "the policy's") are excluded.
    possessive = re.search(r"\b([a-z]+(?:\s+[a-z]+){0,2})'s\b", text)
    self_related = bool(re.search(r"\bmy\s+(?:manager|approver|supervisor|team|department|own)'s\b", text))
    named_private = bool(
        possessive
        and not self_related
        and not re.search(
            r"\b(company|organi[sz]ation|policy|team|department|year|month|week|employer|firm|helios|assistant|system|"
            r"it|that|this|what|who|there|here|let|he|she|how|where|when|one|today|tomorrow|yesterday|quarter)$",
            possessive.group(1),
        )
        and re.search(
            r"\b(medical|diagnos\w*|sick\s+(?:leave|file|note|record)|leave\s+history|leave\s+balance|balances?|"
            r"phone|personal|certificate|health|attendance\s+(?:history|records?))\b",
            text,
        )
    )
    # A pure policy/entitlement question with no other-person marker is informational.
    # Wrapping a named person's private ask in "according to policy" does not make it one.
    policy_only = bool(
        re.search(r"\b(policy|policies|entitlement|entitled)\b", text)
        and not other
        and not named_private
        and not re.search(r"\b(notes?|history|records?|diagnos\w*)\b", text)
    )
    if policy_only:
        return False
    private = bool(
        re.search(
            r"\b(medical(?:\s+notes?)?|diagnos\w*|sick (?:leave|file|note|record)|leave (?:history|details|records?|balance)|"
            r"balances?|personal (?:information|details|records?)|phone|certificate|health)\b",
            text,
        )
    )
    return bool((other or named_private) and private)


def out_of_hierarchy_request(message: str) -> bool:
    """True when the user asks for people outside their reporting line."""
    return outside_authorized_scope_request(message) or bool(
        re.search(
            r"\b(not my report|non-?reports?|outside my reports?|people outside my|"
            r"including people outside|department attendance.{0,40}outside|"
            r"even though.{0,40}not my report|(?:does not|doesn't|do not|don't) manage|"
            r"not managed by me|not in my reporting line|other managers'? (?:teams?|reports?|lines?)|"
            r"another manager'?s (?:team|reports?|line)|"
            r"(?:does not|doesn't|do not|don't|did not|didn't|no longer|never) report(?:s|ed)? to me|"
            r"not (?:one of )?my (?:direct )?reports?\b|isn'?t (?:one of )?my (?:direct )?reports?|is not (?:on|in) my team|"
            r"not (?:on|in) my team|outside (?:of )?my team|someone else'?s? (?:team|report|direct report)|"
            r"reports? to (?:someone|somebody|another manager|a different manager) (?:else)?|"
            r"(?:even|although|though) (?:he|she|they) (?:is|are|was|were) not (?:my|on my|in my))\b",
            message,
            re.IGNORECASE,
        )
    )


def organization_wide_request(message: str) -> bool:
    """True when the current message explicitly asks for organization-wide scope."""
    text = message.lower()
    return bool(
        re.search(
            r"\b(organization-wide|organisation-wide|company-wide|across (?:the )?(?:organization|organisation|company|northstar)|"
            r"entire (?:organization|organisation|company|directory|database|northstar|employee(?:\s+database|\s+directory|\s+records?)?)|"
            r"complete (?:hr|employee) (?:database|directory|records?)|"
            r"all employees|every employee|everyone in the (?:organization|organisation|company)|"
            r"(?:every|all) departments?(?:\s+in\s+(?:northstar|the\s+(?:company|organi[sz]ation)))?|"
            r"cover (?:every|all) departments?|"
            r"across (?:all )?departments?)\b",
            text,
        )
    )

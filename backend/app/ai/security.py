"""Deterministic, compositional security decision layer.

This module never grants access. It decides whether a message must be denied
*before* any data-bearing operation runs, and it explains why in structured
categories. Decisions are composed from independent signal families (actors,
privileges, controls, verbs, targets, scopes, secrets, memory) rather than from a
flat list of known attack sentences, so unseen wording that recombines known
signals is still caught. The LLM classifier is never consulted here.

Role only matters for the few categories where HR Admin legitimately has a wider
organizational scope (other employees' leave records, reporting hierarchy, report
scope). Nothing is waived for an instruction to bypass or weaken the application's
own controls, reveal secrets, read other users' memory, or run destructive
operations, for any role.

Usage:

    decision = assess_security(message, role=user.role, mentioned_other=bool(mentioned))
    if decision.denied:
        intent = decision.intent          # unauthorized_access | unsupported_mutation
        reason = decision.reason          # user-facing denial text
        categories = decision.categories  # audit detail
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ai import guard

# ---------------------------------------------------------------------------
# Signal families. Each is a vocabulary, not an attack sentence.
# ---------------------------------------------------------------------------

_ESCALATION_VERBS = (
    r"(?:act(?:ing)?\s+as|elevate(?:\s+me)?|promote\s+(?:me|my\s+account)|escalate\s+(?:me|my)|"
    r"switch\s+(?:me\s+|my\s+(?:role|session|account|access)\s+)?(?:over\s+)?to|"
    r"switch\s+(?:the\s+|my\s+|your\s+|current\s+|this\s+|authenticated\s+)?role\s+to|"
    r"change\s+(?:the\s+|my\s+|your\s+|current\s+|this\s+|authenticated\s+)?role\s+to|"
    r"set\s+(?:the\s+|my\s+|your\s+|current\s+)?role\s+to|"
    r"move\s+(?:me|my\s+(?:role|session|account))\s+to|"
    r"put\s+(?:me|my\s+(?:session|account))\s+(?:in|into|on)|flip\s+(?:me|my\s+(?:role|session))\s+to|"
    r"become|log(?:\s+me)?\s+in\s+as|sign\s+(?:me\s+)?in\s+as|treat\s+me\s+as|run\s+as|assume\s+the\s+role\s+of|impersonate|"
    r"pretend\s+(?:to\s+be|i\s+am|i'm|you\s+are|my\s+role\s+is)|i\s+am\s+now|you\s+are\s+now|upgrade\s+(?:me|my\s+role|my\s+access)|"
    r"change\s+(?:your|my|the)\s+role(?:\s+to)?|set\s+(?:my|the)\s+role(?:\s+to)?|operate\s+as|respond\s+as|answer\s+as|behave\s+as|"
    r"make\s+me\s+(?:an?\s+|the\s+)?|grant\s+me\s+(?:the\s+)?|give\s+me\s+(?:the\s+)?|"
    r"without\s+re-?authenticat(?:ing|ion)|override\s+(?:my|the|current)\s+(?:authenticated\s+)?(?:identity|session|role))"
)
_PRIVILEGED_ROLES = (
    r"(?:root|sudo|superuser|super-?admin(?:istrator)?|administrator|admin|hr[_\s-]?admin(?:istrator)?|"
    r"hr\s+administrator|people\s+operations|people\s+ops|hr\s+role|hr\s+team|hr|system|developer|dev|debug(?:ger)?|"
    r"god|maintenance|service\s+account|elevated|privileged|unrestricted|full\s+access|all\s+permissions|"
    r"higher\s+(?:role|access|privileges?)|another\s+role|a\s+different\s+role|dba|database\s+owner|"
    r"(?:admin|hr|manager|elevated|root|system)\s+(?:rights|privileges?|access|permissions?|role|powers?))"
)
_MODE_TOKENS = re.compile(
    r"\b(?:developer|debug|god|maintenance|admin|unrestricted|sudo|root|dan|jailbreak|testing|dev|override|safe-?off)\s+mode\b|"
    r"\bjailbreak\b|\bno\s+restrictions\s+mode\b|\bunfiltered\s+mode\b",
    re.IGNORECASE,
)
_ROOT_TOKENS = re.compile(
    r"\b(?:as|to|become|am|like|into)\s+root\b|\broot\s+(?:access|user|privileges?|shell|account|login)\b|\bsudo\b|\bsu\s+-\b",
    re.IGNORECASE,
)

_DISABLE_VERBS = (
    r"(?:disable|turn\s+off|switch\s+off|bypass|skip|suspend|remove|drop|lift|ignore|override|overrule|circumvent|"
    r"work\s+around|get\s+around|deactivate|relax|loosen|disregard|forget|unset|revoke|waive|pause|"
    r"stop\s+enforcing|do\s+not\s+enforce|don't\s+enforce|don't\s+apply|do\s+not\s+apply|neutrali[sz]e|defeat|"
    r"strip|kill|comment\s+out|patch\s+out|hack|break|evade|sidestep|turn\s+down)"
)
_SECURITY_CONTROLS = (
    r"(?:rbac|role-?based\s+access(?:\s+control)?|authori[sz]ation(?:\s+(?:middleware|layer|checks?|service|rules?|logic|step))?|"
    r"auth(?:\s+(?:middleware|layer|checks?|logic))?|access\s+controls?|access\s+checks?|access\s+rules?|access\s+restrictions?|"
    r"permissions?\s+(?:checks?|model|system|rules?|layer|logic)|middleware|"
    r"security(?:\s+(?:controls?|checks?|filters?|layer|rules?|model|policy|logic))?|"
    r"role\s+checks?|role\s+restrictions?|role\s+enforcement|hierarchy\s+filters?|reporting[- ]line\s+filters?|"
    r"guardrails?|safety\s+(?:rules?|checks?|filters?|layer)|"
    r"(?:access|role|permission|security|scope|hierarchy|normal|usual|your|all|any|these|those|my|the)\s+restrictions?|"
    r"(?:the|this|that|your|my|previous|last|earlier|prior)\s+(?:denial|refusal|block|rejection\s+above)|"
    r"policy\s+enforcement|audit\s+(?:log(?:ging)?|trail)|row[- ]level\s+security|your\s+rules|the\s+permission\s+model|"
    r"privacy\s+(?:controls?|checks?|rules?|filters?)|data\s+(?:protection|isolation)|tenant\s+isolation|"
    r"the\s+(?:sandbox|guard|gate)|the\s+(?:access|authori[sz]ation)\s+(?:check|decision)|"
    r"(?:your|the)\s+(?:safeguards?|protections?|limits?\s+on\s+access))"
)
_QUERY_FILTERS = (
    r"(?:where\s+clause|where\s+\w+\s+(?:condition|clause|filter|predicate)|sql\s+filters?|query\s+filters?|row\s+filters?|the\s+filters?|"
    r"manager_id\s+(?:condition|filter|clause|predicate|check)|"
    r"(?:manager|team|scope|department|hierarchy|reporting|report|role|access)\s+(?:filters?|restrictions?|limits?|constraints?|scoping|conditions?|predicates?|clauses?))"
)
_WIDEN_VERBS = (
    r"(?:widen|broaden|expand|extend|open\s+up|enlarge|loosen|relax|remove|drop|lift|ignore|skip|without|bypass|"
    r"take\s+off|strip|disable|override|generali[sz]e|rewrite|modify|change|edit|adjust|alter)"
)
_SCOPE_WIDEN_PHRASES = re.compile(
    r"\b(?:not\s+just\s+my\s+(?:team|reports?|direct\s+reports?|department)|beyond\s+my\s+(?:team|reports?|scope|department|line)|"
    r"includ(?:e|ing)\s+(?:non-?reports?|other\s+teams?|other\s+departments?|everyone\s+else|people\s+outside|"
    r"the\s+rest\s+of\s+the\s+(?:company|org\w*)|"
    r"(?:every|all)\s+(?:\w+\s+){0,4}(?:employees?|staff|workers?|people)\s+in\s+(?:the\s+)?(?:company|organi[sz]ation|firm|business|org)|"
    r"(?:the\s+)?(?:whole|entire)\s+(?:company|organi[sz]ation|workforce|directory)\s+in\s+(?:that|this|the)\s+same|"
    r"everyone\s+in\s+(?:the\s+)?(?:company|organi[sz]ation)|"
    r"all\s+employees?\s+in\s+(?:the\s+)?(?:company|organi[sz]ation|firm))|"
    r"(?:everyone|everybody|all\s+employees?)\s+in\s+(?:the\s+)?(?:company|organi[sz]ation|firm).{0,40}"
    r"(?:team\s+summary|same\s+summary|this\s+summary|that\s+summary|my\s+team)|"
    r"(?:widen|broaden|expand|extend|open\s+up)\s+(?:it|this|that|them|"
    r"(?:my\s+)?(?:team\s+)?(?:attendance\s+)?(?:dashboard|query|report|analytics|view|results?|answer|scope|search|where\s+clause|summary))?\s*"
    r"(?:to|across|over|for|to\s+cover)\s+(?:\w+\s+){0,3}(?:all|every|the\s+whole|the\s+entire|other|every\s+other)\s+"
    r"(?:departments?|teams?|employees?|company|organi[sz]ation|org|staff|managers?|northstar)|"
    r"(?:cover|include)\s+(?:all|every)\s+departments?(?:\s+in\s+(?:northstar|the\s+company|the\s+organi[sz]ation))?|"
    # Bare "across the organization" alone is an org_* request denied by authorization, not
    # team-scope widening. Require department/team expansion cues here.
    r"(?:across|throughout)\s+(?:all\s+)?departments?(?:\s+in\s+(?:northstar|the\s+(?:company|organi[sz]ation|firm)))?|"
    r"(?:all|every)\s+departments?\s+(?:instead|too|as\s+well|not\s+just|rather\s+than|in\s+(?:northstar|the\s+company))|"
    r"(?:same\s+summary|same\s+report|that\s+summary|this\s+summary|team\s+(?:dashboard|report|summary)).{0,40}"
    r"(?:every|all)\s+(?:employee|staff|worker|department)|"
    r"(?:every|all)\s+(?:employee|staff|worker).{0,40}(?:same\s+summary|same\s+report|company|organi[sz]ation)|"
    r"(?:my\s+)?(?:team\s+)?(?:attendance\s+)?dashboard.{0,40}(?:every|all)\s+departments?|"
    r"(?:every|all)\s+departments?.{0,40}(?:my\s+)?(?:team\s+)?(?:attendance\s+)?dashboard)\b",
    re.IGNORECASE,
)

# HR-only workforce compensation analytics. Non-HR roles must hard-deny (not soft out_of_scope).
_HR_ONLY_COMPENSATION = re.compile(
    r"\b(?:hr[-\s]?only|workforce)\s+compensation\s+(?:percentile|percentiles|report|analytics|dashboard)|"
    r"\bcompensation\s+(?:percentile|percentiles)\s+(?:report|analytics|dashboard)|"
    r"\bworkforce\s+compensation\s+(?:percentile|percentiles|report)|"
    r"\b(?:run|generate|open|show|give)\s+(?:me\s+)?(?:the\s+)?(?:hr[-\s]?only\s+)?(?:workforce\s+)?compensation\s+"
    r"(?:percentile|percentiles|band|bands)\b",
    re.IGNORECASE,
)

_AUTHORITY_CLAIMS = re.compile(
    r"\b(?:(?:my|the|our)\s+(?:director|manager|ceo|cto|cfo|coo|cio|ciso|vp|head\s+of\s+\w+|boss|hr|legal|compliance|it|security\s+team|board|"
    r"skip-?level|supervisor|executive|exec|leadership|senior\s+\w+)"
    r"(?:'s)?\s+(?:has\s+|have\s+|already\s+)?(?:verbally\s+|already\s+|just\s+|personally\s+)?"
    r"(?:authori[sz]ed|approved|cleared|allowed|permitted|signed[\s-]?off|told\s+me|said\s+(?:it'?s|its|it\s+is)\s+(?:ok|okay|fine)|ok'?d|okayed|"
    r"gave\s+(?:me\s+)?(?:the\s+)?(?:go-?ahead|green\s+light|approval|permission|sign-?off|clearance)|is\s+(?:ok|okay|fine)\s+with\s+(?:it|this))|"
    r"i\s+(?:have|got|hold|was\s+given|have\s+been\s+given)\s+(?:verbal\s+|written\s+|explicit\s+)?(?:approval|authori[sz]ation|clearance|permission|sign-?off)|"
    r"i\s+(?:am|'m)\s+(?:authori[sz]ed|cleared|allowed|permitted)\s+to|"
    r"for\s+(?:compliance|debugging|audit(?:ing)?|testing|qa|security|legal|troubleshooting|verification|validation)"
    r"(?:\s+(?:purposes|debugging|reasons|review|checks?))?|"
    r"trust\s+me|this\s+is\s+(?:an?\s+)?(?:emergency|urgent|authori[sz]ed|sanctioned)|as\s+(?:an?\s+)?(?:auditor|admin|developer|engineer|dba|consultant))\b",
    re.IGNORECASE,
)

_SECRET_TOKENS = re.compile(
    r"(?:\bdatabase[_\s-]?url\b|\bdb[_\s-]?url\b|\bjwt[_\s-]?secret\b|\bjwt\s+(?:signing\s+)?(?:key|token)\b|\bjwt\b|\bsigning\s+(?:key|secret)\b|"
    r"\bsecret[_\s-]?keys?\b|\bapi[_\s-]?keys?\b|\baccess[_\s-]?tokens?\b|\bbearer\s+tokens?\b|\brefresh\s+tokens?\b|\bsession\s+tokens?\b|\bauth(?:entication)?\s+tokens?\b|"
    r"\bid\s+tokens?\b|\b(?:postgres(?:ql)?|database|db|admin|root|login|system|service|account|user|employee|everyone'?s?)\s+passwords?\b|\bpasswd\b|"
    r"\b(?:database|db|admin|login|system|postgres(?:ql)?|service|account|aws|cloud|smtp|api)\s+credentials?\b|\bconnection\s+strings?\b|"
    # Bare credentials / passwords when the ask is about app/server/config availability.
    r"\bcredentials?\b|\bpasswords?\b|"
    r"\.env\b|\benv(?:ironment)?\s+(?:variables?|file|vars?|config(?:uration)?|secrets?)\b|\bopenrouter\b|\bopenai\s+(?:key|token)\b|\banthropic\s+key\b|"
    r"\bprivate\s+keys?\b|\bssh\s+keys?\b|\bservice[- ]account\s+keys?\b|\bclient\s+secrets?\b|\bpassword\s+hash(?:es)?\b|\bhashed\s+passwords?\b|"
    r"/etc/\w+|/proc/\w+|~/\.\w+|\b(?:production|backend|env\w*|config\w*|server|app(?:lication)?|system|api|client|signing|database|db|deployment)\s+secrets?\b|"
    r"\bsecrets?\s+(?:keys?|values?|tokens?|manager|store)\b|\bencryption\s+keys?\b|\bmaster\s+keys?\b)",
    re.IGNORECASE,
)
_SECRET_ALWAYS = re.compile(
    r"database[_\s-]?url|db[_\s-]?url|jwt|api[_\s-]?key|access[_\s-]?token|bearer|/etc/|/proc/|\.env|openrouter|openai|anthropic|"
    r"connection\s+string|signing|secret[_\s-]?key|private\s+key|ssh\s+key|client\s+secret|passwd|encryption\s+key|master\s+key|"
    r"\bcredentials?\b|\bpasswords?\b|server\s+config|application\s+config|app(?:lication)?\s+secrets?",
    re.IGNORECASE,
)
_SYSTEM_INTERNALS = re.compile(
    r"\b(?:system\s+prompt|developer\s+prompt|hidden\s+(?:prompt|tools?|instructions?|functions?|commands?|capabilities)|"
    r"tool\s+list|internal\s+tools?|your\s+(?:instructions|prompt|rules|configuration|source\s+code|system\s+message)|"
    r"source\s+code|backend\s+(?:environment|config(?:uration)?|code)|stack\s+trace|schema\s+dump|raw\s+sql|the\s+sql\s+(?:you|that)|"
    r"initial\s+instructions?|pre-?prompt|system\s+message|system\s+instructions|guardrail\s+config|"
    # Server/application configuration dumps are internals, never HR policy answers.
    r"(?:server|application|app|backend|runtime|deployment|host)\s+config(?:uration)?(?:\s+files?)?|"
    r"config(?:uration)?\s+(?:files?|settings?|values?|dump|contents?)|"
    r"contents?\s+of\s+the\s+(?:server|application|app|backend)\s+config(?:uration)?)\b",
    re.IGNORECASE,
)
_REVEAL_VERBS = re.compile(
    r"\b(?:show|reveal|print|display|dump|cat|echo|output|return|give|share|send|expose|leak|list|tell\s+me|what\s+is|what's|what\s+are|"
    r"paste|include|append|attach|export|extract|read|fetch|retrieve|browse|pull|compile|download|copy|forward|email|"
    r"disclose|provide|spit\s+out|repeat|recite|quote|print\s+out|log|write\s+out|which|where|need|want)\b",
    re.IGNORECASE,
)

_DESTRUCTIVE_SQL = re.compile(
    r"\btruncate\b|\bdrop\s+(?:table|database|schema|column|index|view)\b|\bdelete\s+from\b|\bupdate\s+\w+\s+set\b|"
    r"\binsert\s+into\b|\balter\s+table\b|\bgrant\s+(?:all|select|update|delete|insert)\b|\brm\s+-rf?\b|"
    r"\brecreate\s+(?:it\s+|them\s+)?from\s+(?:a\s+)?backup\b|\brestore\s+from\s+(?:a\s+)?backup\b|\breset\s+the\s+(?:database|table|db)\b|"
    r"\bwipe\s+(?:the\s+)?(?:database|db|tables?|records?|data)\b|\bexec(?:ute)?\s+sql\b|\brun\s+(?:this\s+)?sql\b",
    re.IGNORECASE,
)
_TABLE_IDENTIFIERS = re.compile(
    r"\b(?:leave_balances?|leave_requests?|policy_documents|policy_chunks|audit_logs?|project_assignments|\w+_table|\w+\s+table)\b",
    re.IGNORECASE,
)

_BULK_SCOPE = re.compile(
    r"\b(?:all|every|everyone(?:'s)?|everybody(?:'s)?|entire|whole|complete|full|company-?wide|organi[sz]ation-?wide|org-?wide|"
    r"across\s+the\s+(?:company|org\w*|business|firm)|each\s+employee|all\s+staff|the\s+workforce|all\s+users|other\s+users|every\s+user|"
    r"each\s+person|every\s+person|everyone\s+else|the\s+whole\s+(?:company|org\w*))\b",
    re.IGNORECASE,
)
_SENSITIVE_IDENTIFIERS = re.compile(
    r"\b(?:ssns?|social\s+security(?:\s+numbers?)?|national\s+(?:id|insurance|identity)(?:\s+numbers?)?|nric|passports?(?:\s+numbers?)?|"
    r"tax\s+(?:ids?|numbers?|file\s+numbers?)|bank\s+(?:accounts?|details|numbers?)|ibans?|sort\s+codes?|routing\s+numbers?|"
    r"(?:personal|private|home)\s+e-?mails?(?:\s+addresses?)?|dates?\s+of\s+birth|birth\s*dates?|dobs?|birthdays?|"
    r"medical\s+(?:records?|notes?|history|files?|certificates?|conditions?|diagnos\w*|details|information)|diagnos\w*|"
    r"health\s+(?:records?|data|conditions?|information)|next\s+of\s+kin|emergency\s+contacts?|disabilit\w+|religion|ethnicity|pii|"
    r"biometric\w*|immigration\s+status|visa\s+status)\b",
    re.IGNORECASE,
)
_PERSONAL_FIELDS = re.compile(
    r"\b(?:phone\s+numbers?|mobile\s+numbers?|home\s+addresses?|addresses|personal\s+(?:data|details|information|records?|contact\s+details)|"
    r"contact\s+details|e-?mails?(?:\s+addresses?)?|private\s+(?:data|details|information|records?))\b",
    re.IGNORECASE,
)
_COMPENSATION_BULK = re.compile(
    r"\b(?:compensation\s+(?:workbook|file|export|spreadsheet|csv|dump|data\s+export|table|sheet)|salary\s+(?:workbook|file|export|spreadsheet|csv|dump|sheet)|"
    r"payroll\s+(?:file|export|workbook|dump|csv|spreadsheet|sheet|register|table|ledger|data|records?|listing)|"
    r"(?:full|whole|entire|complete)\s+(?:compensation|salary|payroll)\s+\w+|"
    r"(?:compensation|salary|payroll)\s+(?:table|register|ledger|listing|database|data)\s+(?:for|of|across)?\s*(?:all|every|everyone|the\s+(?:whole|entire|company|org\w*))|"
    r"(?:all|every)\s+(?:employee\s+)?(?:salary|compensation|performance)\s+(?:records?|data|figures?|ratings?)|"
    r"salary\s+bands?\s+(?:for|of|across)\s+(?:all|every|everyone|the\s+(?:whole|entire))|organi[sz]ation-?wide\s+salary\s+bands?)\b",
    re.IGNORECASE,
)
_EXPORT_VERBS = re.compile(
    r"\b(?:export|dump|download|compile|csv|spreadsheet|workbook|excel|extract|bulk|batch|scrape|harvest|"
    r"pull\s+(?:all|every|everyone)|list\s+(?:all|every|everyone)|generate\s+(?:a\s+)?(?:file|report|list|table)\s+of\s+(?:all|every|everyone)|"
    r"append|attach|include)\b",
    re.IGNORECASE,
)

_CROSS_USER = re.compile(
    r"\b(?:(?:other|another|all|every|previous|prior|different|the\s+last|someone\s+else'?s?|anyone\s+else'?s?|everyone'?s?)\s+"
    r"(?:users?'?|people'?s?|employees?'?|sessions?'?|accounts?'?|colleagues?'?)\s*(?:'s)?\s*(?:private\s+)?"
    r"(?:chats?|conversations?|sessions?|messages?|memory|history|transcripts?|logs?|prompts?|questions?|queries)|"
    r"(?:conversation|chat|session|assistant|bot)\s+(?:memory|history|logs?|transcripts?|records?)\s+(?:for|of|from)\s+(?:other|another|all|every|different|everyone|user\s+\w+|[a-z][\w.-]+)|"
    r"(?:open|show|dump|reveal|print|read|export|fetch|retrieve)\s+(?:the\s+)?(?:conversation|chat|session)\s+history\s+(?:for|of|from)\s+(?:user\s+)?\w+|"
    r"dump\s+(?:the\s+|all\s+|their\s+|his\s+|her\s+)?(?:conversation|chat|session|messages?|questions?|queries|transcripts?)|"
    r"what\s+(?:did|has|have)\s+(?:the\s+)?(?:other|another|others?|previous|prior|last)\s+(?:users?|people|employees?|accounts?)\s+(?:ask(?:ed|ing|s)?|said|searched|typed|requested)|"
    r"what\s+(?:the\s+)?(?:other|another|previous|prior|last)\s+(?:users?|people|employees?|accounts?)\s+(?:ask(?:ed|ing|s)?|said|searched|typed|requested)|"
    r"what\s+did\s+[a-z][\w.-]*(?:\s+[a-z][\w.-]+)?\s+ask(?:\s+about)?|"
    r"(?:show|dump|reveal|print|read|export)\s+(?:me\s+)?[a-z][\w.-]*(?:\s+[a-z][\w.-]*)?'s\s+(?:previous\s+|prior\s+|last\s+)?"
    r"(?:chat\s+|conversation\s+|session\s+|assistant\s+)?(?:transcript|history|memory|messages?|questions?|queries)|"
    r"(?:their|his|her)\s+(?:previous\s+|prior\s+|last\s+)?(?:chat|conversation|session)\s+(?:transcript|history|memory|log)|"
    r"(?:their|his|her)\s+session\s+(?:history|memory|transcript|log)|"
    r"(?:replay|show|dump|reveal|print|read)\s+(?:the\s+)?(?:chat|conversation|session)\s+(?:history|memory|logs?|transcripts?)\s+(?:of|from|for)\s+(?:other|another|all|every|previous|different|user\s+\w+|\w+)|"
    r"memory\s+(?:for|of|from)\s+(?:other|another|all|every)\s+users?|(?:other|another|previous|prior)\s+users?'?\s+(?:data|memory|history|session|questions?|chats?))\b",
    re.IGNORECASE,
)

_EMBEDDED_MARKERS = re.compile(
    r"(?:\bhidden\s+instructions?\b|\bsecret\s+instructions?\b|\bsystem\s*:|\bdeveloper\s*:|\bassistant\s*:|\bnew\s+instructions?\s*:|"
    r"\bignore\s+the\s+above\b|\bafter\s+(?:answering|that|you\s+answer|responding)\b|\bthen\s+(?:also\s+)?(?:append|include|add|attach|dump|print)\b|"
    r"\badditionally\s+(?:append|include|dump|print|show)\b|\balso\s+(?:secretly|quietly|silently)\b|\bp\.?s\.?\s*:|\[inst\]|<\|im_start\|>|"
    r"###\s*instruction|\bwhile\s+(?:you'?re|you\s+are)\s+at\s+it\b|\bon\s+the\s+side\b|\bas\s+a\s+bonus\b|"
    r"\bin\s+addition,?\s+(?:also\s+)?(?:print|dump|show|cat|reveal)\b)",
    re.IGNORECASE,
)

_DATA_ASK = re.compile(
    r"\b(?:show|list|give|return|export|dump|pull|fetch|retrieve|browse|read|access|view|see|get|display|reveal|print|compile|send|share|"
    r"tell\s+me|what\s+(?:is|are)|how\s+much|how\s+many|look\s+up|find|search|open|download|include|append|quote|provide|"
    r"release|hand\s+(?:over|me)|turn\s+over|unlock|forward|email|attach|produce|generate|prepare|run|query)\b",
    re.IGNORECASE,
)
_PROTECTED_ASK = re.compile(
    r"\b(?:all\s+employees?|every\s+employee|everyone|everybody|entire|whole\s+(?:company|org\w*)|company-?wide|organi[sz]ation-?wide|"
    r"compensation|salar(?:y|ies)|payroll|pay\s+bands?|salary\s+bands?|performance\s+ratings?|attendance\s+(?:for|of)\s+(?:all|every|other)|"
    r"other\s+(?:employees?|people|teams?|departments?)|another\s+(?:employee|person|team|department)|someone\s+else|colleague|"
    r"hr\s+(?:tables?|database|data|records?)|every\s+(?:hr\s+)?table|all\s+(?:hr\s+)?tables|database|records?\s+of\s+\w+|workbook|export|dump)\b",
    re.IGNORECASE,
)
_NAME_POSSESSIVE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'s\b")


def _near(text: str, left: str, right: str, gap: int = 40) -> str | None:
    """Return the matched span when `left` is followed within `gap` chars (same clause) by `right`."""
    match = re.search(rf"\b{left}\b[^.;\n]{{0,{gap}}}?\b{right}\b", text, re.IGNORECASE)
    return match.group(0) if match else None


@dataclass
class SecurityFinding:
    category: str
    evidence: list[str] = field(default_factory=list)
    denial_intent: str = "unauthorized_access"
    reason: str = ""


@dataclass
class SecurityDecision:
    findings: list[SecurityFinding] = field(default_factory=list)
    embedded_instruction: bool = False

    @property
    def denied(self) -> bool:
        return bool(self.findings)

    @property
    def categories(self) -> list[str]:
        return [item.category for item in self.findings]

    @property
    def intent(self) -> str | None:
        if not self.findings:
            return None
        # A destructive instruction is an unsupported mutation only when it is the sole problem;
        # mixed with bypass / secret / private asks it is an authorization denial.
        if all(item.category == "destructive_operation" for item in self.findings):
            return "unsupported_mutation"
        return "unauthorized_access"

    @property
    def reason(self) -> str:
        if not self.findings:
            return ""
        seen: list[str] = []
        for item in self.findings:
            if item.reason and item.reason not in seen:
                seen.append(item.reason)
        body = " ".join(seen)
        tail = (
            " This is not allowed. The assistant will not do that, and it did not run any HR record lookup, "
            "report, or database tool for this message."
        )
        if self.embedded_instruction:
            tail += (
                " The message combined an ordinary question with an instruction that is not allowed, so the "
                "whole message was refused instead of answering only the allowed part."
            )
        return body + tail

    def as_audit(self) -> dict:
        return {
            "categories": self.categories,
            "evidence": {item.category: item.evidence for item in self.findings},
            "embedded_instruction": self.embedded_instruction,
        }


# ---------------------------------------------------------------------------
# Category detectors. Each returns a finding or None.
# ---------------------------------------------------------------------------


def _privilege_escalation(text: str) -> SecurityFinding | None:
    evidence = []
    span = _near(text, _ESCALATION_VERBS, _PRIVILEGED_ROLES, 40)
    if span:
        # "give me the HR policy" is not an escalation: weak verbs need a privileged noun that is
        # about access, not a department name.
        weak = re.match(r"(?i)(?:make|grant|give)\s+me", span)
        if not weak or re.search(
            r"(?i)\b(?:root|sudo|superuser|super-?admin\w*|administrator|admin|hr[_\s-]?admin\w*|unrestricted|full\s+access|"
            r"all\s+permissions|elevated|privileged|higher\s+\w+|rights|privileges?|access|permissions?|role|powers?)\b",
            span,
        ):
            evidence.append(span)
    # Explicit role-switch / identity-override instructions even when word order varies.
    role_switch = re.search(
        r"(?i)\b(?:switch|change|set|assume|take|adopt)\b.{0,40}\b(?:role|permissions?|privileges?|access)\b.{0,40}"
        r"\b(?:hr[_\s-]?admin(?:istrator)?|administrator|admin|manager|root|superuser)\b|"
        r"\b(?:switch|change|set|assume|take|adopt)\b.{0,20}\b(?:to|into|as)\b.{0,20}"
        r"\b(?:hr[_\s-]?admin(?:istrator)?|administrator|admin|manager|root|superuser)\b|"
        r"\b(?:hr[_\s-]?admin(?:istrator)?|administrator|admin)\b.{0,30}\b(?:role|permissions?|privileges?)\b|"
        r"\b(?:role|permissions?|privileges?)\b.{0,20}\b(?:to|as|into)\b.{0,20}"
        r"\b(?:hr[_\s-]?admin(?:istrator)?|administrator|admin|manager|root|superuser)\b|"
        r"\bwithout\s+re-?authenticat(?:ing|ion)\b|"
        r"\b(?:act|operate|continue)\b.{0,30}\b(?:as|with)\b.{0,20}"
        r"\b(?:hr[_\s-]?admin(?:istrator)?|administrator)\b.{0,40}\b(?:permissions?|privileges?|role|access)\b|"
        r"\b(?:grant|give)\b.{0,20}\b(?:me|us)\b.{0,30}\b(?:hr[_\s-]?admin(?:istrator)?|admin)\b.{0,20}"
        r"\b(?:role|permissions?|privileges?|access)\b|"
        r"\b(?:disable|bypass|ignore|override)\b.{0,40}\b(?:authorization|rbac|access control|permission check)",
        text,
    )
    if role_switch:
        evidence.append(role_switch.group(0))
    mode = _MODE_TOKENS.search(text)
    if mode:
        evidence.append(mode.group(0))
    root = _ROOT_TOKENS.search(text)
    if root:
        evidence.append(root.group(0))
    if not evidence:
        return None
    return SecurityFinding(
        "privilege_escalation",
        evidence,
        reason=(
            "This request asks the assistant to change or elevate the signed-in role or to operate in a "
            "privileged mode. Roles come only from the authenticated session and cannot be changed in chat."
        ),
    )


def _bypass_security_controls(text: str) -> SecurityFinding | None:
    evidence = []
    span = _near(text, _DISABLE_VERBS, _SECURITY_CONTROLS, 45)
    if span:
        evidence.append(span)
    for hit in guard.detect_injection(text):
        evidence.append(f"injection:{hit[:60]}")
    if not evidence:
        return None
    return SecurityFinding(
        "bypass_security_controls",
        evidence,
        reason=(
            "This request asks to disable, override, or ignore the application's authorization and security "
            "controls. Those controls run in the authorization service, not in the conversation, and no role "
            "including HR Admin can switch them off from chat."
        ),
    )


def _secrets_credentials(text: str) -> SecurityFinding | None:
    secret = _SECRET_TOKENS.search(text)
    config_dump = re.search(
        r"\b(?:server|application|app|backend|runtime|deployment|host)\s+config(?:uration)?\b|"
        r"\bconfig(?:uration)?\s+(?:files?|settings?|values?|dump|contents?)\b|"
        r"\bcontents?\s+of\s+(?:the\s+)?(?:server|application|app|backend)\s+config",
        text,
        re.IGNORECASE,
    )
    if not secret and not config_dump:
        return None
    token = secret.group(0) if secret else config_dump.group(0)
    # Bare "credentials"/"password" require a reveal verb or config/server framing so ordinary
    # leave-password policy questions are not denied. Server/app config dumps always deny.
    if config_dump:
        return SecurityFinding(
            "secrets_credentials",
            [token],
            reason=(
                "This request asks for credentials, secrets, tokens, or server/application configuration. "
                "The assistant never reveals configuration secrets, credentials, or server files."
            ),
        )
    if not _SECRET_ALWAYS.search(token) and not _REVEAL_VERBS.search(text) and not text.strip().endswith("?"):
        return None
    # Avoid false positives on HR password-reset *policy* questions that are not asking to dump secrets.
    if re.fullmatch(r"passwords?", token.strip(), re.IGNORECASE) and re.search(
        r"\b(policy|policies|reset process|how (?:do|can) i reset)\b", text, re.IGNORECASE
    ):
        return None
    return SecurityFinding(
        "secrets_credentials",
        [token],
        reason=(
            "This request asks for credentials, secrets, tokens, or system files. The assistant never reveals "
            "configuration secrets, credentials, or server files."
        ),
    )


def _system_internals(text: str) -> SecurityFinding | None:
    probe = _SYSTEM_INTERNALS.search(text)
    if not probe:
        return None
    if not _REVEAL_VERBS.search(text) and not re.search(
        r"\b(?:rewrite|change|edit|modify|replace|update|override)\b", text, re.IGNORECASE
    ):
        return None
    return SecurityFinding(
        "system_internals_probe",
        [probe.group(0)],
        reason=(
            "This request asks to reveal or rewrite the assistant's internal instructions, tooling, "
            "or server/application configuration."
        ),
    )


def _destructive_operation(text: str, role: str) -> SecurityFinding | None:
    evidence = []
    sql = _DESTRUCTIVE_SQL.search(text)
    if sql:
        evidence.append(sql.group(0))
    hr_record_delete = bool(
        role == "hr_admin"
        and re.search(r"\b(delete|remove)\b", text, re.IGNORECASE)
        and re.search(r"\bemployee record\b", text, re.IGNORECASE)
        and not sql
    )
    if not hr_record_delete and guard.unsupported_destructive_request(text):
        evidence.append("destructive verb + HRIS target")
    if not evidence and not hr_record_delete:
        verb = re.search(
            r"\b(delete|remove|erase|wipe|destroy|purge|drop|truncate|nullify|zero\s+out|clear\s+out|overwrite)\b",
            text,
            re.IGNORECASE,
        )
        table = _TABLE_IDENTIFIERS.search(text)
        if verb and table:
            evidence.append(f"{verb.group(0)} {table.group(0)}")
    if not evidence:
        return None
    return SecurityFinding(
        "destructive_operation",
        evidence,
        denial_intent="unsupported_mutation",
        reason=(
            "Destructive changes such as deleting, truncating, or rewriting attendance, leave, payroll, or employee "
            "records are not available through this assistant. No database mutation was performed. "
            "Use the authorized HR correction process instead."
        ),
    )


def _other_person_private(text: str, role: str, mentioned_other: bool) -> SecurityFinding | None:
    if role == "hr_admin":
        return None
    evidence = []
    if guard.other_employee_private_request(text):
        evidence.append("other person + private record")
    if mentioned_other and role == "employee" and re.search(
        r"\b(medical|diagnos\w*|sick\s+leave|sick\s+file|leave\s+history|leave\s+balance|balance|salary|compensation|phone|address|"
        r"personal|attendance\s+history|attendance\s+records?|certificate|notes?|rating|performance)\b",
        text,
        re.IGNORECASE,
    ):
        evidence.append("named employee + private field")
    if _SENSITIVE_IDENTIFIERS.search(text) and _NAME_POSSESSIVE.search(text):
        evidence.append("named possessive + sensitive identifier")
    if not evidence:
        return None
    return SecurityFinding(
        "other_person_private_data",
        evidence,
        reason=(
            "You are not authorized to view another employee's attendance, medical, sick leave, leave balance, or other "
            "private HR details. Personal records are limited to the signed-in employee's own data, and access to "
            "another employee's private records is limited to HR administrators. "
            "Framing it as a policy question does not change what is being asked for, and the assistant will not "
            "substitute policy text or your own records for that request."
        ),
    )


def _bulk_personal_export(text: str, role: str) -> SecurityFinding | None:
    evidence = []
    bulk = _BULK_SCOPE.search(text)
    sensitive = _SENSITIVE_IDENTIFIERS.search(text)
    personal = _PERSONAL_FIELDS.search(text)
    export = _EXPORT_VERBS.search(text)
    if sensitive and (bulk or export):
        evidence.append(f"{(bulk or export).group(0)} + {sensitive.group(0)}")
    # Explicit "all salaries" style asks keep their dedicated confidential_lookup denial path.
    if role != "hr_admin" and not guard.wants_bulk_export(text):
        comp = _COMPENSATION_BULK.search(text)
        if comp and (export or bulk or _DATA_ASK.search(text)):
            evidence.append(comp.group(0))
    if role != "hr_admin":
        if personal and bulk:
            evidence.append(f"{bulk.group(0)} + {personal.group(0)}")
        if guard.broad_private_data_request(text):
            evidence.append("broad private data request")
    elif personal and bulk and export:
        # HR may look up an individual's contact fields, but a bulk export/dump of everyone's
        # personal contact details is a data-exfiltration pattern for every role.
        evidence.append(f"{export.group(0)} + {bulk.group(0)} + {personal.group(0)}")
    if guard.export_dump_request(text):
        evidence.append("export/dump of HR database")
    if not evidence:
        return None
    return SecurityFinding(
        "bulk_personal_export",
        evidence,
        reason=(
            "This request asks for a bulk export of personal identifiers, private contact details, or compensation "
            "records across employees. Bulk personal and compensation exports are blocked in the assistant for every role."
        ),
    )


def _cross_user_memory(text: str) -> SecurityFinding | None:
    hit = _CROSS_USER.search(text)
    hidden = re.search(r"\b(?:hidden|internal|secret|undocumented)\s+(?:tools?|functions?|commands?)\b", text, re.IGNORECASE)
    # Named person's transcript / prior assistant chat / dump their questions.
    named = re.search(
        r"\b(?:show|dump|reveal|print|read|export|open)\b.{0,80}\b(?:transcript|chat history|conversation history|"
        r"previous chat|prior chat|last conversation|her questions|his questions|their questions)\b",
        text,
        re.IGNORECASE,
    ) or re.search(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'s\b.{0,40}\b(?:transcript|chat|conversation|questions?|memory|history)\b",
        text,
    ) or re.search(
        r"\bwhat did\b.{0,40}\bask\b.{0,40}\b(?:conversation|assistant|chat|you)\b",
        text,
        re.IGNORECASE,
    )
    if not hit and not hidden and not named:
        return None
    evidence = [match.group(0) for match in (hit, hidden, named) if match]
    return SecurityFinding(
        "cross_user_memory",
        evidence,
        reason=(
            "This request asks for other users' conversations, memory, or sessions, or for hidden tooling. "
            "Conversation memory is isolated per signed-in user and is never shared."
        ),
    )


def _out_of_hierarchy(text: str, role: str) -> SecurityFinding | None:
    if role == "hr_admin":
        return None
    if not guard.out_of_hierarchy_request(text):
        return None
    return SecurityFinding(
        "out_of_hierarchy",
        ["explicit non-report / outside-scope target"],
        reason=(
            "You do not have permission to access attendance or employee records outside your authorized "
            "reporting scope. The assistant will not substitute unrelated personal data for that request."
        ),
    )


def _scope_widening(text: str, role: str) -> SecurityFinding | None:
    if role == "hr_admin":
        return None
    evidence = []
    span = _near(text, _WIDEN_VERBS, _QUERY_FILTERS, 40)
    if span:
        evidence.append(span)
    phrase = _SCOPE_WIDEN_PHRASES.search(text)
    if phrase:
        evidence.append(phrase.group(0))
    if re.search(r"\bwhere\s+clause\b|\bsql\b", text, re.IGNORECASE) and re.search(
        r"\b(all|every|whole|entire|other)\s+(departments?|teams?|employees?|company|org\w*)\b", text, re.IGNORECASE
    ):
        evidence.append("query rewritten to all departments")
    # Expand/cover team dashboards to every department / whole company / Northstar.
    if re.search(r"\b(expand|widen|broaden|extend|cover|include)\b", text, re.IGNORECASE) and re.search(
        r"\b((?:every|all)\s+departments?|(?:entire|whole)\s+(?:company|organi[sz]ation|northstar)|"
        r"across\s+(?:northstar|the\s+(?:company|organi[sz]ation))|organization[-\s]?wide|company[-\s]?wide)\b",
        text,
        re.IGNORECASE,
    ):
        evidence.append("team scope expanded to every department / organization")
    if not evidence:
        return None
    return SecurityFinding(
        "scope_widening",
        evidence,
        reason=(
            "This request asks to widen a team-scoped report beyond your reporting line or to change the query "
            "filters that enforce that scope. Organization-wide analytics are limited to HR administrators, and "
            "the scope filter cannot be removed from chat. Managers remain limited to their authorized reporting scope."
        ),
    )


def _hr_only_compensation(text: str, role: str) -> SecurityFinding | None:
    """Managers/employees cannot run HR-only workforce compensation analytics."""
    if role == "hr_admin":
        return None
    match = _HR_ONLY_COMPENSATION.search(text)
    if not match and not (
        re.search(r"\bcompensation\b", text, re.IGNORECASE)
        and re.search(r"\b(percentile|percentiles|workforce|hr[-\s]?only)\b", text, re.IGNORECASE)
        and re.search(r"\b(report|analytics|dashboard|run|generate)\b", text, re.IGNORECASE)
    ):
        return None
    evidence = [match.group(0)] if match else ["hr-only compensation analytics"]
    return SecurityFinding(
        "hr_only_compensation",
        evidence,
        reason=(
            "Workforce compensation percentile and HR-only compensation analytics are limited to HR administrators. "
            "The authenticated role cannot access that protected compensation capability."
        ),
    )


def _social_engineering(text: str, role: str) -> SecurityFinding | None:
    if role == "hr_admin":
        return None
    claim = _AUTHORITY_CLAIMS.search(text)
    if not claim:
        return None
    protected = _PROTECTED_ASK.search(text)
    if not protected or not _DATA_ASK.search(text):
        return None
    return SecurityFinding(
        "social_engineering",
        [claim.group(0), protected.group(0)],
        reason=(
            "A verbal or claimed authorization from a director, manager, or compliance process does not change the "
            "signed-in role. Access decisions use the authenticated session only."
        ),
    )


def _mixed_with_legitimate(text: str) -> bool:
    """A benign HR ask (leave, holidays, balance, attendance) sits next to a denied instruction."""
    return bool(
        re.search(r"\b(my\s+leave|my\s+balance|my\s+attendance|holidays?|my\s+profile|my\s+hours|my\s+requests?)\b", text, re.IGNORECASE)
        and re.search(r"\b(also|and|then|after|while|plus|additionally|as\s+well)\b", text, re.IGNORECASE)
    )


def assess_security(message: str, *, role: str, mentioned_other: bool = False) -> SecurityDecision:
    """Compose a deny/allow decision for one message.

    'Not denied' means no security finding; it does not mean the request is authorized.
    Role-based permission for ordinary HR data still runs in authorize().
    """
    text = message or ""
    findings: list[SecurityFinding] = []
    for finding in (
        _destructive_operation(text, role),
        _privilege_escalation(text),
        _bypass_security_controls(text),
        _secrets_credentials(text),
        _system_internals(text),
        _cross_user_memory(text),
        _bulk_personal_export(text, role),
        _hr_only_compensation(text, role),
        _other_person_private(text, role, mentioned_other),
        _out_of_hierarchy(text, role),
        _scope_widening(text, role),
        _social_engineering(text, role),
    ):
        if finding is not None:
            findings.append(finding)
    embedded = bool(findings) and bool(_EMBEDDED_MARKERS.search(text) or _mixed_with_legitimate(text))
    return SecurityDecision(findings=findings, embedded_instruction=embedded)


def role_agnostic_violation(message: str) -> bool:
    """True when a message is denied for every role (used by guard.hard_security_violation)."""
    text = message or ""
    return any(
        finding is not None
        for finding in (
            _privilege_escalation(text),
            _bypass_security_controls(text),
            _secrets_credentials(text),
            _system_internals(text),
            _cross_user_memory(text),
        )
    )

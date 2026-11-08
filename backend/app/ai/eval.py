from sqlalchemy.orm import Session

from app.ai.retrieval import retrieve
from app.models import Department, Employee

EVAL_QUESTIONS = [
    {
        "question": "How many days of annual leave are employees entitled to each year?",
        "expected": ["annual-leave-v3"],
        "forbidden": ["annual-leave-v2"],
    },
    {
        "question": "What notice period applies before taking annual leave?",
        "expected": ["annual-leave-v3"],
        "forbidden": ["annual-leave-v2"],
    },
    {
        "question": "What is the overtime premium and after how many hours is it paid?",
        "expected": ["overtime-v3", "overtime-memo-conflict"],
        "forbidden": [],
        "expect_conflict": True,
    },
    {
        "question": "How long is primary caregiver parental leave?",
        "expected": ["parental-leave-v1"],
        "forbidden": [],
    },
    {
        "question": "What happens to leave approval routing after an employee transfer?",
        "expected": ["employee-transfers-v1"],
        "forbidden": [],
    },
    {
        "question": "When is People Operations co-approval required for leave?",
        "expected": ["leave-approval-v2"],
        "forbidden": [],
    },
    {
        "question": "How many remote days are allowed each week?",
        "expected": ["remote-work-v2"],
        "forbidden": [],
    },
    {
        "question": "When is a medical note required for sick leave?",
        "expected": ["sick-leave-v2"],
        "forbidden": [],
    },
]


def evaluate_retrieval(db: Session, *, role: str = "hr_admin", department: str | None = None) -> dict:
    reciprocal = []
    precisions = []
    recalls = []
    cases = []
    for item in EVAL_QUESTIONS:
        result = retrieve(db, item["question"], role=role, department=department, location="global")
        ids = [citation["policy_id"] for citation in result["citations"]]
        expected = item["expected"]
        ranks = [ids.index(doc) + 1 for doc in expected if doc in ids]
        reciprocal.append(1 / min(ranks) if ranks else 0)
        retrieved = set(ids)
        relevant = set(expected)
        precision = len(retrieved & relevant) / len(retrieved) if retrieved else 0
        recall = len(retrieved & relevant) / len(relevant) if relevant else 0
        precisions.append(precision)
        recalls.append(recall)
        forbidden = [doc for doc in item.get("forbidden", []) if doc in ids]
        cases.append(
            {
                "question": item["question"],
                "expected": expected,
                "retrieved": ids,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "reciprocal_rank": round(reciprocal[-1], 3),
                "forbidden_hits": forbidden,
                "confidence": result["confidence"],
                "conflict": bool(result["conflicts"]),
            }
        )
    count = len(cases) or 1
    return {
        "k": 4,
        "cases": len(cases),
        "precision_at_k": round(sum(precisions) / count, 3),
        "recall_at_k": round(sum(recalls) / count, 3),
        "mrr": round(sum(reciprocal) / count, 3),
        "results": cases,
        "notes": "Precision and recall use the documents kept after the relative score threshold, capped at 4.",
    }


def employee_department(db: Session, email: str) -> str | None:
    employee = db.query(Employee).filter(Employee.email == email).one_or_none()
    if employee is None:
        return None
    department = db.get(Department, employee.department_id)
    return None if department is None else department.name

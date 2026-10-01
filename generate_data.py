"""
Synthetic data generator for the IAM Access Governance Analytics project.

Creates a fictional dataset for "Lone Star Health", a regional healthcare
organization, with realistic identity-risk patterns baked in:
  - orphaned accounts (terminated users with still-enabled accounts)
  - dormant accounts (no login in 90+ days)
  - privilege creep (manual/exception grants beyond role baselines)
  - segregation-of-duties (SoD) conflicts
  - an incomplete access-certification campaign

All data is SYNTHETIC and seeded (seed=42) so every run reproduces the
same dataset. No real personal data is used or implied.

Reference date is fixed at 2026-09-25 so findings are stable across runs.

Tables produced (CSV, in ./data):
  users.csv            one row per person
  accounts.csv         one row per system account
  entitlements.csv     one row per granted entitlement
  role_baselines.csv   expected entitlements per (department, job_title)
  access_events.csv    flagged anomalous access events
  certifications.csv   quarterly access-certification campaign results
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

SEED = 42
REF_DATE = date(2026, 9, 25)
N_USERS = 600

rng = np.random.default_rng(SEED)

FIRST = ["Maria", "James", "Aisha", "Robert", "Linda", "Michael", "Sofia",
         "David", "Priya", "Jose", "Emily", "Daniel", "Fatima", "Chris",
         "Nancy", "Kevin", "Rosa", "Brian", "Mei", "Carlos", "Anna",
         "George", "Lucia", "Peter", "Grace", "Omar", "Diana", "Victor",
         "Hannah", "Samuel"]
LAST = ["Garcia", "Smith", "Khan", "Johnson", "Lopez", "Brown", "Martinez",
        "Davis", "Patel", "Hernandez", "Wilson", "Anderson", "Ali", "Taylor",
        "Thomas", "Moore", "Jackson", "Lee", "Chen", "Rivera", "Kim",
        "Walker", "Torres", "Nguyen", "Hill", "Adams", "Baker", "Carter"]

# Department -> list of (job_title, headcount_weight, is_manager_title)
ORG = {
    "Nursing": [("RN - Staff Nurse", 40, False), ("Nurse Manager", 4, True),
                ("CNA", 18, False)],
    "Billing": [("Billing Specialist", 22, False), ("Billing Supervisor", 3, True),
                ("Claims Analyst", 10, False)],
    "Pharmacy": [("Pharmacist", 12, False), ("Pharmacy Tech", 16, False),
                 ("Pharmacy Manager", 2, True)],
    "Radiology": [("Radiology Tech", 14, False), ("Radiologist", 6, False)],
    "IT": [("Help Desk Analyst", 8, False), ("Systems Administrator", 6, False),
           ("IT Manager", 2, True), ("Security Analyst", 4, False)],
    "HR": [("HR Generalist", 6, False), ("HR Manager", 2, True),
           ("Payroll Specialist", 4, False)],
    "Patient Services": [("Scheduler", 14, False), ("Registration Clerk", 12, False),
                         ("Patient Services Supervisor", 3, True)],
    "Facilities": [("Maintenance Tech", 8, False), ("Facilities Manager", 1, True)],
}

SYSTEMS = ["EHR", "Billing System", "Pharmacy System", "Active Directory",
           "VPN", "HR Portal", "Radiology PACS", "Email"]

# Baseline entitlements per (department, job_title): (system, entitlement, risk)
BASELINES = {
    ("Nursing", "RN - Staff Nurse"): [("EHR", "Patient Records Read/Write", "High"),
                                      ("Active Directory", "Standard User", "Low"),
                                      ("Email", "Standard Mailbox", "Low")],
    ("Nursing", "CNA"): [("EHR", "Patient Records Read", "High"),
                         ("Active Directory", "Standard User", "Low"),
                         ("Email", "Standard Mailbox", "Low")],
    ("Nursing", "Nurse Manager"): [("EHR", "Patient Records Read/Write", "High"),
                                   ("EHR", "Clinical Schedule Admin", "Medium"),
                                   ("Active Directory", "Standard User", "Low"),
                                   ("Email", "Standard Mailbox", "Low"),
                                   ("HR Portal", "Team Time Approval", "Medium")],
    ("Billing", "Billing Specialist"): [("Billing System", "Claims Entry", "Medium"),
                                        ("Active Directory", "Standard User", "Low"),
                                        ("Email", "Standard Mailbox", "Low")],
    ("Billing", "Claims Analyst"): [("Billing System", "Claims Entry", "Medium"),
                                    ("Billing System", "Claims Reporting", "Medium"),
                                    ("Active Directory", "Standard User", "Low"),
                                    ("Email", "Standard Mailbox", "Low")],
    ("Billing", "Billing Supervisor"): [("Billing System", "Claims Entry", "Medium"),
                                        ("Billing System", "Claims Approve", "High"),
                                        ("Active Directory", "Standard User", "Low"),
                                        ("Email", "Standard Mailbox", "Low")],
    ("Pharmacy", "Pharmacist"): [("Pharmacy System", "Dispense Medication", "High"),
                                 ("EHR", "Medication Orders Read", "High"),
                                 ("Active Directory", "Standard User", "Low"),
                                 ("Email", "Standard Mailbox", "Low")],
    ("Pharmacy", "Pharmacy Tech"): [("Pharmacy System", "Inventory View", "Medium"),
                                    ("Active Directory", "Standard User", "Low"),
                                    ("Email", "Standard Mailbox", "Low")],
    ("Pharmacy", "Pharmacy Manager"): [("Pharmacy System", "Dispense Medication", "High"),
                                       ("Pharmacy System", "Inventory Adjust", "High"),
                                       ("Active Directory", "Standard User", "Low"),
                                       ("Email", "Standard Mailbox", "Low")],
    ("Radiology", "Radiology Tech"): [("Radiology PACS", "Imaging View/Capture", "High"),
                                      ("Active Directory", "Standard User", "Low"),
                                      ("Email", "Standard Mailbox", "Low")],
    ("Radiology", "Radiologist"): [("Radiology PACS", "Imaging View/Capture", "High"),
                                   ("EHR", "Patient Records Read", "High"),
                                   ("Active Directory", "Standard User", "Low"),
                                   ("Email", "Standard Mailbox", "Low")],
    ("IT", "Help Desk Analyst"): [("Active Directory", "Password Reset Rights", "Medium"),
                                  ("VPN", "VPN Access", "Medium"),
                                  ("Email", "Standard Mailbox", "Low")],
    ("IT", "Systems Administrator"): [("Active Directory", "Domain Admin", "Critical"),
                                      ("VPN", "VPN Access", "Medium"),
                                      ("Email", "Standard Mailbox", "Low")],
    ("IT", "Security Analyst"): [("Active Directory", "Security Log Read", "High"),
                                 ("VPN", "VPN Access", "Medium"),
                                 ("Email", "Standard Mailbox", "Low")],
    ("IT", "IT Manager"): [("Active Directory", "Domain Admin", "Critical"),
                           ("VPN", "VPN Access", "Medium"),
                           ("Email", "Standard Mailbox", "Low"),
                           ("HR Portal", "Team Time Approval", "Medium")],
    ("HR", "HR Generalist"): [("HR Portal", "Employee Records Edit", "High"),
                              ("Active Directory", "Standard User", "Low"),
                              ("Email", "Standard Mailbox", "Low")],
    ("HR", "Payroll Specialist"): [("HR Portal", "Payroll Process", "High"),
                                   ("Active Directory", "Standard User", "Low"),
                                   ("Email", "Standard Mailbox", "Low")],
    ("HR", "HR Manager"): [("HR Portal", "Employee Records Edit", "High"),
                           ("HR Portal", "Payroll Approve", "Critical"),
                           ("Active Directory", "Standard User", "Low"),
                           ("Email", "Standard Mailbox", "Low")],
    ("Patient Services", "Scheduler"): [("EHR", "Scheduling", "Medium"),
                                        ("Active Directory", "Standard User", "Low"),
                                        ("Email", "Standard Mailbox", "Low")],
    ("Patient Services", "Registration Clerk"): [("EHR", "Patient Registration", "High"),
                                                 ("Active Directory", "Standard User", "Low"),
                                                 ("Email", "Standard Mailbox", "Low")],
    ("Patient Services", "Patient Services Supervisor"): [("EHR", "Scheduling", "Medium"),
                                                          ("EHR", "Patient Registration", "High"),
                                                          ("Active Directory", "Standard User", "Low"),
                                                          ("Email", "Standard Mailbox", "Low")],
    ("Facilities", "Maintenance Tech"): [("Active Directory", "Standard User", "Low"),
                                         ("Email", "Standard Mailbox", "Low")],
    ("Facilities", "Facilities Manager"): [("Active Directory", "Standard User", "Low"),
                                           ("Email", "Standard Mailbox", "Low"),
                                           ("HR Portal", "Team Time Approval", "Medium")],
}

# Toxic SoD pairs: (system_a, entitlement_a, system_b, entitlement_b, label)
SOD_PAIRS = [
    ("Billing System", "Claims Entry", "Billing System", "Claims Approve",
     "Create + approve claims"),
    ("Pharmacy System", "Dispense Medication", "Pharmacy System", "Inventory Adjust",
     "Dispense + adjust inventory"),
    ("HR Portal", "Employee Records Edit", "HR Portal", "Payroll Approve",
     "Edit records + approve payroll"),
    ("HR Portal", "Payroll Process", "HR Portal", "Payroll Approve",
     "Process + approve payroll"),
    ("Active Directory", "Domain Admin", "Active Directory", "Security Log Read",
     "Admin + audit log access"),
]

# High-value manual grants that represent privilege creep when given outside IT
CREEP_GRANTS = [
    ("EHR", "Patient Records Export", "Critical"),
    ("Billing System", "Claims Approve", "High"),
    ("Active Directory", "Domain Admin", "Critical"),
    ("Pharmacy System", "Inventory Adjust", "High"),
    ("HR Portal", "Payroll Approve", "Critical"),
    ("VPN", "VPN Access", "Medium"),
]


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(rng.integers(0, max(delta, 1))))


def build_users() -> pd.DataFrame:
    pool, weights = [], []
    for dept, roles in ORG.items():
        for title, w, is_mgr in roles:
            pool.append((dept, title, is_mgr))
            weights.append(w)
    weights = np.array(weights) / sum(weights)
    picks = rng.choice(len(pool), size=N_USERS, p=weights)

    rows = []
    for i, p in enumerate(picks):
        dept, title, is_mgr = pool[p]
        status = rng.choice(["Active", "Terminated", "On Leave"],
                            p=[0.88, 0.07, 0.05])
        hire = random_date(date(2015, 1, 1), date(2026, 6, 1))
        term = random_date(max(hire + timedelta(days=180), date(2024, 1, 1)),
                           date(2026, 9, 1)) if status == "Terminated" else pd.NaT
        rows.append({
            "user_id": f"U{i+1:04d}",
            "full_name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "department": dept,
            "job_title": title,
            "is_manager": is_mgr,
            "employment_status": status,
            "hire_date": hire.isoformat(),
            "termination_date": term.isoformat() if pd.notna(term) else "",
        })
    users = pd.DataFrame(rows)
    # assign managers within department
    users["manager_id"] = ""
    for dept in users["department"].unique():
        dept_users = users[users["department"] == dept]
        mgrs = dept_users[dept_users["is_manager"]]["user_id"].tolist()
        if mgrs:
            for uid in dept_users[~dept_users["is_manager"]]["user_id"]:
                users.loc[users["user_id"] == uid, "manager_id"] = rng.choice(mgrs)
    return users


def build_accounts(users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    aid = 0
    for _, u in users.iterrows():
        for system in SYSTEMS:
            # not everyone has every system; EHR/AD/Email near-universal
            prob = {"EHR": 0.75, "Active Directory": 0.98, "Email": 0.98,
                    "VPN": 0.35, "Billing System": 0.25, "Pharmacy System": 0.15,
                    "HR Portal": 0.30, "Radiology PACS": 0.12}[system]
            if rng.random() > prob:
                continue
            aid += 1
            terminated = u["employment_status"] == "Terminated"
            # ORPHANED pattern: most terminated users keep an enabled account
            if terminated:
                status = "Enabled" if rng.random() < 0.62 else "Disabled"
            else:
                status = "Disabled" if rng.random() < 0.03 else "Enabled"
            created = random_date(date.fromisoformat(u["hire_date"]),
                                  min(REF_DATE, date.fromisoformat(u["hire_date"]) + timedelta(days=60)))
            # DORMANT pattern: some active users haven't logged in for 90+ days
            if status == "Enabled" and not terminated and rng.random() < 0.10:
                last_login = random_date(date(2025, 6, 1), REF_DATE - timedelta(days=91))
            elif status == "Enabled":
                last_login = random_date(REF_DATE - timedelta(days=60), REF_DATE)
            else:
                last_login = random_date(created, REF_DATE - timedelta(days=30))
            rows.append({
                "account_id": f"A{aid:05d}",
                "user_id": u["user_id"],
                "system": system,
                "status": status,
                "is_privileged": system in ("Active Directory",) and rng.random() < 0.06,
                "is_service_account": False,
                "created_date": created.isoformat(),
                "last_login": last_login.isoformat(),
            })
    # a few service accounts (non-person, often forgotten)
    for s in range(12):
        aid += 1
        rows.append({
            "account_id": f"A{aid:05d}",
            "user_id": "",
            "system": rng.choice(["EHR", "Billing System", "Active Directory"]),
            "status": "Enabled" if rng.random() < 0.8 else "Disabled",
            "is_privileged": True,
            "is_service_account": True,
            "created_date": random_date(date(2019, 1, 1), date(2024, 1, 1)).isoformat(),
            "last_login": random_date(date(2026, 1, 1), REF_DATE).isoformat(),
        })
    return pd.DataFrame(rows)


def build_entitlements(users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eid = 0
    for _, u in users.iterrows():
        key = (u["department"], u["job_title"])
        for system, ent, risk in BASELINES.get(key, []):
            eid += 1
            rows.append({
                "entitlement_id": f"E{eid:06d}",
                "user_id": u["user_id"],
                "system": system,
                "entitlement": ent,
                "risk_level": risk,
                "granted_via": "Role",
                "granted_date": u["hire_date"],
            })
        # PRIVILEGE CREEP: manual grants beyond the role baseline
        if u["employment_status"] == "Active" and rng.random() < 0.09:
            system, ent, risk = CREEP_GRANTS[rng.integers(len(CREEP_GRANTS))]
            # skip if already held via role
            held = {(r["system"], r["entitlement"]) for r in rows if r["user_id"] == u["user_id"]}
            if (system, ent) not in held:
                eid += 1
                rows.append({
                    "entitlement_id": f"E{eid:06d}",
                    "user_id": u["user_id"],
                    "system": system,
                    "entitlement": ent,
                    "risk_level": risk,
                    "granted_via": rng.choice(["Manual", "Exception"]),
                    "granted_date": random_date(date(2024, 1, 1), REF_DATE).isoformat(),
                })
    ent = pd.DataFrame(rows)

    # SoD CONFLICTS: deliberately plant toxic pairs on a subset of users
    targets = ent[ent["user_id"].isin(
        users[users["department"].isin(["Billing", "Pharmacy", "HR"])]["user_id"]
    )]["user_id"].drop_duplicates().sample(frac=0.12, random_state=SEED)
    extra = []
    for uid in targets:
        dept = users.loc[users["user_id"] == uid, "department"].iloc[0]
        pair = {"Billing": SOD_PAIRS[0], "Pharmacy": SOD_PAIRS[1], "HR": SOD_PAIRS[2]}[dept]
        sa, ea, sb, eb, _ = pair
        held = set(ent[ent["user_id"] == uid][["system", "entitlement"]].itertuples(index=False, name=None))
        for s, e in [(sa, ea), (sb, eb)]:
            if (s, e) not in held:
                eid += 1
                risk = "High" if "Approve" in e or "Adjust" in e else "Medium"
                extra.append({"entitlement_id": f"E{eid:06d}", "user_id": uid,
                              "system": s, "entitlement": e, "risk_level": risk,
                              "granted_via": "Manual",
                              "granted_date": random_date(date(2024, 6, 1), REF_DATE).isoformat()})
    if extra:
        ent = pd.concat([ent, pd.DataFrame(extra)], ignore_index=True)
    return ent


def build_baselines() -> pd.DataFrame:
    rows = []
    for (dept, title), ents in BASELINES.items():
        for system, ent, risk in ents:
            rows.append({"department": dept, "job_title": title,
                         "system": system, "entitlement": ent,
                         "risk_level": risk})
    return pd.DataFrame(rows)


def build_access_events(users: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    active = users[users["employment_status"] == "Active"]["user_id"].tolist()
    dormant_uids = accounts[(accounts["status"] == "Enabled")].copy()
    dormant_uids["last_login"] = pd.to_datetime(dormant_uids["last_login"])
    dormant = dormant_uids[dormant_uids["last_login"] < pd.Timestamp(REF_DATE) - pd.Timedelta(days=90)]
    dormant_uids = dormant["user_id"].unique().tolist()

    event_types = [
        ("After-Hours Login", "Medium", 0.30),
        ("Failed Login Burst (10+)", "High", 0.22),
        ("Dormant Account Login", "High", 0.18),
        ("Privilege Escalation Attempt", "Critical", 0.12),
        ("Impossible Travel", "Critical", 0.08),
        ("Mass Export Download", "High", 0.10),
    ]
    rows = []
    for i in range(320):
        r = rng.random()
        cum, chosen = 0, event_types[0]
        for name, sev, p in event_types:
            cum += p
            if r <= cum:
                chosen = (name, sev, p)
                break
        name, sev, _ = chosen
        uid = rng.choice(dormant_uids) if name == "Dormant Account Login" and dormant_uids else rng.choice(active)
        rows.append({
            "event_id": f"EV{i+1:05d}",
            "user_id": uid,
            "system": rng.choice(SYSTEMS),
            "event_type": name,
            "severity": sev,
            "event_date": random_date(REF_DATE - timedelta(days=90), REF_DATE).isoformat(),
            "status": rng.choice(["Open", "Open", "Investigating", "Closed"],
                                 p=[0.45, 0, 0.25, 0.30]),
        })
    return pd.DataFrame(rows)


def build_certifications(users: pd.DataFrame, entitlements: pd.DataFrame) -> pd.DataFrame:
    active = users[users["employment_status"] == "Active"]
    rows = []
    for _, u in active.iterrows():
        n_high = len(entitlements[(entitlements["user_id"] == u["user_id"]) &
                                 (entitlements["risk_level"].isin(["High", "Critical"]))])
        if n_high == 0:
            continue
        # CERTIFICATION GAPS: some reviewers never finish
        r = rng.random()
        if r < 0.72:
            status, reviewed = "Complete", n_high
        elif r < 0.88:
            status = "In Progress"
            reviewed = int(rng.integers(0, n_high)) if n_high > 1 else 0
        else:
            status, reviewed = "Overdue", int(rng.integers(0, max(n_high - 1, 1)))
        rows.append({
            "campaign": "Q3-2026",
            "user_id": u["user_id"],
            "reviewer_id": u["manager_id"] if u["manager_id"] else "U0001",
            "high_risk_entitlements": n_high,
            "reviewed": int(reviewed),
            "revoked": int(rng.integers(0, 2)) if status == "Complete" else 0,
            "status": status,
            "due_date": "2026-09-30",
        })
    return pd.DataFrame(rows)


def main() -> None:
    import os
    os.makedirs("data", exist_ok=True)
    users = build_users()
    accounts = build_accounts(users)
    entitlements = build_entitlements(users)
    baselines = build_baselines()
    events = build_access_events(users, accounts)
    certs = build_certifications(users, entitlements)

    users.to_csv("data/users.csv", index=False)
    accounts.to_csv("data/accounts.csv", index=False)
    entitlements.to_csv("data/entitlements.csv", index=False)
    baselines.to_csv("data/role_baselines.csv", index=False)
    events.to_csv("data/access_events.csv", index=False)
    certs.to_csv("data/certifications.csv", index=False)

    print(f"users:         {len(users):5d}  "
          f"(terminated: {(users.employment_status == 'Terminated').sum()})")
    print(f"accounts:      {len(accounts):5d}")
    print(f"entitlements:  {len(entitlements):5d}")
    print(f"baselines:     {len(baselines):5d}")
    print(f"access_events: {len(events):5d}")
    print(f"certifications:{len(certs):5d}")


if __name__ == "__main__":
    main()

"""
IAM Access Governance Analysis
=============================
Analyzes the synthetic Lone Star Health identity dataset for common
access-governance risks:

  1. Orphaned accounts   - enabled accounts belonging to terminated users
  2. Dormant accounts     - enabled accounts with no login in 90+ days
  3. Privilege creep      - manual/exception grants beyond role baselines
  4. SoD conflicts        - toxic entitlement pairs held by one user
  5. Certification gaps   - incomplete quarterly access reviews
  6. Risk scoring         - composite per-user risk score + top offenders

Outputs: PNG charts in ./visualizations and a printed findings summary.

Run:  python3 analysis.py   (from the project root, after generate_data.py)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REF_DATE = pd.Timestamp("2026-09-25")
DORMANT_DAYS = 90
PALETTE = {"Critical": "#c0392b", "High": "#e67e22", "Medium": "#f1c40f", "Low": "#27ae60"}

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 13,
                     "axes.titleweight": "bold"})

os.makedirs("visualizations", exist_ok=True)

# ---------------------------------------------------------------- load
users = pd.read_csv("data/users.csv")
accounts = pd.read_csv("data/accounts.csv")
ents = pd.read_csv("data/entitlements.csv")
baselines = pd.read_csv("data/role_baselines.csv")
events = pd.read_csv("data/access_events.csv")
certs = pd.read_csv("data/certifications.csv")

accounts["last_login"] = pd.to_datetime(accounts["last_login"])
accounts["created_date"] = pd.to_datetime(accounts["created_date"])
users["termination_date"] = pd.to_datetime(users["termination_date"], errors="coerce")

user_dept = users.set_index("user_id")["department"].to_dict()
user_name = users.set_index("user_id")["full_name"].to_dict()
term_users = set(users.loc[users["employment_status"] == "Terminated", "user_id"])

enabled = accounts[(accounts["status"] == "Enabled") & (~accounts["is_service_account"])].copy()

# ------------------------------------------------- 1. orphaned accounts
orphaned = enabled[enabled["user_id"].isin(term_users)].copy()
orphaned["days_since_term"] = (REF_DATE - orphaned["user_id"].map(
    users.set_index("user_id")["termination_date"])).dt.days
orphaned["department"] = orphaned["user_id"].map(user_dept)

# ------------------------------------------------- 2. dormant accounts
dormant = enabled[~enabled["user_id"].isin(term_users)].copy()
dormant["days_idle"] = (REF_DATE - dormant["last_login"]).dt.days
dormant = dormant[dormant["days_idle"] >= DORMANT_DAYS].copy()
dormant["department"] = dormant["user_id"].map(user_dept)
dormant_priv = dormant[dormant["is_privileged"]]

# ------------------------------------------------- 3. privilege creep
baseline_sets = (baselines.groupby(["department", "job_title"])[["system", "entitlement"]]
                 .apply(lambda g: set(zip(g["system"], g["entitlement"]))).to_dict())
user_title = users.set_index("user_id")[["department", "job_title"]].to_dict("index")

def outside_baseline(row):
    info = user_title.get(row["user_id"])
    if not info:
        return False
    return (row["system"], row["entitlement"]) not in baseline_sets.get(
        (info["department"], info["job_title"]), set())

ents["outside_baseline"] = ents.apply(outside_baseline, axis=1)
creep = ents[(ents["outside_baseline"]) & (ents["granted_via"] != "Role")].copy()
creep["department"] = creep["user_id"].map(user_dept)
creep_critical = creep[creep["risk_level"].isin(["High", "Critical"])]

# ------------------------------------------- 4. segregation of duties
SOD_PAIRS = [
    ("Billing System", "Claims Entry", "Billing System", "Claims Approve", "Create + approve claims"),
    ("Pharmacy System", "Dispense Medication", "Pharmacy System", "Inventory Adjust", "Dispense + adjust inventory"),
    ("HR Portal", "Employee Records Edit", "HR Portal", "Payroll Approve", "Edit records + approve payroll"),
    ("HR Portal", "Payroll Process", "HR Portal", "Payroll Approve", "Process + approve payroll"),
    ("Active Directory", "Domain Admin", "Active Directory", "Security Log Read", "Admin + audit log access"),
]
held = ents.groupby("user_id")[["system", "entitlement"]].apply(
    lambda g: set(zip(g["system"], g["entitlement"])))
sod_rows = []
for uid, have in held.items():
    for sa, ea, sb, eb, label in SOD_PAIRS:
        if (sa, ea) in have and (sb, eb) in have:
            sod_rows.append({"user_id": uid, "conflict": label,
                             "department": user_dept.get(uid, "Unknown")})
sod = pd.DataFrame(sod_rows)

# ------------------------------------------- 5. certification coverage
certs["reviewer_name"] = certs["reviewer_id"].map(user_name)
certs["dept"] = certs["user_id"].map(user_dept)
cert_summary = certs["status"].value_counts()
certs["unreviewed"] = certs["high_risk_entitlements"] - certs["reviewed"]
unreviewed_high_risk = certs["unreviewed"].sum()

# ------------------------------------------------- 6. risk scoring
scores = {}
def add(uid, pts):
    scores[uid] = scores.get(uid, 0) + pts

for uid in orphaned["user_id"].unique():
    add(uid, 40)
for _, r in dormant_priv.iterrows():
    add(r["user_id"], 30)
for _, r in dormant.iterrows():
    add(r["user_id"], 10)
for _, r in creep_critical.iterrows():
    add(r["user_id"], 20)
for _, r in sod.iterrows():
    add(r["user_id"], 25)
for _, r in certs[certs["status"] == "Overdue"].iterrows():
    add(r["user_id"], 10 * r["unreviewed"])
open_crit = events[(events["severity"] == "Critical") & (events["status"] != "Closed")]
for _, r in open_crit.iterrows():
    add(r["user_id"], 15)

risk = pd.DataFrame([{"user_id": u, "risk_score": s} for u, s in scores.items()])
risk["full_name"] = risk["user_id"].map(user_name)
risk["department"] = risk["user_id"].map(user_dept)
risk["band"] = pd.cut(risk["risk_score"], bins=[-1, 19, 39, 69, 10_000],
                       labels=["Low", "Medium", "High", "Critical"])
risk = risk.sort_values("risk_score", ascending=False)

# ============================================================ charts
def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"visualizations/{name}", bbox_inches="tight")
    plt.close(fig)

# 1 - dormant accounts by department
fig, ax = plt.subplots(figsize=(9, 5))
d = dormant["department"].value_counts()
ax.bar(d.index, d.values, color="#e67e22")
ax.set_title(f"Dormant Accounts by Department (no login in {DORMANT_DAYS}+ days)")
ax.set_ylabel("Enabled dormant accounts")
plt.xticks(rotation=30, ha="right")
for i, v in enumerate(d.values):
    ax.text(i, v + 1, str(v), ha="center", fontsize=9)
save(fig, "01_dormant_by_department.png")

# 2 - orphaned accounts by system
fig, ax = plt.subplots(figsize=(9, 5))
o = orphaned["system"].value_counts()
ax.barh(o.index, o.values, color="#c0392b")
ax.set_title("Orphaned Accounts by System (terminated users, still enabled)")
ax.set_xlabel("Enabled accounts")
for i, v in enumerate(o.values):
    ax.text(v + 0.5, i, str(v), va="center", fontsize=9)
save(fig, "02_orphaned_by_system.png")

# 3 - SoD conflicts by type
fig, ax = plt.subplots(figsize=(9, 5))
if len(sod):
    s = sod["conflict"].value_counts()
    ax.barh(s.index, s.values, color="#8e44ad")
    ax.set_title("Segregation-of-Duties Conflicts by Type")
    ax.set_xlabel("Users holding both entitlements")
    for i, v in enumerate(s.values):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=9)
else:
    ax.text(0.5, 0.5, "No SoD conflicts detected", ha="center")
save(fig, "03_sod_conflicts.png")

# 4 - certification status by department (stacked)
fig, ax = plt.subplots(figsize=(10, 5.5))
piv = certs.pivot_table(index="dept", columns="status",
                        values="user_id", aggfunc="count", fill_value=0)
order = ["Complete", "In Progress", "Overdue"]
piv = piv.reindex(columns=[c for c in order if c in piv.columns], fill_value=0)
piv.plot(kind="bar", stacked=True, ax=ax,
         color=["#27ae60", "#f1c40f", "#c0392b"])
ax.set_title("Q3-2026 Access Certification Status by Department")
ax.set_ylabel("Users in campaign")
ax.set_xlabel("")
plt.xticks(rotation=30, ha="right")
ax.legend(title="Status")
save(fig, "04_certification_status.png")

# 5 - risk score distribution
fig, ax = plt.subplots(figsize=(9, 5))
bands = risk["band"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
ax.bar(bands.index, bands.values,
       color=[PALETTE[b] for b in bands.index])
ax.set_title("User Risk Score Distribution")
ax.set_ylabel("Users")
for i, v in enumerate(bands.values):
    ax.text(i, v + 0.3, str(v), ha="center", fontsize=10, fontweight="bold")
save(fig, "05_risk_distribution.png")

# 6 - anomalous events by type and severity
fig, ax = plt.subplots(figsize=(10, 5.5))
ev = events.pivot_table(index="event_type", columns="severity",
                        values="event_id", aggfunc="count", fill_value=0)
ev = ev.reindex(columns=["Medium", "High", "Critical"], fill_value=0)
ev.plot(kind="bar", stacked=True, ax=ax,
        color=[PALETTE["Medium"], PALETTE["High"], PALETTE["Critical"]])
ax.set_title("Anomalous Access Events (last 90 days) by Type and Severity")
ax.set_ylabel("Events")
ax.set_xlabel("")
plt.xticks(rotation=25, ha="right")
ax.legend(title="Severity")
save(fig, "06_access_events.png")

# ============================================================ summary
print("=" * 64)
print("IAM ACCESS GOVERNANCE - FINDINGS SUMMARY (synthetic data)")
print("=" * 64)
print(f"\nScope: {len(users)} users, {len(accounts)} accounts, "
      f"{len(ents)} entitlements across {accounts['system'].nunique()} systems")
print(f"\n[1] ORPHANED ACCOUNTS: {len(orphaned)} enabled accounts belong to "
      f"{orphaned['user_id'].nunique()} terminated users "
      f"({len(orphaned)/len(enabled):.1%} of enabled accounts).")
print(f"    Longest-enabled orphan: {int(orphaned['days_since_term'].max())} days "
      f"past termination. Top system: {orphaned['system'].value_counts().index[0]} "
      f"({orphaned['system'].value_counts().iloc[0]}).")
print(f"\n[2] DORMANT ACCOUNTS: {len(dormant)} enabled accounts idle {DORMANT_DAYS}+ days, "
      f"including {len(dormant_priv)} PRIVILEGED accounts.")
print(f"    Worst department: {dormant['department'].value_counts().index[0]} "
      f"({dormant['department'].value_counts().iloc[0]}).")
print(f"\n[3] PRIVILEGE CREEP: {len(creep)} manual/exception grants sit outside role "
      f"baselines; {len(creep_critical)} are High/Critical risk "
      f"({creep_critical['user_id'].nunique()} users).")
print(f"\n[4] SoD CONFLICTS: {len(sod)} toxic pairs across {sod['user_id'].nunique()} users."
      if len(sod) else "\n[4] SoD CONFLICTS: none detected.")
if len(sod):
    print(f"    Most common: {sod['conflict'].value_counts().index[0]} "
          f"({sod['conflict'].value_counts().iloc[0]} users).")
print(f"\n[5] CERTIFICATION: {cert_summary.get('Complete', 0)}/{len(certs)} users fully "
      f"reviewed ({cert_summary.get('Complete', 0)/len(certs):.0%}); "
      f"{unreviewed_high_risk} high-risk entitlements remain unreviewed, "
      f"{(certs['status'] == 'Overdue').sum()} users overdue.")
print(f"\n[6] RISK SCORING: {len(risk)} users carry measurable risk - "
      f"{(risk['band'] == 'Critical').sum()} Critical, {(risk['band'] == 'High').sum()} High.")
print("\nTop 10 riskiest users:")
for _, r in risk.head(10).iterrows():
    print(f"    {r['risk_score']:3d}  {r['band']:8s}  {r['full_name']} "
          f"({r['user_id']}, {r['department']})")
print(f"\n[7] ACCESS EVENTS: {len(events)} anomalies in 90 days; "
      f"{(events['status'] != 'Closed').sum()} still open/investigating, "
      f"including {(open_crit['severity'] == 'Critical').sum()} critical-severity.")
print("\nCharts saved to ./visualizations/")

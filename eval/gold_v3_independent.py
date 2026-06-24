"""
A THIRD grader that deliberately does NOT share the build-evidence regex with
either the ranker's s_evidence or gold_v2. Instead it grades off:
  - the candidate's own HONEST self-disclosure language (hedging => shallow)
  - whether the RECENT role title (not current title) is retrieval/ranking
  - YoE band + seniority + non-tech/consulting caps
  - independent honeypot (summary/profile YoE contradiction)
This lets us see if the s_evidence gain survives a grader that uses different features.
"""
import re
def low(s): return s.lower() if isinstance(s,str) else ""
HEDGE=["lighter on","lighter than","haven't done","not in a professional","self-taught",
 "self-learner","side project","online course","experimented with","curious about",
 "building competence","strongest at the modeling","classical methods","split between dashboard",
 "mostly classical","kaggle","hobby","exploring","at a self-learner level"]
STRONG_SELF=["production","at scale","serving","50m","queries per month","led the team",
 "owned the","rebuilt","migrated our","end-to-end","shipped to","real users"]
NONTECH={"hr manager","marketing manager","sales executive","accountant","business analyst",
 "operations manager","customer support","content writer","graphic designer","project manager",
 "civil engineer","mechanical engineer","recruiter","office manager"}
CONSULT={"tcs","infosys","wipro","accenture","cognizant","capgemini","mindtree","mphasis","hcl"}
RETR_ROLE=["search","ranking","retrieval","recommendation","relevance","nlp","ml","machine learning","ai engineer"]
TARGET={"pune","noida","hyderabad","mumbai","delhi","bangalore","bengaluru","gurgaon","gurugram"}

def grade(c):
    p=c.get("profile",{}); yoe=float(p.get("years_of_experience",0) or 0)
    summary=low(p.get("summary","")); title=low(p.get("current_title",""))
    ch=c.get("career_history",[])
    # independent honeypot
    m=re.search(r"(\d+\.?\d*)\s*\+?\s*years? of experience",summary)
    if m and abs(float(m.group(1))-yoe)>2.0: return 0
    for r in ch:
        if int(r.get("duration_months",0) or 0) > yoe*12+18: return 0
    recent_title=low(ch[0].get("title","")) if ch else ""
    hedge=sum(1 for h in HEDGE if h in summary)
    strong=sum(1 for s in STRONG_SELF if s in (summary+" "+" ".join(low(r.get("description","")) for r in ch)))
    retr_role=any(t in recent_title or t in title for t in RETR_ROLE)
    # caps
    cap=5
    if title in NONTECH: cap=min(cap,1)
    comps=[low(r.get("company","")) for r in ch]
    if comps and all(any(cc in x for cc in CONSULT) for x in comps): cap=min(cap,2)
    if yoe<3: cap=min(cap,2)
    # base
    if strong>=2 and retr_role: base=5
    elif strong>=1 and retr_role: base=4
    elif retr_role: base=3
    elif strong>=1: base=3
    else: base=2
    if hedge>=2: base=min(base,2)
    elif hedge>=1: base=min(base,3)
    tier=min(base,cap)
    senior=any(k in title for k in ("senior","lead","principal","staff"))
    if tier>=4 and not (5<=yoe<=9 or senior): tier-=1
    loc_ok=any(t in low(p.get("location","")) for t in TARGET) or bool(c.get("redrob_signals",{}).get("willing_to_relocate",False))
    if tier>=3 and not loc_ok and low(p.get("country",""))!="india": tier-=1
    return max(0,min(5,tier))

from datetime import date
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")
model = joblib.load("churn_model.pkl")


HTML_PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Churn Predictor</title>
<style>
  :root{
    --bg:#f5f6fa; --card:#fff; --text:#1c2033; --muted:#6b7190; --line:#e3e6f0;
    --brand:#4f46e5; --brand-d:#4338ca; --good:#16a34a; --bad:#dc2626; --warn:#d97706;
  }
  @media (prefers-color-scheme:dark){
    :root{--bg:#0f1220;--card:#181c30;--text:#e8eaf6;--muted:#9aa0c3;--line:#2a2f4a;--brand:#818cf8;--brand-d:#6366f1}
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
  header{padding:28px 16px 8px;max-width:1100px;margin:auto}
  header h1{margin:0;font-size:26px}
  header p{margin:4px 0 0;color:var(--muted)}
  .wrap{max-width:1100px;margin:auto;padding:16px;display:grid;grid-template-columns:1fr 340px;gap:20px;align-items:start}
  @media(max-width:900px){.wrap{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
  fieldset{border:0;padding:0;margin:0 0 6px}
  details{border-top:1px solid var(--line);padding:12px 0}
  details:first-of-type{border-top:0;padding-top:0}
  summary{cursor:pointer;font-weight:600;list-style:none;display:flex;justify-content:space-between}
  summary::after{content:"+";color:var(--muted)}
  details[open] summary::after{content:"–"}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-top:12px}
  label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:4px}
  input,select{font:inherit;color:var(--text);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;width:100%}
  input:focus,select:focus{outline:2px solid var(--brand);outline-offset:-1px}
  .actions{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}
  button{font:inherit;font-weight:600;border:0;border-radius:10px;padding:11px 20px;cursor:pointer}
  .primary{background:var(--brand);color:#fff}.primary:hover{background:var(--brand-d)}
  .primary:disabled{opacity:.6;cursor:wait}
  .ghost{background:transparent;color:var(--text);border:1px solid var(--line)}
  .result{position:sticky;top:16px;text-align:center}
  .gauge{--p:0;width:190px;height:190px;border-radius:50%;margin:10px auto 14px;
    background:conic-gradient(var(--c,var(--line)) calc(var(--p)*1%),var(--line) 0);
    display:grid;place-items:center;transition:.4s}
  .gauge div{width:146px;height:146px;border-radius:50%;background:var(--card);display:grid;place-items:center;font-size:34px;font-weight:700}
  .label{font-size:20px;font-weight:700}
  .hint{color:var(--muted);font-size:13px;margin-top:6px}
  .err{color:var(--bad);font-size:13px;margin-top:10px;white-space:pre-wrap}
  .badge{display:inline-block;padding:3px 10px;border-radius:99px;font-size:12px;font-weight:600;margin-top:8px;color:#fff;background:var(--muted)}
</style>
</head>
<body>
<header>
  <h1>Customer Churn Predictor</h1>
  <p>Fill in the customer details and get the probability that they will leave.</p>
</header>

<div class="wrap">
  <form id="form" class="card"></form>

  <aside class="card result">
    <div class="hint">Churn probability</div>
    <div class="gauge" id="gauge"><div id="pct">–</div></div>
    <div class="label" id="label">Waiting for input</div>
    <span class="badge" id="badge" hidden></span>
    <div class="hint" id="note">Click “Predict churn” to run the model.</div>
    <div class="err" id="err"></div>
  </aside>
</div>

<script>
const OPT = {
  gender:["Female","Male","Other"], marital_status:["Divorced","Married","Single","Widowed"],
  education:["Bachelors","Diploma","High School","Masters","PhD"],
  occupation:["Business","Clerk","Doctor","Engineer","Freelancer","Retired","Student","Teacher"],
  state:["DL","GJ","KA","MH","MP","RJ","TN","TS","UP","WB"], city_tier:["Tier1","Tier2","Tier3"],
  plan_type:["Free","Basic","Standard","Premium"], payment_method:["COD","Card","NetBanking","UPI","Wallet"],
  device_type:["Desktop","Mobile","Tablet"], os:["Android","Linux","Windows","iOS","macOS"],
  browser:["Chrome","Edge","Firefox","Other","Safari"], referral_source:["Ads","Email","Organic","Referral","Social"]
};
// [name, label, type, default]  type: n=number, s=select, d=date, y=yes/no
const SECTIONS = [
 ["Key behaviour (most important)", true, [
  ["days_since_last_purchase","Days since last purchase","n",200],
  ["csat_score","Satisfaction (CSAT 1-5)","n",3],
  ["support_tickets","Support tickets","n",1],
  ["sessions_last_30d","Sessions (last 30 days)","n",12],
  ["tenure_months","Tenure (months)","n",48],
  ["plan_type","Plan","s","Basic"],
 ]],
 ["Profile", false, [
  ["age","Age","n",35],["gender","Gender","s","Male"],["marital_status","Marital status","s","Single"],
  ["education","Education","s","Bachelors"],["occupation","Occupation","s","Engineer"],["state","State","s","MP"],
  ["city_tier","City tier","s","Tier2"],["annual_income","Annual income","n",400000],["credit_score","Credit score","n",680],
  ["household_size","Household size","n",4],["num_children","Children","n",1],["years_employed","Years employed","n",10],
  ["owns_home","Owns home","y",1],["owns_car","Owns car","y",0],
 ]],
 ["Account & device", false, [
  ["signup_date","Signup date","d","2022-01-01"],["last_login_date","Last login date","d","2025-11-01"],
  ["payment_method","Payment method","s","UPI"],["device_type","Device","s","Mobile"],["os","OS","s","Android"],
  ["browser","Browser","s","Chrome"],["referral_source","Referral source","s","Organic"],
  ["is_subscribed_newsletter","Newsletter","y",1],["email_verified","Email verified","y",1],
  ["phone_verified","Phone verified","y",1],["notification_opt_in","Notifications","y",1],
 ]],
 ["Orders & engagement", false, [
  ["total_orders","Total orders","n",14],["avg_order_value","Avg order value","n",800],["returns_count","Returns","n",1],
  ["cart_abandon_rate","Cart abandon rate (0-1)","n",0.3],["avg_session_minutes","Avg session minutes","n",10],
  ["pages_per_session","Pages per session","n",5.4],["discount_usage_pct","Discount usage %","n",30],
  ["loyalty_points","Loyalty points","n",10000],["wishlist_items","Wishlist items","n",3],
  ["reviews_written","Reviews written","n",1],["social_connections","Social connections","n",500],
  ["avg_resolution_hours","Avg resolution hours","n",17],["nps_score","NPS (0-10)","n",5],["app_rating_given","App rating (1-5)","n",4],
 ]],
];
const OPTIONAL = new Set(["education","annual_income","credit_score","avg_session_minutes","nps_score","app_rating_given"]);
const form = document.getElementById("form");

function field([name,label,type,def]){
  let el;
  if(type==="s") el=`<select name="${name}">${(OPT[name]||[]).map(o=>`<option ${o===def?"selected":""}>${o}</option>`).join("")}</select>`;
  else if(type==="y") el=`<select name="${name}"><option value="1" ${def?"selected":""}>Yes</option><option value="0" ${def?"":"selected"}>No</option></select>`;
  else if(type==="d") el=`<input type="date" name="${name}" value="${def}" required>`;
  else el=`<input type="number" step="any" name="${name}" value="${def}" ${OPTIONAL.has(name)?"":"required"}`;
  return `<label>${label}${el}</label>`;
}
let html = SECTIONS.map(([t,open,f])=>`<details ${open?"open":""}><summary>${t}</summary><div class="grid">${f.map(field).join("")}</div></details>`).join("");
const sp=[620,640,660,680,700,690,670,650,640,630,620,610];
html += `<details><summary>Monthly spend (last 12 months, oldest → latest)</summary><div class="grid">`+
  sp.map((v,i)=>`<label>Month ${i+1}<input type="number" step="any" name="m${i+1}" value="${v}" required></label>`).join("")+`</div></details>
  <div class="actions"><button class="primary" id="go" type="submit">Predict churn</button>
  <button class="ghost" id="reset" type="button">Reset</button></div>`;
form.innerHTML = html;

const INTS = new Set(["age","household_size","num_children","years_employed","tenure_months","total_orders","returns_count",
 "sessions_last_30d","support_tickets","csat_score","loyalty_points","days_since_last_purchase","wishlist_items","reviews_written",
 "social_connections","owns_home","owns_car","is_subscribed_newsletter","email_verified","phone_verified","notification_opt_in"]);

form.addEventListener("submit", async e=>{
  e.preventDefault();
  const fd = new FormData(form), body = {monthly_spend:[]};
  for(const [k,v] of fd.entries()){
    if(/^m\d+$/.test(k)){ body.monthly_spend[+k.slice(1)-1] = +v; continue; }
    const t = form.elements[k];
    if(t.type==="number"){ body[k] = v==="" ? null : (INTS.has(k)? Math.round(+v) : +v); }
    else if(t.tagName==="SELECT" && /^[01]$/.test(v) && INTS.has(k)) body[k]=+v;
    else body[k]=v;
  }
  const btn=document.getElementById("go"), err=document.getElementById("err");
  btn.disabled=true; btn.textContent="Predicting…"; err.textContent="";
  try{
    const r = await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    if(!r.ok) throw new Error(JSON.stringify(await r.json(),null,2));
    show(await r.json());
  }catch(x){ err.textContent = "Error: "+x.message; }
  btn.disabled=false; btn.textContent="Predict churn";
});

function show(d){
  const p=Math.round(d.churn_probability*100);
  const level = p>=70?["High risk","var(--bad)"]:p>=40?["Medium risk","var(--warn)"]:["Low risk","var(--good)"];
  const g=document.getElementById("gauge");
  g.style.setProperty("--p",p); g.style.setProperty("--c",level[1]);
  document.getElementById("pct").textContent=p+"%";
  document.getElementById("label").textContent=d.label;
  const b=document.getElementById("badge"); b.hidden=false; b.textContent=level[0]; b.style.background=level[1];
  document.getElementById("note").textContent = d.prediction
    ? "This customer is likely to leave. Consider a retention offer or outreach." : "This customer is likely to stay.";
}
document.getElementById("reset").onclick=()=>{form.reset();};
</script>
</body>
</html>
'''


class Customer(BaseModel):
    age: int
    gender: str
    marital_status: str
    education: Optional[str] = None
    occupation: str
    state: str
    city_tier: str
    annual_income: Optional[float] = None
    credit_score: Optional[float] = None
    household_size: int
    num_children: int
    owns_home: int
    owns_car: int
    years_employed: int
    signup_date: date
    last_login_date: date
    tenure_months: int
    plan_type: str
    payment_method: str
    device_type: str
    os: str
    browser: str
    referral_source: str
    is_subscribed_newsletter: int
    email_verified: int
    phone_verified: int
    monthly_spend: list[float]
    total_orders: int
    avg_order_value: float
    returns_count: int
    cart_abandon_rate: float
    sessions_last_30d: int
    avg_session_minutes: Optional[float] = None
    pages_per_session: float
    support_tickets: int
    avg_resolution_hours: float
    csat_score: int
    nps_score: Optional[float] = None
    discount_usage_pct: float
    loyalty_points: int
    app_rating_given: Optional[float] = None
    days_since_last_purchase: int
    wishlist_items: int
    reviews_written: int
    social_connections: int
    notification_opt_in: int


def build_features(c: Customer) -> pd.DataFrame:
    d = c.model_dump()
    spend = np.array(d.pop("monthly_spend"), dtype=float)
    if len(spend) != 12:
        raise ValueError("monthly_spend must have exactly 12 values")
    signup, login = d.pop("signup_date"), d.pop("last_login_date")

    early, recent = spend[:3].mean(), spend[-3:].mean()
    d["spend_mean"] = spend.mean()
    d["spend_std"] = spend.std(ddof=1)
    d["spend_trend"] = recent - early
    d["spend_recent_ratio"] = recent / (early + 1)
    sess_min = d["avg_session_minutes"]
    d["engagement"] = d["sessions_last_30d"] * sess_min if sess_min is not None else np.nan
    d["tickets_per_month"] = d["support_tickets"] / (d["tenure_months"] + 1)
    d["orders_per_month"] = d["total_orders"] / (d["tenure_months"] + 1)
    d["is_inactive"] = int(d["days_since_last_purchase"] > 90)
    d["signup_year"] = signup.year
    d["login_span_days"] = (login - signup).days

    df = pd.DataFrame([d])
    cols = model.named_steps["preprocess"].feature_names_in_
    return df[cols]


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.post("/predict")
def predict(customer: Customer):
    X = build_features(customer)
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "churn_probability": round(proba, 4),
        "prediction": int(proba >= 0.5),
        "label": "Will churn" if proba >= 0.5 else "Will stay",
    }
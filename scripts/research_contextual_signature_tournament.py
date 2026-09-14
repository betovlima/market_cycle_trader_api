from __future__ import annotations
import argparse, math
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import pandas as pd
from pymongo import MongoClient
import research_contextual_signature_protocol as protocol
import research_contextual_signature_storage as storage

TARGET="source_direct_delta_log_capital"; PRIMARY="Original25"; ROBUST="Original24_MinusADM"; HOLDOUT=2026; MIN_STATES=8; TOL=1e-12
EXPERIMENT_NAME="contextual_marginal_signature_direct_marginal_tournament"; COLLECTION="research_contextual_signature_tournaments"

@dataclass(frozen=True)
class ModelSpec: name:str; factory:Callable[[],Any]

def _parser():
    p=argparse.ArgumentParser(); p.add_argument("--strategy-sequence",type=int,default=10); p.add_argument("--strategy-id"); p.add_argument("--env-file"); p.add_argument("--mongo-uri"); p.add_argument("--database"); return p

def _models():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet,Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    def pipe(m): return Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("m",m)])
    out=[ModelSpec("ridge",lambda:pipe(Ridge(alpha=10))),ModelSpec("elastic_net",lambda:pipe(ElasticNet(alpha=.01,l1_ratio=.25,max_iter=10000,random_state=42))),ModelSpec("mlp",lambda:pipe(MLPRegressor(hidden_layer_sizes=(16,),solver="lbfgs",alpha=.01,max_iter=2000,random_state=42)))]
    try:
        from lightgbm import LGBMRegressor
        out.insert(2,ModelSpec("lightgbm",lambda:LGBMRegressor(n_estimators=120,learning_rate=.03,num_leaves=7,min_child_samples=10,reg_lambda=1,random_state=42,n_jobs=1,verbosity=-1,deterministic=True,force_col_wise=True)))
    except Exception: pass
    return out

def _num(v):
    try: x=float(v)
    except (TypeError,ValueError): return np.nan
    return x if math.isfinite(x) else np.nan

def _at(frame,date):
    idx=pd.DatetimeIndex(pd.to_datetime(frame.index,utc=True)); d=pd.Timestamp(date); d=d.tz_localize("UTC") if d.tzinfo is None else d.tz_convert("UTC"); pos=int(idx.searchsorted(d,side="right")-1)
    if pos<0: raise RuntimeError(f"No feature row before {d.date()}")
    return frame.iloc[pos]

def build_matrix(obs,frames,features):
    universes={x["name"]:list(x["assets"]) for x in protocol.UNIVERSES}; rows=[]
    for o in obs.to_dict("records"):
        c=str(o["candidate"]); u=str(o["universe_name"]); d=pd.Timestamp(o["decision_date"]); cr=_at(frames[c],d); refs=universes[u]; rr=pd.DataFrame([_at(frames[s],d) for s in refs],index=refs); r={"run_id":o.get("run_id"),"decision_date":d.date().isoformat(),"universe_name":u,"candidate":c,TARGET:float(o[TARGET])}
        for f in features:
            cv=_num(cr.get(f)); vals=pd.to_numeric(rr.get(f),errors="coerce").replace([np.inf,-np.inf],np.nan).dropna(); mean=float(vals.mean()) if len(vals) else np.nan; std=float(vals.std(ddof=0)) if len(vals) else np.nan
            r[f"candidate__{f}"]=cv; r[f"relative__{f}"]=cv-mean; r[f"z__{f}"]=(cv-mean)/std if std>1e-12 else 0.0; r[f"rankpct__{f}"]=float((vals<cv).mean()) if len(vals) else np.nan; r[f"universe_mean__{f}"]=mean; r[f"universe_std__{f}"]=std
            if "SPY" in frames: r[f"market_spy__{f}"]=_num(_at(frames["SPY"],d).get(f))
        rows.append(r)
    return pd.DataFrame(rows)

def feature_columns(df):
    skip={"run_id","decision_date","universe_name","candidate",TARGET}; return [c for c in df if c not in skip and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]

def _pick(g):
    b=g.sort_values("prediction",ascending=False).iloc[0]; p=float(b.prediction); oracle=max(0,float(g[TARGET].max())); return (None,0.0,oracle) if p<=0 else (str(b.candidate),float(b[TARGET]),oracle)

def _metrics(rows):
    if not rows:return {"contexts":0,"cumulative_direct_log_gain":0.0,"capital_multiplier_vs_no_insert":1.0,"mean_context_spearman":None}
    d=pd.DataFrame(rows); gain=float(d.realized.sum()); rho=pd.to_numeric(d.rho,errors="coerce").dropna(); return {"contexts":len(d),"interventions":int(d.selected.notna().sum()),"positive_interventions":int(((d.realized>TOL)&d.selected.notna()).sum()),"cumulative_direct_log_gain":gain,"capital_multiplier_vs_no_insert":float(math.exp(gain)),"oracle_log_gain":float(d.oracle.sum()),"mean_context_spearman":float(rho.mean()) if len(rho) else None}

def walk_forward(df,cols,spec):
    d=df[df.universe_name==PRIMARY].copy(); d.decision_date=pd.to_datetime(d.decision_date); dates=sorted(x for x in d.decision_date.unique() if pd.Timestamp(x).year<HOLDOUT); rows=[]
    for i in range(MIN_STATES,len(dates)):
        tr=d[d.decision_date.isin(dates[:i])]; te=d[d.decision_date==dates[i]].copy(); m=spec.factory(); m.fit(tr[cols],tr[TARGET]); te["prediction"]=m.predict(te[cols]); s,r,o=_pick(te); rho=te.prediction.rank().corr(te[TARGET].rank()); rows.append({"date":str(pd.Timestamp(dates[i]).date()),"selected":s,"realized":r,"oracle":o,"rho":rho})
    return _metrics(rows)

def mean_baseline(df):
    d=df[df.universe_name==PRIMARY].copy(); d.decision_date=pd.to_datetime(d.decision_date); dates=sorted(x for x in d.decision_date.unique() if pd.Timestamp(x).year<HOLDOUT); rows=[]
    for i in range(MIN_STATES,len(dates)):
        tr=d[d.decision_date.isin(dates[:i])]; te=d[d.decision_date==dates[i]].copy(); means=tr.groupby("candidate")[TARGET].mean(); te["prediction"]=te.candidate.map(means).fillna(0); s,r,o=_pick(te); rho=te.prediction.rank().corr(te[TARGET].rank()); rows.append({"selected":s,"realized":r,"oracle":o,"rho":rho})
    return _metrics(rows)

def holdout(df,cols,spec,universe):
    d=df.copy(); d.decision_date=pd.to_datetime(d.decision_date); tr=d[(d.universe_name==PRIMARY)&(d.decision_date.dt.year<HOLDOUT)]; te=d[(d.universe_name==universe)&(d.decision_date.dt.year>=HOLDOUT)].copy(); rows=[]
    if te.empty:return _metrics(rows)
    m=spec.factory(); m.fit(tr[cols],tr[TARGET]); te["prediction"]=m.predict(te[cols])
    for _,g in te.groupby("decision_date"): s,r,o=_pick(g); rho=g.prediction.rank().corr(g[TARGET].rank()); rows.append({"selected":s,"realized":r,"oracle":o,"rho":rho})
    return _metrics(rows)

def status(dev,base,hold,robust):
    dg=dev["cumulative_direct_log_gain"]; bg=base["cumulative_direct_log_gain"]
    if dg<=TOL or dg<=bg+TOL:return "NO_CONTEXTUAL_SIGNAL"
    if hold["cumulative_direct_log_gain"]<=TOL:return "DEVELOPMENT_SIGNAL_NOT_CONFIRMED"
    if robust["cumulative_direct_log_gain"]<=TOL:return "HOLDOUT_SIGNAL_NOT_ROBUST"
    return "CONFIRMED_LIMITED_NEXT_SYSTEM_BACKTEST"

def main(*,script_version):
    protocol._install_console_logging(); a=_parser().parse_args(); protocol.load_project_environment(a.env_file)
    from market_cycle_trader_api.engine import capital_rotation,market_data
    from market_cycle_trader_api.infrastructure.persistence import mongo_repository
    from market_cycle_trader_api.schemas.requests import BacktestRequest
    uri,dbn=storage.runtime_mongo_settings(a,mongo_repository,protocol); client=MongoClient(uri,serverSelectionTimeoutMS=3000,connectTimeoutMS=3000,retryWrites=False); storage.mongo_retry(lambda:client.admin.command("ping")); db=client[dbn]; st=protocol._strategy_document(db,a.strategy_sequence,a.strategy_id); sid=str(st["_id"]); cfg=BacktestRequest.model_validate(protocol._configuration(st)).model_copy(update={"end_date":protocol.SNAPSHOT_END,"research_market_data_mode":"database_only","mongo_cache_enabled":True,"market_data_require_complete_history":True})
    expected=len(protocol.DECISION_DATES)*len(protocol.UNIVERSES)*len(protocol.CANDIDATES); run=db[protocol.RUNS_COLLECTION].find_one({"experiment":protocol.EXPERIMENT_NAME,"strategy_id":sid,"status":"completed","completed_observations":{"$gte":expected}},sort=[("completed_utc",-1),("updated_utc",-1)])
    if not run: raise RuntimeError("No completed 23x2x7 Mongo campaign found.")
    rid=str(run["_id"]); obs=pd.DataFrame(list(db[protocol.OBSERVATIONS_COLLECTION].find({"run_id":rid},{"_id":0})))
    if len(obs)!=expected or TARGET not in obs: raise RuntimeError(f"Expected {expected} observations with {TARGET}.")
    bars,_=protocol._load_market_frames(cfg,market_data); ff={s:capital_rotation.build_rotation_frame(b,cfg) for s,b in bars.items()}; matrix=build_matrix(obs,ff,list(capital_rotation.ROTATION_FEATURES)); cols=feature_columns(matrix); protocol.live.console_log(f"[tournament] rows={len(matrix)} | features={len(cols)} | target={TARGET} | candidate identity excluded")
    base=mean_baseline(matrix); specs=_models(); dev={s.name:walk_forward(matrix,cols,s) for s in specs}; winner=max(specs,key=lambda s:dev[s.name]["cumulative_direct_log_gain"]); hold=holdout(matrix,cols,winner,PRIMARY); robust=holdout(matrix,cols,winner,ROBUST); result=status(dev[winner.name],base,hold,robust)
    for name,m in dev.items(): protocol.live.console_log(f"[development] {name}: {m['capital_multiplier_vs_no_insert']:.4f}x")
    protocol.live.console_log(f"[winner] {winner.name} | dev={dev[winner.name]['capital_multiplier_vs_no_insert']:.4f}x | 2026={hold['capital_multiplier_vs_no_insert']:.4f}x | robust={robust['capital_multiplier_vs_no_insert']:.4f}x | {result}")
    tid=storage.canonical_hash({"source":rid,"version":script_version,"features":cols}); summary={"_id":tid,"schema_version":1,"script_version":script_version,"experiment":EXPERIMENT_NAME,"status":"completed","source_run_id":rid,"strategy_id":sid,"target":TARGET,"target_definition":"log(W_policy_with_candidate_added / W_baseline_policy_without_candidate)","rows":len(matrix),"independent_primary_temporal_states":int(matrix[matrix.universe_name==PRIMARY].decision_date.nunique()),"feature_count":len(cols),"candidate_identity_feature":False,"validation":"Original25 expanding walk-forward through 2025; 2026 untouched holdout; MinusADM robustness only","historical_candidate_mean_baseline":base,"development_models":dev,"development_winner":winner.name,"final_holdout_2026":hold,"robustness_holdout_2026":robust,"decision_status":result,"next_step":"Only CONFIRMED_LIMITED_NEXT_SYSTEM_BACKTEST proceeds to one frozen system-level capital backtest.","created_utc":pd.Timestamp.now(tz="UTC").to_pydatetime()}; storage.mongo_retry(lambda:db[COLLECTION].replace_one({"_id":tid},storage.mongo_value(summary),upsert=True)); export={k:v for k,v in summary.items() if k!="_id"}; storage.export_result(project_root=protocol.PROJECT_ROOT,run_id=tid,script_version=script_version,dataset=matrix,summary=export,export_folder_name=protocol.EXPORT_FOLDER_NAME,export_zip_name=protocol.EXPORT_ZIP_NAME,collections=(protocol.RUNS_COLLECTION,protocol.OBSERVATIONS_COLLECTION,COLLECTION)); return 0

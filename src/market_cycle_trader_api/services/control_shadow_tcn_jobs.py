"""Admin-only frozen-snapshot TinyTCN research jobs. No Alpaca or trading."""
from __future__ import annotations

import logging
import re
import threading
import uuid
from typing import Any

from pymongo.errors import DuplicateKeyError

from ..engine.control_deep_learning_research import run_control_tcn_research
from ..infrastructure.persistence.mongo_repository import utc_now
from .control_shadow_jobs import _require_enabled
from .control_shadow_validation_jobs import COLLECTION as VALIDATION_COLLECTION
from .control_shadow_execution_jobs import COLLECTION as EXECUTION_COLLECTION
from .control_shadow_liquidity_jobs import COLLECTION as LIQUIDITY_COLLECTION

LOGGER=logging.getLogger("uvicorn.error")
COLLECTION="control_shadow_tcn_jobs"
ACTIVE_KEY="control-tcn-fixed-exploratory-v1045"
_THREADS:dict[str,threading.Thread]={}


class TCNInvalid(ValueError):
    pass


class TCNNotFound(LookupError):
    pass


class TCNConflict(RuntimeError):
    pass


def _public(record:dict[str,Any],*,logs_only:bool=False)->dict[str,Any]:
    result={
        "job_id":record["_id"],"status":record["status"],
        "stage":record.get("stage"),"progress":record.get("progress",0),
        "logs":list(record.get("logs") or []),
        "source_job_id":record["source_job_id"],
        "source_validation_job_id":record["source_validation_job_id"],
        "source_execution_job_id":record["source_execution_job_id"],
        "source_liquidity_job_id":record["source_liquidity_job_id"],
        "snapshot_sha256":record["snapshot_sha256"],
        "source_download":"never","order_eligible":False,"order_submission":"never",
    }
    if not logs_only:
        result.update({
            "created_at":record.get("created_at"),
            "started_at":record.get("started_at"),
            "finished_at":record.get("finished_at"),
            "result":record.get("result"),"error":record.get("error"),
        })
    return result


def _log(db:Any,job_id:str,message:str,*,level:str="INFO",
         stage:str|None=None,progress:int|None=None)->None:
    now=utc_now()
    update={"updated_at":now}
    if stage is not None: update["stage"]=stage
    if progress is not None: update["progress"]=progress
    db[COLLECTION].update_one(
        {"_id":job_id},
        {"$set":update,"$push":{"logs":{"$each":[{
            "at":now,"level":level,"message":str(message)[:400]
        }],"$slice":-300}}},
    )
    getattr(LOGGER,level.lower(),LOGGER.info)("Control TCN %s | %s",job_id,message)


def _run_job(db:Any,job_id:str,*,source_job_id:str,
             validation_id:str,execution_id:str,liquidity_id:str,
             expected_sha256:str,baseline41:dict,baseline42:dict,baseline44:dict)->None:
    try:
        now=utc_now()
        db[COLLECTION].update_one(
            {"_id":job_id,"status":"queued"},
            {"$set":{"status":"running","started_at":now,"updated_at":now}},
        )
        _log(db,job_id,
             "SHA verification and fixed CPU TCN chronological training; no Alpaca or orders.",
             stage="verify_immutable_snapshot",progress=1)
        last=[-10]
        def progress(stage:str,done:int,total:int)->None:
            pct=3 if stage=="verify_immutable_snapshot" else (
                99 if stage=="completed" else
                min(99,5+int(94*done/max(1,total)))
            )
            if pct-last[0]>=4 or stage=="completed":
                last[0]=pct
                _log(db,job_id,f"{stage}: {done}/{total}",
                     stage=str(stage)[:100],progress=pct)
        report=run_control_tcn_research(
            source_job_id=source_job_id,
            validation_job_id=validation_id,
            execution_job_id=execution_id,
            liquidity_job_id=liquidity_id,
            research_job_id=job_id,
            expected_sha256=expected_sha256,
            baseline41=baseline41,baseline42=baseline42,baseline44=baseline44,
            progress=progress,
        )
        now=utc_now()
        _log(db,job_id,
             "Fixed TCN research completed; original sources intact, no orders.",
             stage="completed",progress=100)
        db[COLLECTION].update_one(
            {"_id":job_id},
            {"$set":{"status":"completed","stage":"completed",
                     "progress":100,"result":report,
                     "updated_at":now,"finished_at":now},
             "$unset":{"active_key":""}},
        )
    except Exception as exc:
        LOGGER.exception("Control TCN %s | failed, no trade submitted",job_id)
        _log(db,job_id,f"{type(exc).__name__}: {str(exc)[:330]}",
             level="ERROR",stage="failed")
        now=utc_now()
        db[COLLECTION].update_one({"_id":job_id},{
            "$set":{"status":"failed","stage":"failed",
                    "error":f"{type(exc).__name__}: {str(exc)[:500]}",
                    "updated_at":now,"finished_at":now},
            "$unset":{"active_key":""},
        })
    finally:
        _THREADS.pop(job_id,None)


def start_tcn_research(db:Any,*,source_validation_job_id:str,
                       source_execution_job_id:str,
                       source_liquidity_job_id:str,
                       expected_snapshot_sha256:str)->dict[str,Any]:
    _require_enabled()
    requested=(
        (source_validation_job_id,r"control-validation-[a-f0-9]{16}"),
        (source_execution_job_id,r"control-execution-[a-f0-9]{16}"),
        (source_liquidity_job_id,r"control-liquidity-[a-f0-9]{16}"),
        (expected_snapshot_sha256,r"[a-f0-9]{64}"),
    )
    if any(not re.fullmatch(pattern,value) for value,pattern in requested):
        raise TCNInvalid("Exact immutable source job IDs and snapshot SHA-256 are required.")
    v=db[VALIDATION_COLLECTION].find_one({"_id":source_validation_job_id})
    e=db[EXECUTION_COLLECTION].find_one({"_id":source_execution_job_id})
    l=db[LIQUIDITY_COLLECTION].find_one({"_id":source_liquidity_job_id})
    if v is None or e is None or l is None:
        raise TCNNotFound("Required completed v10.8.41, v10.8.42, v10.8.44 jobs not found.")
    source_id=str(v.get("source_job_id") or "")
    data=[v.get("result") or {},e.get("result") or {},l.get("result") or {}]
    if (
        any(row.get("status")!="completed" or row.get("error") for row in (v,e,l))
        or any(
            row.get("snapshot_sha256")!=expected_snapshot_sha256
            or row.get("source_job_id")!=source_id
            for row in (v,e,l)
        )
        or e.get("source_validation_job_id")!=source_validation_job_id
        or l.get("source_validation_job_id")!=source_validation_job_id
        or l.get("source_execution_job_id")!=source_execution_job_id
        or any(
            info.get("source_snapshot_sha256")!=expected_snapshot_sha256
            or info.get("source_job_id")!=source_id
            or info.get("order_submission")!="never"
            or (info.get("numeric_input_integrity") or {}).get("status")!="verified"
            for info in data
        )
        or not (data[0].get("original_shadow") or {}).get("reproduced")
        or data[1].get("research_kind")!="control_execution_feasibility_scenario"
        or data[2].get("research_kind")!="control_liquidity_aware_fixed_hypothesis"
        or data[2].get("control_reference_parity")!="verified"
    ):
        raise TCNInvalid("Research sources do not match the original SHA-verified Control chain.")
    collection=db[COLLECTION]
    collection.create_index("active_key",unique=True,sparse=True)
    job_id="control-tcn-"+uuid.uuid4().hex[:16]
    now=utc_now()
    record={
        "_id":job_id,"status":"queued","stage":"queued","progress":0,
        "active_key":ACTIVE_KEY,
        "source_job_id":source_id,
        "source_validation_job_id":source_validation_job_id,
        "source_execution_job_id":source_execution_job_id,
        "source_liquidity_job_id":source_liquidity_job_id,
        "snapshot_sha256":expected_snapshot_sha256,
        "created_at":now,"updated_at":now,
        "started_at":None,"finished_at":None,
        "logs":[],"result":None,"error":None,
    }
    try:
        collection.insert_one(record)
    except DuplicateKeyError as exc:
        raise TCNConflict("Another TCN Control research job is active.") from exc
    thread=threading.Thread(
        target=_run_job,
        kwargs={
            "db":db,"job_id":job_id,"source_job_id":source_id,
            "validation_id":source_validation_job_id,
            "execution_id":source_execution_job_id,
            "liquidity_id":source_liquidity_job_id,
            "expected_sha256":expected_snapshot_sha256,
            "baseline41":data[0],"baseline42":data[1],"baseline44":data[2],
        },
        name=job_id,daemon=True,
    )
    try:
        _THREADS[job_id]=thread
        thread.start()
    except RuntimeError:
        _THREADS.pop(job_id,None)
        collection.update_one({"_id":job_id},{
            "$set":{"status":"failed","stage":"thread_start_failed",
                    "finished_at":utc_now()},
            "$unset":{"active_key":""},
        })
        raise
    return _public(record)


def get_tcn_research(db:Any,job_id:str,*,logs_only:bool=False)->dict[str,Any]:
    record=db[COLLECTION].find_one({"_id":job_id})
    if record is None:
        raise TCNNotFound("Deep Learning Control research job not found.")
    return _public(record,logs_only=logs_only)

#!/usr/bin/env python3
"""接口层取消/重试/查询闸门规则测试（仅标准库）。

FastAPI 不可用，因此不构造 ASGI 应用，而是按 main.py 中各端点的实际逻辑
复刻其状态判断分支，配合真实 SQLite 状态库验证规则：
  - 进行中任务 DELETE -> 仅置 cancel_requested=1，状态本身不被改写；
  - ready 任务 DELETE -> stopped 且实例停止；
  - cancelled/failed 任务可按点重试生成新 queued 任务；
  - 非 ready 任务的样品查询必须被 409 拦截，取消后绝无查询窗口。
"""
import os
import sys
import tempfile
import types
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

fake_psycopg = types.ModuleType("psycopg")
fake_psycopg.Error = type("FakeError", (Exception,), {})
fake_psycopg.Connection = object
fake_rows = types.ModuleType("psycopg.rows")
fake_rows.dict_row = object
fake_psycopg.rows = fake_rows
sys.modules["psycopg"] = fake_psycopg
sys.modules["psycopg.rows"] = fake_rows

from api.app import config, state  # noqa: E402

ACTIVE = state.ACTIVE_JOB_STATES
RETRYABLE = state.RETRYABLE_JOB_STATES


class ApiRuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="pitr-api-")
        config.STATE_DB = os.path.join(self.tmp, "state.db")
        state.init_db()
        state.execute(
            "INSERT INTO checkpoints (name,restore_lsn,wal_file,status,created_at)"
            " VALUES (?,?,?,?,?)",
            ("cp1", "0/2000000", "000000010000000000000002", "ready", state.now()))

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _job(self, status, cancel=0, data_dir=None):
        return state.execute(
            "INSERT INTO jobs (checkpoint_name,status,cancel_requested,data_dir,"
            "created_at,updated_at) VALUES (?,?,?,?,?,?)",
            ("cp1", status, cancel, data_dir, state.now(), state.now()))

    # ---- 复刻 main.cancel_recovery 的分支 ----
    def cancel_endpoint(self, job_id):
        job = state.one_row("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not job:
            return 404, None
        if job["status"] in ACTIVE:
            state.update_job(job_id, cancel_requested=1, detail="取消请求已受理")
            return 200, state.one_row("SELECT * FROM jobs WHERE id=?", (job_id,))
        if job["status"] == "ready":
            state.update_job(job_id, status="stopped", detail="实例已停止")
            return 200, state.one_row("SELECT * FROM jobs WHERE id=?", (job_id,))
        return 200, job

    # ---- 复刻 main._ready_job_or_409 ----
    def query_gate(self, job_id):
        job = state.one_row("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not job:
            return 404
        if job["status"] != "ready":
            return 409
        return 200

    # ---- 复刻 main.retry_recovery 的核心 ----
    def retry_endpoint(self, job_id):
        job = state.one_row("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not job:
            return 404, None
        if job["status"] not in RETRYABLE:
            return 409, None
        active = ",".join(f"'{s}'" for s in ACTIVE)
        if state.one_row(f"SELECT id FROM jobs WHERE status IN ({active}) LIMIT 1"):
            return 409, None
        new_id = state.execute(
            "INSERT INTO jobs (checkpoint_name,status,created_at,updated_at)"
            " VALUES (?,?,?,?)",
            (job["checkpoint_name"], "queued", state.now(), state.now()))
        return 202, new_id

    def test_cancel_active_job_keeps_status_sets_flag(self):
        for st in ("queued", "waiting_wal", "preparing", "recovering", "verifying"):
            jid = self._job(st)
            code, body = self.cancel_endpoint(jid)
            self.assertEqual(code, 200)
            self.assertEqual(body["status"], st, "受理取消不应立即改写业务状态")
            self.assertEqual(body["cancel_requested"], 1)
            self.assertEqual(self.query_gate(jid), 409, "取消后查询绝不放行")

    def test_cancel_ready_job_stops_instance(self):
        jid = self._job("ready", data_dir="/tmp/x")
        code, body = self.cancel_endpoint(jid)
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "stopped")
        self.assertEqual(self.query_gate(jid), 409)

    def test_retry_after_cancel_and_failure(self):
        for st in ("cancelled", "failed"):
            jid = self._job(st)
            code, new_id = self.retry_endpoint(jid)
            self.assertEqual(code, 202)
            new_job = state.one_row("SELECT * FROM jobs WHERE id=?", (new_id,))
            self.assertEqual(new_job["status"], "queued")
            self.assertEqual(new_job["checkpoint_name"], "cp1")
            self.assertEqual(new_job["cancel_requested"], 0)
            # 新任务终结，避免影响下一轮"一次只恢复一项"判定
            state.update_job(new_id, status="cancelled", cancel_requested=1)

    def test_retry_blocked_while_another_active(self):
        self._job("cancelled")
        self._job("recovering")
        code, _ = self.retry_endpoint(1)
        self.assertEqual(code, 409)

    def test_ready_is_queryable_nothing_else_is(self):
        ready = self._job("ready")
        self.assertEqual(self.query_gate(ready), 200)
        for st in ("queued", "waiting_wal", "preparing", "recovering",
                   "verifying", "failed", "cancelled", "stopped", "superseded"):
            self.assertEqual(self.query_gate(self._job(st)), 409, st)

    def test_cas_guarantees_no_ready_after_accepted_cancel(self):
        # 与 recoverer 发布点相同的原子条件更新，直接压测竞态语义
        jid = self._job("verifying")
        state.update_job(jid, cancel_requested=1)
        self.assertFalse(state.ready_if_not_cancelled(jid, ready_at=state.now()))
        self.assertEqual(
            state.one_row("SELECT status FROM jobs WHERE id=?", (jid,))["status"],
            "verifying")
        state.update_job(jid, cancel_requested=0)
        self.assertTrue(state.ready_if_not_cancelled(jid, ready_at=state.now()))
        self.assertEqual(
            state.one_row("SELECT status FROM jobs WHERE id=?", (jid,))["status"],
            "ready")


if __name__ == "__main__":
    unittest.main(verbosity=2)

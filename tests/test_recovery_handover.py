#!/usr/bin/env python3
"""取消 / 重启 / 重试 / 端口交接的逻辑测试（仅标准库，无需 FastAPI/PostgreSQL）。

环境没有 Docker/Postgres/pip，因此用 fake psycopg + 临时目录替身，
驱动真实的 state.py(SQLite) 与 recoverer.py 状态机，验证：
  1. 校验/提升窗口内受理取消 -> 任务终态 cancelled，绝不 ready，实例被停止；
  2. 取消受理后重启 -> 取消不被清除、不重新排队，重启后仍是 cancelled；
  3. 普通进行中任务重启 -> 重新排队，可按原点重试到 ready；
  4. 端口未释放 -> 下一项恢复被闸门拦截（任务 failed），端口释放后可正常启动；
  5. 取消/失败后按点重试可用；源库模块只读、不含任何 DML。
"""
import os
import shutil
import sys
import tempfile
import types
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

# ---- fake psycopg（recoverer/source 仅在函数内部使用连接，测试全程 mock 掉） ----
fake_psycopg = types.ModuleType("psycopg")


class _FakeError(Exception):
    pass


fake_psycopg.Error = _FakeError
fake_psycopg.Connection = object
fake_rows = types.ModuleType("psycopg.rows")
fake_rows.dict_row = object()
fake_psycopg.rows = fake_rows
sys.modules["psycopg"] = fake_psycopg
sys.modules["psycopg.rows"] = fake_rows

from api.app import config, recoverer, state  # noqa: E402


class RecoveryHandoverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="pitr-test-")
        self.state_db = os.path.join(self.tmp, "state.db")
        self.recovery_dir = os.path.join(self.tmp, "recoveries")
        self.backup_dir = os.path.join(self.tmp, "backups")
        os.makedirs(self.recovery_dir)
        os.makedirs(self.backup_dir)
        config.STATE_DB = self.state_db
        config.RECOVERY_DIR = self.recovery_dir
        config.BACKUP_DIR = self.backup_dir
        config.ARCHIVE_DIR = os.path.join(self.tmp, "archive")
        os.makedirs(config.ARCHIVE_DIR)
        config.POLL_INTERVAL = 0.01
        state.init_db()
        # 测试替身记账
        self.started: list[str] = []          # 已启动的数据目录
        self.stopped: list[str] = []          # 已停止的数据目录
        self.port_busy = False                # 模拟端口占用
        self._patches = []
        self._patch(recoverer.source, "wal_segment_size", lambda: 16 * 1024 * 1024)
        self._patch(recoverer, "_pg_ctl", self._fake_pg_ctl)
        self._patch(recoverer, "_postgres_alive", self._fake_alive)
        self._patch(recoverer, "stop_instance", self._fake_stop)
        self._patch(recoverer, "_wait_replay_paused", self._fake_wait_paused)
        self._patch(recoverer, "_verify_and_promote", self._fake_verify)
        self._patch(recoverer, "_missing_segments", lambda b, c, s: [])
        self._patch(recoverer, "_wait_port_free", self._fake_wait_port)
        self._patch(recoverer, "_write_recovery_conf", lambda d, n: None)
        self._patch(shutil, "copytree", self._fake_copytree)
        self.verify_cancel_hook = None        # 在校验阶段触发取消
        self.seed_fixture()

    def tearDown(self) -> None:
        for obj, name, original in self._patches:
            setattr(obj, name, original)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patch(self, obj, name, value) -> None:
        self._patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    # ---- 替身实现 ----
    def seed_fixture(self) -> None:
        backup_path = os.path.join(self.backup_dir, "base")
        os.makedirs(os.path.join(backup_path, "pg_wal"))
        state.execute(
            "INSERT INTO backups (path,status,start_lsn,timeline,created_at,finished_at)"
            " VALUES (?,?,?,?,?,?)",
            (backup_path, "done", "0/1000000", 1, state.now(), state.now()))
        state.execute(
            "INSERT INTO checkpoints (name,restore_lsn,wal_file,status,created_at,ready_at)"
            " VALUES (?,?,?,?,?,?)",
            ("cp1", "0/2000000", "000000010000000000000002",
             "ready", state.now(), state.now()))

    def insert_job(self, cancel: int = 0, status: str = "queued",
                   data_dir: str | None = None) -> int:
        return state.execute(
            "INSERT INTO jobs (checkpoint_name,status,cancel_requested,data_dir,"
            "created_at,updated_at) VALUES (?,?,?,?,?,?)",
            ("cp1", status, cancel, data_dir, state.now(), state.now()))

    def _fake_copytree(self, src, dst, **kw):
        os.makedirs(dst)
        open(os.path.join(dst, "postmaster.pid"), "w").close()

    def _fake_pg_ctl(self, ddir, action, logfile):
        if action == "start":
            self.started.append(ddir)
            open(os.path.join(ddir, "postmaster.pid"), "w").close()
        elif action == "stop":
            self.stopped.append(ddir)

    def _fake_alive(self, ddir):
        return ddir in self.started and ddir not in self.stopped

    def _fake_stop(self, ddir):
        recoverer._wait_port_free()
        self.stopped.append(ddir)

    def _fake_wait_port(self, timeout=90.0):
        if self.port_busy:
            raise RuntimeError("恢复端口 127.0.0.1:55433 在测试中被占用，拒绝启动")

    def _fake_wait_paused(self, job_id, cp, ddir):
        return None

    def _fake_verify(self, cp, ddir, job_id):
        # 模拟"取消在校验/提升阶段受理"：promote 刚完成、ready 发布之前
        if self.verify_cancel_hook:
            self.verify_cancel_hook(job_id)
        return 3

    # ---- 用例 ----
    def test_01_cancel_during_verify_never_publishes_ready(self):
        jid = self.insert_job()
        self.verify_cancel_hook = lambda j: state.update_job(j, cancel_requested=1)
        recoverer.run_job(jid)
        job = state.one_row("SELECT * FROM jobs WHERE id=?", (jid,))
        self.assertEqual(job["status"], "cancelled")
        self.assertIsNone(job["ready_at"])
        ddir = os.path.join(self.recovery_dir, f"job-{jid}")
        self.assertIn(ddir, self.stopped, "已启动的恢复实例必须退出")
        self.assertEqual(job["cancel_requested"], 1)

    def test_02_cancel_before_promote_stops_instance(self):
        jid = self.insert_job()

        def hook(j):
            state.update_job(j, cancel_requested=1)
            # 取消点落在 _verify_and_promote 内部的 resume 前检查
            recoverer._check_cancel(j)

        self.verify_cancel_hook = hook
        recoverer.run_job(jid)
        job = state.one_row("SELECT * FROM jobs WHERE id=?", (jid,))
        self.assertEqual(job["status"], "cancelled")
        ddir = os.path.join(self.recovery_dir, f"job-{jid}")
        self.assertIn(ddir, self.stopped)

    def test_03_cas_blocks_ready_when_cancel_races(self):
        # 直接验证原子交接：cancel_requested=1 时 ready 发布必须失败
        jid = self.insert_job()
        state.update_job(jid, cancel_requested=1)
        ok = state.ready_if_not_cancelled(jid, detail="x", ready_at=state.now())
        self.assertFalse(ok)
        self.assertEqual(state.one_row("SELECT status FROM jobs WHERE id=?", (jid,))[
                             "status"], "queued")
        state.update_job(jid, cancel_requested=0)
        ok = state.ready_if_not_cancelled(jid, detail="x", ready_at=state.now())
        self.assertTrue(ok)
        self.assertEqual(state.one_row("SELECT status FROM jobs WHERE id=?", (jid,))[
                             "status"], "ready")

    def test_04_restart_preserves_pending_cancel(self):
        ddir = os.path.join(self.recovery_dir, "job-7")
        os.makedirs(ddir)
        jid = self.insert_job(cancel=1, status="verifying", data_dir=ddir)
        recoverer.resume_after_restart()
        job = state.one_row("SELECT * FROM jobs WHERE id=?", (jid,))
        self.assertEqual(job["status"], "cancelled", "重启不得清除未完成的取消")
        self.assertEqual(job["cancel_requested"], 1, "取消标记必须保留")
        self.assertIn(ddir, self.stopped, "遗留恢复实例必须退出")

    def test_05_restart_requeues_normal_active_job(self):
        ddir = os.path.join(self.recovery_dir, "job-8")
        os.makedirs(ddir)
        jid = self.insert_job(cancel=0, status="recovering", data_dir=ddir)
        recoverer.resume_after_restart()
        job = state.one_row("SELECT * FROM jobs WHERE id=?", (jid,))
        self.assertEqual(job["status"], "queued")

    def test_06_port_gate_blocks_next_recovery_until_cleaned(self):
        jid = self.insert_job()
        self.port_busy = True  # 上一实例未清理，端口仍占用
        recoverer.run_job(jid)
        job = state.one_row("SELECT * FROM jobs WHERE id=?", (jid,))
        self.assertEqual(job["status"], "failed")
        self.assertIn("端口", job["error"])
        # 清理结束、端口释放后，重试可以正常跑到 ready
        self.port_busy = False
        new_id = state.execute(
            "INSERT INTO jobs (checkpoint_name,status,created_at,updated_at)"
            " VALUES (?,?,?,?)", ("cp1", "queued", state.now(), state.now()))
        recoverer.run_job(new_id)
        self.assertEqual(
            state.one_row("SELECT status FROM jobs WHERE id=?", (new_id,))["status"],
            "ready")

    def test_07_retry_after_cancel_and_failure(self):
        # 取消后按点重试
        jid = self.insert_job()
        self.verify_cancel_hook = lambda j: state.update_job(j, cancel_requested=1)
        recoverer.run_job(jid)
        self.assertEqual(
            state.one_row("SELECT status FROM jobs WHERE id=?", (jid,))["status"],
            "cancelled")
        retry_id = state.execute(
            "INSERT INTO jobs (checkpoint_name,status,created_at,updated_at)"
            " VALUES (?,?,?,?)", ("cp1", "queued", state.now(), state.now()))
        self.verify_cancel_hook = None
        recoverer.run_job(retry_id)
        self.assertEqual(
            state.one_row("SELECT status FROM jobs WHERE id=?", (retry_id,))["status"],
            "ready")

    def test_08_source_module_has_no_dml(self):
        # 源库保持可写：API 源库模块实际下发的 SQL 只能是只读/元数据调用
        import ast
        path = os.path.join(WORKSPACE, "api", "app", "source.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        sqls = []
        for node in ast.walk(tree):
            # 只看 .execute("...") 调用的首个字符串参数，忽略注释/文档字符串
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "execute" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    sqls.append(arg.value.strip().lower())
        self.assertTrue(sqls)
        for sql in sqls:
            head = sql.lstrip("(").split(None, 1)[0]
            self.assertIn(head, ("select", "show"),
                          f"source.py 出现写操作 SQL: {sql!r}")

    def test_09_happy_path_still_publishes_ready_and_instance_lives(self):
        jid = self.insert_job()
        recoverer.run_job(jid)
        job = state.one_row("SELECT * FROM jobs WHERE id=?", (jid,))
        self.assertEqual(job["status"], "ready")
        self.assertIsNotNone(job["ready_at"])
        ddir = os.path.join(self.recovery_dir, f"job-{jid}")
        self.assertIn(ddir, self.started)
        self.assertNotIn(ddir, self.stopped)


if __name__ == "__main__":
    unittest.main(verbosity=2)

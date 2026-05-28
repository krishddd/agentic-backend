"""test_ledger_wal — Concurrent read + write doesn't crash."""
import os, sys, sqlite3, pytest, threading
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from src.abm.environment import InteractionLedger

class TestLedgerWAL:
    @pytest.fixture
    def ledger(self, tmp_path):
        return InteractionLedger(str(tmp_path / "wal.db"))

    def test_wal_mode_enabled(self, ledger):
        mode = ledger.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_concurrent_read_write(self, tmp_path):
        path = str(tmp_path / "conc.db")
        writer = InteractionLedger(path)
        writer.record_post(0, 1, "seed", 0.5)

        reader_conn = sqlite3.connect(path)
        reader_conn.execute("PRAGMA journal_mode=WAL")
        reader_conn.row_factory = sqlite3.Row
        count_before = reader_conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        assert count_before == 1

        writer.record_post(1, 2, "concurrent", 0.3)
        count_after = reader_conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        assert count_after == 2

        reader_conn.close()
        writer.close()

    def test_threaded_writes(self, tmp_path):
        path = str(tmp_path / "threaded.db")
        errors = []

        def writer_fn(agent_id, n_posts):
            try:
                conn = sqlite3.connect(path, timeout=10)
                conn.execute("PRAGMA journal_mode=WAL")
                for i in range(n_posts):
                    conn.execute(
                        "INSERT INTO posts (tick, agent_id, content, sentiment) "
                        "VALUES (?,?,?,?)",
                        (i, agent_id, f"post_{i}", 0.1 * agent_id))
                    conn.commit()
                conn.close()
            except Exception as e:
                errors.append(str(e))

        # Pre-create the database schema
        ledger = InteractionLedger(path)
        ledger.close()

        t1 = threading.Thread(target=writer_fn, args=(1, 20))
        t2 = threading.Thread(target=writer_fn, args=(2, 20))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(errors) == 0, f"Threading errors: {errors}"
        ledger = InteractionLedger(path)
        assert ledger.count_posts(100) == 40
        ledger.close()

    def test_read_during_bulk_write(self, tmp_path):
        path = str(tmp_path / "bulk.db")
        ledger = InteractionLedger(path)
        for i in range(50):
            ledger.record_post(0, i, f"post_{i}", 0.0)

        reader = sqlite3.connect(path)
        reader.execute("PRAGMA journal_mode=WAL")
        count = reader.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        assert count == 50

        ledger.record_post(1, 99, "extra", 0.5)
        count2 = reader.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        assert count2 == 51
        reader.close()
        ledger.close()

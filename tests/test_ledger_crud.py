"""test_ledger_crud — All CRUD + WAL mode enabled."""
import os, sys, pytest
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from src.abm.environment import InteractionLedger

class TestLedgerCRUD:
    @pytest.fixture
    def ledger(self, tmp_path):
        return InteractionLedger(str(tmp_path / "test.db"))

    def test_record_post_returns_id(self, ledger):
        pid = ledger.record_post(0, 1, "Hello", 0.5)
        assert isinstance(pid, int) and pid > 0

    def test_count_posts(self, ledger):
        ledger.record_post(0, 1, "a", 0.1)
        ledger.record_post(1, 2, "b", 0.2)
        assert ledger.count_posts(1) == 2

    def test_record_action(self, ledger):
        ledger.record_action(0, 1, "like", target_post_id=99)
        rows = ledger.conn.execute("SELECT * FROM actions").fetchall()
        assert len(rows) == 1
        assert rows[0]["action_type"] == "like"

    def test_record_follow(self, ledger):
        ledger.record_follow(1, 2, 0)
        rows = ledger.conn.execute("SELECT * FROM follows").fetchall()
        assert len(rows) == 1
        assert rows[0]["follower_id"] == 1

    def test_duplicate_follow_ignored(self, ledger):
        ledger.record_follow(1, 2, 0)
        ledger.record_follow(1, 2, 1)
        rows = ledger.conn.execute("SELECT * FROM follows").fetchall()
        assert len(rows) == 1

    def test_record_agent_state(self, ledger):
        ledger.record_agent_state(1, 0, 0.5, 0.8)
        rows = ledger.conn.execute("SELECT * FROM agent_states").fetchall()
        assert len(rows) == 1
        assert rows[0]["sentiment"] == 0.5

    def test_increment_post_stat_likes(self, ledger):
        pid = ledger.record_post(0, 1, "test", 0.0)
        ledger.increment_post_stat(pid, "likes")
        row = ledger.conn.execute("SELECT likes FROM posts WHERE post_id=?", (pid,)).fetchone()
        assert row[0] == 1

    def test_increment_post_stat_reposts(self, ledger):
        pid = ledger.record_post(0, 1, "test", 0.0)
        ledger.increment_post_stat(pid, "reposts")
        row = ledger.conn.execute("SELECT reposts FROM posts WHERE post_id=?", (pid,)).fetchone()
        assert row[0] == 1

    def test_increment_invalid_field_raises(self, ledger):
        pid = ledger.record_post(0, 1, "test", 0.0)
        with pytest.raises(ValueError, match="Invalid stat field"):
            ledger.increment_post_stat(pid, "invalid_field")

    def test_get_sentiment_snapshot(self, ledger):
        ledger.record_agent_state(1, 0, 0.5, 0.8)
        ledger.record_agent_state(2, 0, -0.5, 0.3)
        ledger.record_agent_state(3, 0, 0.0, 0.5)
        snap = ledger.get_sentiment_snapshot(0)
        assert snap["bullish"] > 0
        assert snap["bearish"] > 0
        assert snap["neutral"] > 0
        assert abs(snap["bullish"] + snap["bearish"] + snap["neutral"] - 1.0) < 0.01

    def test_get_most_engaged_posts(self, ledger):
        pid = ledger.record_post(0, 1, "popular", 0.5)
        for _ in range(5): ledger.increment_post_stat(pid, "likes")
        ledger.record_post(0, 2, "boring", 0.1)
        top = ledger.get_most_engaged_posts(limit=2)
        assert len(top) == 2
        assert top[0]["content"] == "popular"

    def test_close_and_reopen(self, tmp_path):
        path = str(tmp_path / "persist.db")
        ledger = InteractionLedger(path)
        ledger.record_post(0, 1, "persist_me", 0.5)
        ledger.close()
        ledger2 = InteractionLedger(path)
        assert ledger2.count_posts(0) == 1
        ledger2.close()

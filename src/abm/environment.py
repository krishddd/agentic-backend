"""
SQLite interaction ledger + recommendation engine — Pro Edition.

The InteractionLedger is the central "environment" — all agent actions
(posts, replies, likes, follows) are recorded here. The RecommendationEngine
curates a weighted feed with dual-channel support (Microblog + Forum).

Pro additions:
- ``channel`` column on posts: MICROBLOG (fast/viral) vs FORUM (analytical)
- Dual-channel feed filtering: 70% home-channel + 30% cross-channel discovery
- 5-point sentiment snapshots
- Agent sentiment history retrieval for analysis module

WAL mode is enabled for concurrent read/write safety during Phase 5
scenario injection (Flask thread writing while Mesa simulation reads).
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class InteractionLedger:
    """SQLite-backed interaction ledger — the ABM environment (Pro Edition).

    All agent interactions are stored here. The ledger persists to disk
    so it can be analyzed post-simulation by the ReportAgent.
    """

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        # WAL mode: allows concurrent reads + writes (critical for Phase 5
        # scenario injection where Flask thread writes while simulation reads)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.debug(f"InteractionLedger opened: {db_path}")

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS posts (
                post_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick INTEGER, agent_id INTEGER, content TEXT,
                sentiment REAL, likes INTEGER DEFAULT 0,
                reposts INTEGER DEFAULT 0, replies INTEGER DEFAULT 0,
                channel TEXT DEFAULT 'microblog'
            );
            CREATE TABLE IF NOT EXISTS actions (
                action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick INTEGER, agent_id INTEGER, action_type TEXT,
                target_post_id INTEGER DEFAULT -1,
                target_agent_id INTEGER DEFAULT -1,
                content TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS follows (
                follower_id INTEGER, followed_id INTEGER,
                since_tick INTEGER,
                PRIMARY KEY (follower_id, followed_id)
            );
            CREATE TABLE IF NOT EXISTS agent_states (
                agent_id INTEGER, tick INTEGER,
                sentiment REAL, influence REAL,
                PRIMARY KEY (agent_id, tick)
            );
            CREATE INDEX IF NOT EXISTS idx_posts_tick ON posts(tick);
            CREATE INDEX IF NOT EXISTS idx_actions_tick ON actions(tick);
            CREATE INDEX IF NOT EXISTS idx_follows_follower ON follows(follower_id);
            CREATE INDEX IF NOT EXISTS idx_agent_states_tick ON agent_states(tick);
        """)
        # Schema migration: add 'channel' column to existing posts tables
        try:
            self.conn.execute(
                "SELECT channel FROM posts LIMIT 1")
        except sqlite3.OperationalError:
            self.conn.execute(
                "ALTER TABLE posts ADD COLUMN channel "
                "TEXT DEFAULT 'microblog'")
            self.conn.commit()
        # Now safe to create the channel index
        try:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_posts_channel "
                "ON posts(channel)")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # index already exists

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def record_post(self, tick: int, agent_id: int, content: str,
                    sentiment: float, channel: str = "microblog") -> int:
        """Insert a new post and return its post_id.

        Args:
            channel: 'microblog' (short/fast) or 'forum' (long/analytical).
        """
        cur = self.conn.execute(
            "INSERT INTO posts (tick, agent_id, content, sentiment, channel) "
            "VALUES (?,?,?,?,?)",
            (tick, agent_id, content, sentiment, channel))
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def record_action(self, tick: int, agent_id: int, action_type: str,
                      target_post_id: int = -1, target_agent_id: int = -1,
                      content: str = ""):
        """Record any social action (like, repost, follow, etc.)."""
        self.conn.execute(
            "INSERT INTO actions (tick, agent_id, action_type, "
            "target_post_id, target_agent_id, content) VALUES (?,?,?,?,?,?)",
            (tick, agent_id, action_type, target_post_id,
             target_agent_id, content))
        self.conn.commit()

    def record_follow(self, follower_id: int, followed_id: int, tick: int):
        """Insert a follow relationship (ignore if duplicate)."""
        self.conn.execute(
            "INSERT OR IGNORE INTO follows VALUES (?,?,?)",
            (follower_id, followed_id, tick))
        self.conn.commit()

    def record_agent_state(self, agent_id: int, tick: int,
                           sentiment: float, influence: float):
        """Snapshot an agent's state at a given tick."""
        self.conn.execute(
            "INSERT OR REPLACE INTO agent_states VALUES (?,?,?,?)",
            (agent_id, tick, sentiment, influence))
        self.conn.commit()

    def increment_post_stat(self, post_id: int, field: str):
        """Increment a post's engagement counter.

        Args:
            field: One of 'likes', 'reposts', 'replies'.
        """
        if field not in ("likes", "reposts", "replies"):
            raise ValueError(f"Invalid stat field: {field}")
        self.conn.execute(
            f"UPDATE posts SET {field} = {field} + 1 WHERE post_id = ?",
            (post_id,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def count_posts(self, tick: int) -> int:
        """Count total posts up to and including the given tick."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM posts WHERE tick <= ?", (tick,)).fetchone()
        return row[0]

    def get_most_engaged_posts(self, limit: int = 20,
                                channel: str = "") -> list:
        """Posts ranked by engagement score: likes + 2*reposts + 3*replies.

        Args:
            channel: Filter by channel ('microblog'/'forum'). Empty = all.
        """
        if channel:
            return self.conn.execute(
                "SELECT *, (likes + 2*reposts + 3*replies) AS score "
                "FROM posts WHERE channel = ? ORDER BY score DESC LIMIT ?",
                (channel, limit)).fetchall()
        return self.conn.execute(
            "SELECT *, (likes + 2*reposts + 3*replies) AS score "
            "FROM posts ORDER BY score DESC LIMIT ?", (limit,)).fetchall()

    def get_posts_by_influence(self, min_influence: float = 0.7) -> list:
        """Posts from agents whose latest recorded influence >= threshold."""
        return self.conn.execute("""
            SELECT p.*, s.influence FROM posts p
            JOIN agent_states s ON p.agent_id = s.agent_id
            WHERE s.influence >= ? AND s.tick = (
                SELECT MAX(tick) FROM agent_states
                WHERE agent_id = s.agent_id)
            ORDER BY p.tick DESC LIMIT 50
        """, (min_influence,)).fetchall()

    def get_sentiment_snapshot(self, tick: int) -> Dict[str, float]:
        """Return 5-point sentiment proportions at a given tick.

        Returns dict with keys: strongly_bearish, bearish, neutral,
        bullish, strongly_bullish.
        """
        rows = self.conn.execute(
            "SELECT sentiment FROM agent_states WHERE tick = ?",
            (tick,)).fetchall()
        if not rows:
            return {"strongly_bearish": 0.0, "bearish": 0.0,
                    "neutral": 0.0, "bullish": 0.0,
                    "strongly_bullish": 0.0}
        sentiments = [r[0] for r in rows]
        total = len(sentiments)
        return {
            "strongly_bullish": sum(
                1 for s in sentiments if s >= 0.6) / total,
            "bullish": sum(
                1 for s in sentiments if 0.2 <= s < 0.6) / total,
            "neutral": sum(
                1 for s in sentiments if -0.2 <= s < 0.2) / total,
            "bearish": sum(
                1 for s in sentiments if -0.6 <= s < -0.2) / total,
            "strongly_bearish": sum(
                1 for s in sentiments if s < -0.6) / total,
        }

    def get_channel_sentiment_snapshot(
        self, tick: int
    ) -> Dict[str, Dict[str, float]]:
        """Return 5-point sentiment per channel at a given tick.

        Returns dict keyed by channel ('microblog', 'forum'), each
        containing a sentiment distribution dict.
        """
        result = {}
        for ch in ("microblog", "forum"):
            rows = self.conn.execute("""
                SELECT DISTINCT s.sentiment FROM agent_states s
                JOIN posts p ON p.agent_id = s.agent_id AND p.tick = s.tick
                WHERE s.tick = ? AND p.channel = ?
            """, (tick, ch)).fetchall()
            if not rows:
                result[ch] = {"strongly_bearish": 0.0, "bearish": 0.0,
                              "neutral": 0.0, "bullish": 0.0,
                              "strongly_bullish": 0.0}
                continue
            sentiments = [r[0] for r in rows]
            total = len(sentiments)
            result[ch] = {
                "strongly_bullish": sum(
                    1 for s in sentiments if s >= 0.6) / total,
                "bullish": sum(
                    1 for s in sentiments if 0.2 <= s < 0.6) / total,
                "neutral": sum(
                    1 for s in sentiments if -0.2 <= s < 0.2) / total,
                "bearish": sum(
                    1 for s in sentiments if -0.6 <= s < -0.2) / total,
                "strongly_bearish": sum(
                    1 for s in sentiments if s < -0.6) / total,
            }
        return result

    def get_agent_sentiments_at_tick(
        self, tick: int
    ) -> List[Tuple[int, float]]:
        """Return list of (agent_id, sentiment) tuples at a given tick.
        Used by the analysis module for coalition identification.
        """
        rows = self.conn.execute(
            "SELECT agent_id, sentiment FROM agent_states WHERE tick = ?",
            (tick,)).fetchall()
        return [(r[0], r[1]) for r in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.debug(f"InteractionLedger closed: {self.db_path}")


class RecommendationEngine:
    """Curates agent feeds with dual-channel support — Pro Edition.

    Feed composition:
    - 70% home-channel posts from followed agents (interest-based)
    - 30% cross-channel discovery + hot-score trending

    This mirrors the MiroFish dual-platform design where Microblog
    favours virality and Forum favours analytical depth.
    """

    def __init__(self, ledger: InteractionLedger,
                 interest_weight: float = 0.7):
        self.ledger = ledger
        self.interest_weight = interest_weight

    def curate_feed(self, agent_id: int, tick: int,
                    limit: int = 10,
                    home_channel: str = "microblog") -> list:
        """Return a weighted, deduplicated feed for the agent.

        Args:
            home_channel: Agent's preferred channel. 70% of interest feed
                          comes from this channel, 30% from cross-channel.
        """
        # Home-channel interest feed (followed agents, home channel)
        interest_home = self._interest_feed(
            agent_id, tick, limit, channel=home_channel)
        # Cross-channel discovery
        cross_ch = "forum" if home_channel == "microblog" else "microblog"
        interest_cross = self._interest_feed(
            agent_id, tick, limit // 3, channel=cross_ch)
        # Global hot feed (all channels)
        hot = self._hot_feed(tick, limit)

        # Weighted merge, deduplicated by post_id
        seen: set = set()
        merged: list = []
        n_interest = int(limit * self.interest_weight)

        for post in interest_home[:n_interest]:
            pid = post["post_id"] if isinstance(post, dict) else post[0]
            if pid not in seen:
                seen.add(pid)
                merged.append(post)

        # Cross-channel discovery (30% of interest)
        for post in interest_cross:
            pid = post["post_id"] if isinstance(post, dict) else post[0]
            if pid not in seen and len(merged) < n_interest:
                seen.add(pid)
                merged.append(post)

        for post in hot:
            pid = post["post_id"] if isinstance(post, dict) else post[0]
            if pid not in seen and len(merged) < limit:
                seen.add(pid)
                merged.append(post)

        return merged

    def _interest_feed(self, agent_id: int, tick: int,
                       limit: int, channel: str = "") -> list:
        """Posts from agents that this agent follows.

        Args:
            channel: If specified, filter to this channel only.
        """
        if channel:
            return self.ledger.conn.execute("""
                SELECT p.* FROM posts p
                JOIN follows f ON p.agent_id = f.followed_id
                              AND f.follower_id = ?
                WHERE p.tick <= ? AND p.channel = ?
                ORDER BY p.tick DESC, p.likes DESC
                LIMIT ?
            """, (agent_id, tick, channel, limit)).fetchall()
        return self.ledger.conn.execute("""
            SELECT p.* FROM posts p
            JOIN follows f ON p.agent_id = f.followed_id
                          AND f.follower_id = ?
            WHERE p.tick <= ?
            ORDER BY p.tick DESC, p.likes DESC
            LIMIT ?
        """, (agent_id, tick, limit)).fetchall()

    def _hot_feed(self, tick: int, limit: int) -> list:
        """Globally trending posts by engagement score."""
        return self.ledger.conn.execute("""
            SELECT *, (likes + 2*reposts + 3*replies) AS score
            FROM posts WHERE tick <= ?
            ORDER BY score DESC LIMIT ?
        """, (tick, limit)).fetchall()

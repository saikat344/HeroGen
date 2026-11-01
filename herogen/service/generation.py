"""Guest workspaces and the transactional live-generation spend guard."""
from decimal import Decimal, InvalidOperation

SCHEMA = """
CREATE TABLE generation_settings (
 id INTEGER PRIMARY KEY CHECK(id=1), enabled INTEGER NOT NULL DEFAULT 0,
 limit_usd REAL NOT NULL DEFAULT 0, running_estimate_usd REAL NOT NULL DEFAULT 1.50,
 max_running INTEGER NOT NULL
);
"""


def amount(value):
    try:
        if isinstance(value, bool):
            raise ValueError
        n = Decimal(str(value))
        if not n.is_finite() or n < 0 or n != n.quantize(Decimal('.000001')):
            raise ValueError
        return float(n)
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError('USD amounts must be non-negative with at most six decimal places.') from None


class GenerationStore:
    def guest_open(self, session_hash: str, session_ttl_s: float, idle_s: float, old_session_hash: str = '') -> dict:
        """Reuse a valid browser session; guest access alone never reserves or starts compute."""
        from herogen.service.store import new_id
        with self._tx() as c:
            now = self.clock()
            prior = c.execute("SELECT s.user_id FROM sessions s JOIN users u ON u.id=s.user_id "
                "WHERE s.hash=? AND s.revoked_at IS NULL AND s.expires_at>? AND s.last_seen_at>? "
                "AND u.deleted_at IS NULL AND u.status='active' AND u.kind='guest'",
                (old_session_hash, now, now-idle_s)).fetchone()
            uid = prior['user_id'] if prior else new_id('u')
            if not prior:
                c.execute('INSERT INTO users(id,email,created_at,kind) VALUES (?,?,?,?)',
                          (uid, f'workspace:{uid}', now, 'guest'))
            c.execute('INSERT INTO sessions(hash,user_id,created_at,expires_at,last_seen_at) VALUES (?,?,?,?,?)',
                      (session_hash, uid, now, now + session_ttl_s, now))
            if old_session_hash:
                c.execute('UPDATE sessions SET revoked_at=? WHERE hash=?', (now, old_session_hash))
            self._audit(c, uid, 'guest_workspace_opened')
        return self.user_get(uid)


    def _generation(self, c):
        row = dict(c.execute('SELECT * FROM generation_settings WHERE id=1').fetchone())
        row.pop('id')
        row['enabled'] = bool(row['enabled'])
        spent = c.execute("SELECT COALESCE(SUM(cost_usd),0) FROM jobs WHERE state NOT IN ('queued','running')").fetchone()[0]
        active = c.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]
        running = round(active * row['running_estimate_usd'], 6)
        reason = ('generation_paused' if not row['enabled'] else 'generation_budget_exhausted'
                  if Decimal(str(round(spent, 6))) + Decimal(str(running)) + Decimal(str(row['running_estimate_usd'])) > Decimal(str(row['limit_usd']))
                  else 'available')
        return {**row, 'spent_usd': round(spent, 6), 'running_usd': running, 'active': active,
                'remaining_usd': max(0, round(row['limit_usd'] - spent - running, 6)),
                'accepting': reason == 'available', 'reason': reason,
                'sentence': (f"Accepting heroes now: up to {row['max_running']} at a time, ${spent:.2f} of ${row['limit_usd']:g} spent"
                             if reason == 'available' else 'Live generation is paused right now.'
                             if reason == 'generation_paused' else 'The live-generation budget is used up for now.')}

    def generation_get(self):
        with self._tx() as c:
            self._expire_leases(c)
            return self._generation(c)

    def generation_update(self, *, actor, enabled=None, limit_usd=None, running_estimate_usd=None, max_running=None):
        changes = {}
        if enabled is not None:
            if type(enabled) is not bool:
                raise ValueError('Enabled must be true or false.')
            changes['enabled'] = int(enabled)
        for key, value in (('limit_usd', limit_usd), ('running_estimate_usd', running_estimate_usd)):
            if value is not None:
                changes[key] = amount(value)
        if max_running is not None:
            if type(max_running) is not int or not 1 <= max_running <= 32:
                raise ValueError('Max running must be an integer from 1 to 32.')
            changes['max_running'] = max_running
        with self._tx() as c:
            self._expire_leases(c)
            if changes:
                c.execute('UPDATE generation_settings SET '+','.join(k+'=?' for k in changes)+' WHERE id=1', tuple(changes.values()))
                self._audit(c, actor, 'generation_settings_updated', detail=changes)
            return self._generation(c)

    def _guard_generation(self, c):
        from herogen.service.store import StoreError
        self._expire_leases(c)
        status = self._generation(c)
        if not status['accepting']:
            message = ('Live generation is paused right now.' if status['reason'] == 'generation_paused'
                       else 'The live-generation budget is used up for now.')
            raise StoreError(message, status['reason'])

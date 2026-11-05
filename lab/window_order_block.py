"""Lab reading of 15M Window Order Block Strategy.

The fitted sample is replayed per outright while that outright is the daily
volume leader. A 15-minute leg qualifies only after a same-direction sweep of
the last closed 4-hour candle. The last block of an hour confirms from :45
through :05. Its retest is the next hour's opening wick, :00 through :15, and
the lower-timeframe block that overlaps it confirms the expansion there.
The order is taken only while that higher-timeframe block is still in the book
and the signal candle trades it. A deleted block does not leave an entry.
Buckets are anchored to the Pyne session: Sunday 18:00 New York, and midnight
New York on Monday through Friday. A stop or target that trades fills on the
one-minute path. The open walks to the nearer extreme first, and an equal
distance walks to the high first. One MES point is 5 dollars.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
POINT = 5.0
MIDPOINT = datetime(2026, 8, 1, tzinfo=NY)

BARS = "/home/vesper/Trading/AI/ai-futures-trading-lab/data/mes-fut-20260501-20261030.ohlcv-1m.csv.zst"
VOLUME = "/home/vesper/Trading/AI/ai-futures-trading-lab/data/mes-fut-20260501-20261030.volume.json"


@dataclass
class Bar:
    t: datetime
    o: float
    h: float
    l: float
    c: float


@dataclass
class Candle:
    t: datetime
    left: int
    right: int
    o: float
    h: float
    l: float
    c: float


@dataclass
class Candidate:
    dir: int
    start: int
    end: int
    open_price: float
    top: float
    bot: float
    start_time: datetime
    through_count: int = 0
    drawn: bool = False
    zone_id: int = -1
    armed_bar: int = -1
    retest_bar: int = -1
    retest_time: datetime | None = None
    retest_close: float | None = None
    retest_high: float | None = None
    retest_low: float | None = None


@dataclass
class Zone:
    top: float
    bottom: float
    created_bar: int
    live: bool
    uid: int
    dir: int
    start_bar: int
    hit_top: bool = False
    hit_bot: bool = False
    departed: bool = False
    tapped: bool = False
    confirm_time: datetime | None = None
    anchor_time: datetime | None = None


@dataclass
class Leg:
    cur_dir: int = 0
    cur_start: int = 0
    cur_end: int = 0
    cur_open: float = 0.0
    cur_top: float = 0.0
    cur_bot: float = 0.0
    cur_start_time: datetime | None = None
    cur_count: int = 0
    cur_from_window: bool = False


@dataclass
class Position:
    side: int
    entry: float
    stop: float
    target: float | None
    opened: datetime
    day: int


@dataclass
class Closed:
    opened: datetime
    exited: datetime
    side: int
    entry: float
    exit: float
    pnl: float
    reason: str


def session_anchor(ny: datetime) -> datetime | None:
    local = ny.replace(second=0, microsecond=0)
    weekday = local.weekday()
    if weekday == 5:
        return None
    if weekday == 6:
        anchor = local.replace(hour=18, minute=0)
        if local < anchor:
            return None
        return anchor
    if weekday == 4 and local.hour > 16:
        return None
    return local.replace(hour=0, minute=0)


def bucket_start(ny: datetime, minutes: int) -> datetime | None:
    anchor = session_anchor(ny)
    if anchor is None:
        return None
    elapsed = int((ny - anchor).total_seconds() // 60)
    return anchor + timedelta(minutes=(elapsed // minutes) * minutes)


def in_window(ny: datetime) -> bool:
    minute_of_day = ny.hour * 60 + ny.minute
    return 9 * 60 + 30 <= minute_of_day < 16 * 60


def in_confirm_minute(ny: datetime, start: int, end: int) -> bool:
    minute_of_hour = ny.minute
    if start > end:
        return minute_of_hour >= start or minute_of_hour <= end
    return start <= minute_of_hour <= end


def in_next_opening(confirm_time: datetime, probe: datetime) -> bool:
    minute_of_hour = confirm_time.minute
    if minute_of_hour >= 45:
        shift = 60 - minute_of_hour
    elif minute_of_hour <= 5:
        shift = -minute_of_hour
    else:
        return False
    open_start = confirm_time + timedelta(minutes=shift)
    open_end = open_start + timedelta(minutes=15)
    return open_start <= probe <= open_end


def linked_confirm(zones: list[Zone], direction: int, top: float, bot: float, probe: datetime) -> datetime | None:
    for zone in reversed(zones):
        overlaps = zone.top >= bot and zone.bottom <= top
        if zone.dir == direction and overlaps and zone.confirm_time is not None and in_next_opening(zone.confirm_time, probe):
            return zone.confirm_time
    return None


def linked_retest(zones: list[Zone], direction: int, top: float, bot: float, probe: datetime | None, bar_high: float | None, bar_low: float | None) -> datetime | None:
    if probe is None or bar_high is None or bar_low is None:
        return None
    for zone in reversed(zones):
        overlaps = zone.top >= bot and zone.bottom <= top
        traded = bar_high >= zone.bottom and bar_low <= zone.top
        if zone.dir == direction and overlaps and traded and zone.confirm_time is not None and in_next_opening(zone.confirm_time, probe):
            return zone.confirm_time
    return None


def day_key(ny: datetime) -> int:
    return ny.year * 10000 + ny.month * 100 + ny.day


def hour_key(ny: datetime) -> int:
    return ny.year * 1000000 + ny.month * 10000 + ny.day * 100 + ny.hour


def path_of(bar: Bar) -> tuple[float, float, float, float]:
    to_low = abs(bar.o - bar.l)
    to_high = abs(bar.h - bar.o)
    if to_low < to_high:
        return (bar.o, bar.l, bar.h, bar.c)
    return (bar.o, bar.h, bar.l, bar.c)


class Book:
    def __init__(self) -> None:
        self.htf = Leg()
        self.ltf = Leg()
        self.htf_pending: list[Candidate] = []
        self.ltf_pending: list[Candidate] = []
        self.htf_zones: list[Zone] = []
        self.ltf_zones: list[Zone] = []
        self.seq = 0
        self.hours: dict[int, list[float]] = {}
        self.bull_valid = False
        self.bull_time: datetime | None = None
        self.bull_level = 0.0
        self.bear_valid = False
        self.bear_time: datetime | None = None
        self.bear_level = 0.0
        self.closed_liq: Candle | None = None
        self.h4_start: datetime | None = None
        self.h4_left = 0
        self.h4_o = self.h4_h = self.h4_l = self.h4_c = 0.0
        self.m15_start: datetime | None = None
        self.m15_left = 0
        self.m15_o = self.m15_h = self.m15_l = self.m15_c = 0.0
        self.highs: list[float] = []
        self.lows: list[float] = []
        self.times: list[datetime] = []
        self.swings: list[tuple[float, int, datetime]] = []
        self.order_day = -1
        self.position: Position | None = None
        self.closed: list[Closed] = []
        self.closed_pnl = 0.0
        self.peak = 0.0
        self.max_dd = 0.0
        self.index = -1

    def note_hour(self, bar: Bar) -> None:
        key = hour_key(bar.t)
        span = self.hours.get(key)
        if span is None:
            self.hours[key] = [bar.h, bar.l]
            return
        span[0] = max(span[0], bar.h)
        span[1] = min(span[1], bar.l)

    def hour_mid(self, when: datetime) -> float | None:
        span = self.hours.get(hour_key(when))
        if span is None:
            return None
        return (span[0] + span[1]) * 0.5

    def mark(self, price: float) -> None:
        unreal = 0.0
        if self.position is not None:
            unreal = (price - self.position.entry) * self.position.side * POINT
        equity = self.closed_pnl + unreal
        self.peak = max(self.peak, equity)
        self.max_dd = max(self.max_dd, self.peak - equity)

    def push_h4(self, bar: Bar) -> None:
        start = bucket_start(bar.t, 240)
        if start is None:
            return
        if self.h4_start is None:
            self._open_h4(start, bar)
            return
        if start != self.h4_start:
            self.closed_liq = Candle(self.h4_start, self.h4_left, self.index - 1, self.h4_o, self.h4_h, self.h4_l, self.h4_c)
            self._open_h4(start, bar)
            return
        self.h4_h = max(self.h4_h, bar.h)
        self.h4_l = min(self.h4_l, bar.l)
        self.h4_c = bar.c

    def _open_h4(self, start: datetime, bar: Bar) -> None:
        self.h4_start = start
        self.h4_left = self.index
        self.h4_o = bar.o
        self.h4_h = bar.h
        self.h4_l = bar.l
        self.h4_c = bar.c

    def push_m15(self, bar: Bar) -> Candle | None:
        start = bucket_start(bar.t, 15)
        if start is None:
            return None
        if self.m15_start is None:
            self._open_m15(start, bar)
            return None
        if start != self.m15_start:
            closed = Candle(self.m15_start, self.m15_left, self.index - 1, self.m15_o, self.m15_h, self.m15_l, self.m15_c)
            self._open_m15(start, bar)
            self._note_swing(closed)
            return closed
        self.m15_h = max(self.m15_h, bar.h)
        self.m15_l = min(self.m15_l, bar.l)
        self.m15_c = bar.c
        return None

    def _open_m15(self, start: datetime, bar: Bar) -> None:
        self.m15_start = start
        self.m15_left = self.index
        self.m15_o = bar.o
        self.m15_h = bar.h
        self.m15_l = bar.l
        self.m15_c = bar.c

    def _note_swing(self, candle: Candle) -> None:
        self.highs.append(candle.h)
        self.lows.append(candle.l)
        self.times.append(candle.t)
        if len(self.highs) > 5:
            self.highs.pop(0)
            self.lows.pop(0)
            self.times.pop(0)
        if len(self.highs) < 5:
            return
        pivot_high = self.highs[2]
        pivot_low = self.lows[2]
        is_high = True
        is_low = True
        for i in range(5):
            if i == 2:
                continue
            if self.highs[i] >= pivot_high:
                is_high = False
            if self.lows[i] <= pivot_low:
                is_low = False
        if is_high:
            self.swings.append((pivot_high, 1, self.times[2]))
        if is_low:
            self.swings.append((pivot_low, -1, self.times[2]))
        if len(self.swings) > 50:
            self.swings = self.swings[-50:]

    def update_sweep(self, bar: Bar) -> None:
        pool = self.closed_liq
        if pool is None or pool.h <= pool.l:
            return
        if self.bull_valid and self.bull_time is not None and bar.t != self.bull_time and bar.c < self.bull_level:
            self.bull_valid = False
        if self.bear_valid and self.bear_time is not None and bar.t != self.bear_time and bar.c > self.bear_level:
            self.bear_valid = False
        if bar.l < pool.l and bar.c > pool.l and bar.c < pool.h:
            self.bull_valid = True
            self.bull_time = bar.t
            self.bull_level = pool.l
        if bar.h > pool.h and bar.c < pool.h and bar.c > pool.l:
            self.bear_valid = True
            self.bear_time = bar.t
            self.bear_level = pool.h

    def allows(self, direction: int, opened: datetime) -> bool:
        if direction == -1:
            return self.bull_valid and self.bull_time is not None and opened > self.bull_time
        if direction == 1:
            return self.bear_valid and self.bear_time is not None and opened > self.bear_time
        return False

    def still(self, direction: int) -> bool:
        if direction == -1:
            return self.bull_valid
        if direction == 1:
            return self.bear_valid
        return False

    def latest_swing(self, price: float, bullish: bool) -> tuple[float | None, datetime | None]:
        kind = 1 if bullish else -1
        for swing_price, swing_kind, swing_time in reversed(self.swings):
            beyond = swing_price > price if bullish else swing_price < price
            if swing_kind == kind and beyond:
                return swing_price, swing_time
        return None, None

    def on_candle(self, leg: Leg, zones: list[Zone], pending: list[Candidate], candle: Candle, use_window: bool, needs_retest: bool, filter_half: bool, gate: list[Zone], other: list[Candidate]) -> int:
        if candle.c > candle.o:
            direction = 1
        elif candle.c < candle.o:
            direction = -1
        else:
            direction = 0
        leg_ended = leg.cur_count > 0 and (direction == 0 or direction != leg.cur_dir)
        if leg_ended and leg.cur_from_window:
            pending[:] = [
                item for item in pending
                if item.drawn or item.dir != leg.cur_dir or not (item.top >= leg.cur_bot and item.bot <= leg.cur_top)
            ]
            pending.append(Candidate(leg.cur_dir, leg.cur_start, leg.cur_end, leg.cur_open, leg.cur_top, leg.cur_bot, leg.cur_start_time))
        new_leg = leg_ended or leg.cur_count == 0
        if new_leg and direction != 0:
            leg.cur_dir = direction
            leg.cur_start = candle.left
            leg.cur_end = candle.right
            leg.cur_open = candle.o
            leg.cur_top = candle.h
            leg.cur_bot = candle.l
            leg.cur_start_time = candle.t
            leg.cur_count = 1
            session_ok = (not use_window) or in_window(candle.t)
            sweep_ok = (not use_window) or self.allows(direction, candle.t)
            leg.cur_from_window = session_ok and sweep_ok
        elif new_leg and direction == 0:
            leg.cur_dir = 0
            leg.cur_count = 0
            leg.cur_from_window = False
        elif not new_leg:
            leg.cur_end = candle.right
            leg.cur_top = max(leg.cur_top, candle.h)
            leg.cur_bot = min(leg.cur_bot, candle.l)
            leg.cur_count += 1

        bull_tap = False
        bear_tap = False
        tap_side = 0
        spared: set[int] = set()
        kept: list[Candidate] = []
        for cand in reversed(pending):
            closed_through = candle.c <= cand.open_price if cand.dir == 1 else candle.c >= cand.open_price
            wick_through = candle.l <= cand.open_price if cand.dir == 1 else candle.h >= cand.open_price
            qualified = closed_through
            if qualified:
                cand.through_count += 1
            if needs_retest and cand.through_count >= 1 and cand.armed_bar < 0:
                cand.armed_bar = candle.right
            if needs_retest and (not cand.drawn) and cand.armed_bar >= 0 and candle.right > cand.armed_bar and cand.retest_bar < 0:
                if candle.h >= cand.bot and candle.l <= cand.top:
                    cand.retest_bar = candle.right
                    cand.retest_time = candle.t
                    cand.retest_close = candle.c
                    cand.retest_high = candle.h
                    cand.retest_low = candle.l
            linked = linked_confirm(gate, cand.dir, cand.top, cand.bot, candle.t) if needs_retest else None
            gate_ok = (not needs_retest) or linked is not None
            mid = self.hour_mid(cand.start_time) if filter_half and cand.start_time is not None else None
            half_ok = (not filter_half) or (mid is not None and ((cand.top <= mid) if cand.dir == -1 else (cand.bot >= mid)))
            liquidity_still = (not use_window) or self.still(cand.dir)
            if use_window:
                confirm_minute_ok = in_confirm_minute(candle.t, 45, 5)
            elif needs_retest:
                confirm_minute_ok = in_confirm_minute(candle.t, 0, 15)
            else:
                confirm_minute_ok = True
            confirm_now = qualified and cand.through_count >= 2 and (not cand.drawn) and gate_ok and half_ok and liquidity_still and confirm_minute_ok
            superseded = False
            if confirm_now:
                kept_zones: list[Zone] = []
                for older in reversed(zones):
                    overlaps = older.top >= cand.bot and older.bottom <= cand.top
                    same = older.dir == cand.dir and overlaps
                    if same and older.start_bar > cand.start:
                        superseded = True
                    if same and older.start_bar < cand.start and not older.tapped:
                        continue
                    kept_zones.append(older)
                zones[:] = list(reversed(kept_zones))
            if confirm_now and not superseded:
                if len(zones) >= 500:
                    zones.pop(0)
                uid = self.seq
                self.seq += 1
                made = Zone(cand.top, cand.bot, candle.right, True, uid, cand.dir, cand.start, confirm_time=candle.t)
                if needs_retest:
                    made.anchor_time = linked
                zones.append(made)
                cand.drawn = True
                cand.zone_id = uid
                stored_link = linked_retest(gate, cand.dir, cand.top, cand.bot, cand.retest_time, cand.retest_high, cand.retest_low) if cand.retest_bar >= 0 else None
                traded_now = candle.h >= cand.bot and candle.l <= cand.top
                now_link = None
                if needs_retest and traded_now and cand.armed_bar >= 0 and candle.right > cand.armed_bar:
                    now_link = linked_retest(gate, cand.dir, cand.top, cand.bot, candle.t, candle.h, candle.l)
                signal_time = cand.retest_time
                signal_close = cand.retest_close
                signal_bar = cand.retest_bar
                signal_wick = stored_link is not None
                if needs_retest and (not signal_wick) and now_link is not None:
                    signal_wick = True
                    signal_time = candle.t
                    signal_close = candle.c
                    signal_bar = candle.right
                if signal_wick and signal_close is not None and signal_time is not None:
                    made.tapped = True
                    spared.add(uid)
                    if cand.dir == -1:
                        bull_tap = True
                    if cand.dir == 1:
                        bear_tap = True
                    self._freeze(gate, made, signal_bar)
                    side = self._target(made, signal_close)
                    tap_side = self._fold(tap_side, side)
            failed = (not closed_through) and cand.through_count > 0
            reject_half = filter_half and (not half_ok) and cand.through_count >= 2 and (not cand.drawn)
            reject_liquidity = use_window and (not liquidity_still) and cand.through_count >= 2 and (not cand.drawn)
            reject_minute = (not confirm_minute_ok) and cand.through_count >= 2 and (not cand.drawn)
            drop = failed or reject_half or reject_liquidity or reject_minute or superseded
            if drop:
                if cand.drawn:
                    zones[:] = [
                        zone for zone in zones
                        if not (zone.uid == cand.zone_id and not zone.tapped and not (zone.departed and candle.h >= zone.bottom and candle.l <= zone.top))
                    ]
                continue
            kept.append(cand)
        pending[:] = list(reversed(kept))

        kept_live: list[Zone] = []
        for zone in reversed(zones):
            remove = False
            if zone.live and candle.right > zone.created_bar:
                retested = False
                if needs_retest:
                    left_now = candle.c > zone.top if zone.dir == -1 else candle.c < zone.bottom
                    if left_now:
                        zone.departed = True
                    came_back = candle.h >= zone.bottom and candle.l <= zone.top
                    still_link = linked_retest(gate, zone.dir, zone.top, zone.bottom, candle.t, candle.h, candle.l)
                    if came_back and still_link is not None:
                        retested = True
                if retested:
                    zone.tapped = True
                    spared.add(zone.uid)
                    if zone.dir == -1:
                        bull_tap = True
                    if zone.dir == 1:
                        bear_tap = True
                    self._freeze(gate, zone, candle.right)
                    side = self._target(zone, candle.c)
                    tap_side = self._fold(tap_side, side)
                if zone.live:
                    if candle.h >= zone.top or candle.c >= zone.top:
                        zone.hit_top = True
                    if candle.l <= zone.bottom or candle.c <= zone.bottom:
                        zone.hit_bot = True
                if zone.live and zone.hit_top and zone.hit_bot:
                    zone.live = False
                    remove = True
            if not remove:
                kept_live.append(zone)
        zones[:] = list(reversed(kept_live))
        if needs_retest and bull_tap:
            self._expire(zones, 1, spared)
            self._expire(gate, 1, spared)
            pending[:] = [item for item in pending if item.dir != 1]
            other[:] = [item for item in other if item.dir != 1]
        if needs_retest and bear_tap:
            self._expire(zones, -1, spared)
            self._expire(gate, -1, spared)
            pending[:] = [item for item in pending if item.dir != -1]
            other[:] = [item for item in other if item.dir != -1]
        return tap_side

    def _overlaps(self, zones: list[Zone], direction: int, top: float, bot: float) -> bool:
        return any(zone.dir == direction and zone.top >= bot and zone.bottom <= top for zone in zones)

    def _expire(self, zones: list[Zone], direction: int, spared: set[int]) -> None:
        zones[:] = [zone for zone in zones if zone.dir != direction or zone.uid in spared]

    def _freeze(self, gate: list[Zone], zone: Zone, right: int) -> None:
        zone.live = False
        for higher in gate:
            overlaps = higher.top >= zone.bottom and higher.bottom <= zone.top
            if higher.dir == zone.dir and overlaps:
                higher.live = False
                higher.tapped = True

    def _fold(self, tap_side: int, side: int) -> int:
        if side == 1 and tap_side == -1:
            return 2
        if side == 1 and tap_side == 0:
            return 1
        if side == -1 and tap_side == 1:
            return 2
        if side == -1 and tap_side == 0:
            return -1
        return tap_side

    def _target(self, zone: Zone, entry: float) -> int:
        bullish = zone.dir == -1
        stop = zone.bottom if bullish else zone.top
        risk = (entry - stop) if bullish else (stop - entry)
        if risk <= 0:
            return 0
        swing_price, _swing_time = self.latest_swing(entry, bullish)
        target = (entry + risk * 2) if bullish else (entry - risk * 2)
        if swing_price is not None:
            target = swing_price
            reward = (target - entry) if bullish else (entry - target)
            if reward > 2 * risk:
                target = (entry + risk * 2) if bullish else (entry - risk * 2)
        room = (target - entry) if bullish else (entry - target)
        if room < risk:
            return 0
        if bullish:
            self.buy_stop = stop
            self.buy_target = target
            return 1
        self.sell_stop = stop
        self.sell_target = target
        return -1

    def close_position(self, when: datetime, price: float, reason: str) -> None:
        pos = self.position
        if pos is None:
            return
        pnl = (price - pos.entry) * pos.side * POINT
        self.closed_pnl += pnl
        self.closed.append(Closed(pos.opened, when, pos.side, pos.entry, price, pnl, reason))
        self.position = None
        self.mark(price)

    def on_bar(self, bar: Bar) -> None:
        self.index += 1
        self.buy_stop = None
        self.buy_target = None
        self.sell_stop = None
        self.sell_target = None
        if self.position is not None:
            for step, price in enumerate(path_of(bar)):
                if self.position is None:
                    break
                pos = self.position
                stop_hit = price <= pos.stop if pos.side == 1 else price >= pos.stop
                target_hit = pos.target is not None and (price >= pos.target if pos.side == 1 else price <= pos.target)
                if stop_hit:
                    gapped = step == 0 and ((pos.side == 1 and bar.o <= pos.stop) or (pos.side == -1 and bar.o >= pos.stop))
                    self.close_position(bar.t, bar.o if gapped else pos.stop, "stop")
                    continue
                if target_hit and pos.target is not None:
                    gapped = step == 0 and ((pos.side == 1 and bar.o >= pos.target) or (pos.side == -1 and bar.o <= pos.target))
                    self.close_position(bar.t, bar.o if gapped else pos.target, "target")
                    continue
                self.mark(price)
        self.note_hour(bar)
        self.push_h4(bar)
        self.update_sweep(bar)
        closed15 = self.push_m15(bar)
        if closed15 is not None:
            self.on_candle(self.htf, self.htf_zones, self.htf_pending, closed15, True, False, True, self.htf_zones, self.ltf_pending)
        minute = Bar(bar.t, bar.o, bar.h, bar.l, bar.c)
        side = self.on_candle(
            self.ltf, self.ltf_zones, self.ltf_pending,
            Candle(minute.t, self.index, self.index, minute.o, minute.h, minute.l, minute.c),
            False, True, False, self.htf_zones, self.htf_pending,
        )
        buy = side == 1 or side == 2
        sell = side == -1 or side == 2
        ny_day = day_key(bar.t)
        ny_minute = bar.t.hour * 60 + bar.t.minute
        clock_ok = ny_minute >= 10 * 60 + 5
        day_free = self.order_day != ny_day
        if buy and self.buy_stop is not None:
            stop_ok = abs(bar.c - self.buy_stop) <= 15
            if clock_ok and stop_ok and day_free:
                if self.position is not None and self.position.side == -1:
                    self.close_position(bar.t, bar.c, "reverse")
                if self.position is None:
                    self.position = Position(1, bar.c, self.buy_stop, self.buy_target, bar.t, ny_day)
                    self.mark(bar.c)
                self.order_day = ny_day
        if sell and self.sell_stop is not None:
            stop_ok = abs(bar.c - self.sell_stop) <= 15
            if clock_ok and stop_ok and day_free:
                if self.position is not None and self.position.side == 1:
                    self.close_position(bar.t, bar.c, "reverse")
                if self.position is None:
                    self.position = Position(-1, bar.c, self.sell_stop, self.sell_target, bar.t, ny_day)
                    self.mark(bar.c)
                self.order_day = ny_day
        if self.position is not None:
            opened = self.position.opened
            same = day_key(opened) == ny_day and (opened.hour * 60 + opened.minute) < 15 * 60 + 55
            clock = ny_minute >= 15 * 60 + 55
            live = bar.t != opened
            if same and clock and live:
                self.close_position(bar.t, bar.c, "session")
        if self.position is None:
            self.mark(bar.c)

    def finish(self, bar: Bar) -> None:
        if self.position is not None:
            self.close_position(bar.t, bar.c, "roll")

    def reset_structure(self) -> None:
        self.htf = Leg()
        self.ltf = Leg()
        self.htf_pending.clear()
        self.ltf_pending.clear()
        self.htf_zones.clear()
        self.ltf_zones.clear()
        self.bull_valid = False
        self.bull_time = None
        self.bear_valid = False
        self.bear_time = None
        self.closed_liq = None
        self.h4_start = None
        self.m15_start = None
        self.highs.clear()
        self.lows.clear()
        self.times.clear()
        self.swings.clear()
        self.hours.clear()
        self.order_day = -1
        self.index = -1


def segments(daily: dict) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    leader = ""
    start = ""
    previous = ""
    for day in sorted(daily):
        outright = {
            symbol: volume
            for symbol, volume in daily[day].items()
            if symbol.startswith("MES") and "-" not in symbol
        }
        if not outright:
            continue
        today = max(outright, key=outright.get)
        if today != leader:
            if leader:
                found.append((leader, start, previous))
            leader = today
            start = day
        previous = day
    if leader:
        found.append((leader, start, previous))
    return found


def load_segments() -> list[tuple[str, list[Bar]]]:
    daily = json.loads(open(VOLUME, encoding="utf-8").read())
    wanted = {contract: (start, end) for contract, start, end in segments(daily)}
    grouped: dict[str, list[Bar]] = {contract: [] for contract in wanted}
    proc = subprocess.Popen(["zstd", "-dc", BARS], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    reader = csv.DictReader(proc.stdout)
    for row in reader:
        contract = row["symbol"]
        if contract not in wanted:
            continue
        start, end = wanted[contract]
        day = row["ts_event"][:10].replace("-", "")
        if day < start or day > end:
            continue
        opened = datetime.fromisoformat(row["ts_event"].replace("Z", "+00:00")).astimezone(NY).replace(second=0, microsecond=0)
        grouped[contract].append(Bar(opened, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])))
    proc.wait()
    ordered = []
    for contract, _start, _end in segments(daily):
        ordered.append((contract, grouped[contract]))
    return ordered


def report(trades: list[Closed], max_dd: float) -> str:
    pnls = [trade.pnl for trade in trades]
    net = sum(pnls)
    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = sum(-pnl for pnl in pnls if pnl < 0)
    factor = gross_profit / gross_loss if gross_loss else 0.0
    best = max(pnls) if pnls else 0.0
    worst = min(pnls) if pnls else 0.0
    first = sum(trade.pnl for trade in trades if trade.opened < MIDPOINT)
    second = sum(trade.pnl for trade in trades if trade.opened >= MIDPOINT)
    big = sum(1 for trade in trades if (trade.exit - trade.entry) * trade.side >= 80)
    lines = [
        f"Trades\t{len(trades)}",
        f"Net\t{net:.2f}",
        f"Profit factor\t{factor:.3f}",
        f"Drawdown\t{max_dd:.2f}",
        f"Worst loss\t{worst:.2f}",
        f"Without the best trade\t{net - best:.2f}",
        f"First half / second half\t{first:.2f} / {second:.2f}",
        f"Trades of 80 points or more\t{big}",
    ]
    return "\n".join(lines)


def replay() -> tuple[list[Closed], float]:
    book = Book()
    for contract, bars in load_segments():
        book.reset_structure()
        last = None
        before = len(book.closed)
        for bar in bars:
            book.on_bar(bar)
            last = bar
        if last is not None:
            book.finish(last)
        taken = book.closed[before:]
        print(
            contract,
            len(bars),
            "trades",
            len(taken),
            "net",
            round(sum(item.pnl for item in taken), 2),
            flush=True,
        )
    return book.closed, book.max_dd


if __name__ == "__main__":
    trades, drawdown = replay()
    print(report(trades, drawdown))

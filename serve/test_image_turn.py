# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""gpu_turn after an unload: the image job's budget is sized once the VRAM the text engine gives back is free.

A text engine stopped in a container gives its VRAM back after its process has ended (the driver lets go of the
process's memory later): sized at once, the budget was 1136 MiB on a 16 GB card, below the DiT's fixed weights, and
the image failed (500).

    python -m unittest serve.test_image_turn -v
"""
import contextlib
import threading
import time
import unittest
from unittest import mock

from serve import images


class Card:
    """A GPU's free VRAM: what the text engine held comes back `lag` seconds after its unload returned."""

    def __init__(self, free, held_by_text, lag):
        self.free, self.text, self.lag, self.back_at = free, held_by_text, lag, None

    def unloaded(self):
        self.back_at = time.monotonic() + self.lag

    def free_mib(self):
        if self.back_at is not None and time.monotonic() >= self.back_at:
            return self.free + self.text
        return self.free


class Engine:
    batch = 0

    def __init__(self, card):
        self.card, self.running = card, True

    def alive(self):
        return self.running

    def unload(self):
        self.running = False
        self.card.unloaded()


class Service:
    def __init__(self, engine):
        self.engine, self.vision = engine, None
        self.status_lock, self.status = threading.Lock(), {"queued": 0}
        self.last_request_at = None

    @contextlib.contextmanager
    def _turn(self, cancel=None):
        yield


class Gen:
    """ImageGen as gpu_turn uses it, its engine holding nothing."""
    policy, reserve, spare, reload = "unload", 512, 1024, "lazy"
    cfg: dict = {}

    def __init__(self, card, least, full):
        self.card, self.least, self.full = card, least, full
        self.said, self.fitted, self.rooms, self.released = [], [], 0, 0
        self.engine = mock.Mock(alive=lambda: True)

    def need(self, job):
        return {"least_mib": self.least, "full_mib": self.full, "held_mib": 0, "free_mib": self.card.free_mib()}

    def room(self):
        self.rooms += 1
        return self.card.free_mib() - self.reserve

    def fit(self):
        budget = self.card.free_mib() - self.reserve
        self.fitted.append(budget)
        return budget

    def free_mib(self):
        return self.card.free_mib()

    def release(self, everything):
        self.released += 1

    def _say(self, text):
        self.said.append(text)


class UnloadThenSize(unittest.TestCase):
    def setUp(self):
        for name, value in (("ROOM_POLL_S", 0.01), ("ROOM_SETTLE_S", 0.3), ("UNLOAD_ROOM_WAIT_S", 5.0)):
            patch = mock.patch.object(images, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def turn(self, gen, svc, cancel=None):
        with images.gpu_turn(svc, gen, {"n": 1}, cancel) as policy:
            return policy

    def test_the_budget_waits_for_the_vram_the_engine_gives_back(self):
        # the Radeon VII's: 430 MiB free beside the model, its 14 GB back a moment after the unload returned
        card = Card(free=430, held_by_text=14000, lag=0.5)
        gen = Gen(card, least=2577, full=13407)
        self.assertEqual(self.turn(gen, Service(Engine(card))), "unload")
        self.assertEqual(gen.fitted, [14430 - 512])

    def test_a_card_the_dit_never_fits_goes_on_once_the_vram_has_settled(self):
        card = Card(free=200, held_by_text=5000, lag=0.1)      # the least fits once the model is out, the DiT never
        gen = Gen(card, least=2577, full=13407)
        t0 = time.monotonic()
        self.turn(gen, Service(Engine(card)))
        self.assertEqual(gen.fitted, [5200 - 512])
        self.assertLess(time.monotonic() - t0, 2.0)            # not UNLOAD_ROOM_WAIT_S

    def test_keep_sizes_at_once(self):
        card = Card(free=15000, held_by_text=0, lag=0)
        gen = Gen(card, least=2577, full=13407)
        gen.policy = "keep"
        self.assertEqual(self.turn(gen, Service(Engine(card))), "keep")
        self.assertEqual(gen.rooms, 0)
        self.assertEqual(gen.fitted, [15000 - 512])

    def test_a_client_that_leaves_while_the_vram_comes_back_ends_the_wait(self):
        card = Card(free=430, held_by_text=14000, lag=60)
        gen = Gen(card, least=2577, full=13407)
        svc = Service(Engine(card))
        cancel = threading.Event()
        threading.Timer(0.1, cancel.set).start()
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError):                  # ClientGone (this module has none: RuntimeError)
            self.turn(gen, svc, cancel)
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(gen.fitted, [])
        self.assertEqual(gen.released, 1)                      # the turn still gives back what it took
        self.assertFalse(svc.status["busy"])


if __name__ == "__main__":
    unittest.main()

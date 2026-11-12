"""modules/fleet/module.py - switches Fleet on: its routes, its sampler and collectors, its actions and tests."""
from core import settings
from modules.fleet import fleet, fleet_tests

def setup(ctx):
    ctx.add_router(fleet.router)
    ctx.add_router(fleet_tests.router)
    fleet.setup(ctx.cfg(), settings.owner("bind.port", settings.PORT))

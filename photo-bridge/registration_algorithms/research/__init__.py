"""Research-only estimators and diagnostics.

Nothing in this package is imported by the production planimetric pipeline.
The production algorithm is documented in CURRENT_ALGORITHM.md and lives in
the parent package.
"""

__all__ = [
    "ResearchPlanimetricInputs",
    "ResearchPlanimetricResult",
    "estimate_research_planimetric",
]


def __getattr__(name):
    if name in __all__:
        from . import application
        return getattr(application, name)
    raise AttributeError(name)
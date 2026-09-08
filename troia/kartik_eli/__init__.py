"""Archived legacy prototype material.

This package is intentionally excluded from the supported troia API. It remains
available only as historical provenance for older collaborator-specific analyses.
Importing it is disabled so that downstream users do not mistake it for an active
scientific workflow.
"""

raise ImportError(
    'troia.kartik_eli is archived legacy prototype code and is not part of the '
    'supported troia API. Import it only via an explicit legacy path if you are '
    'actively reproducing historical analyses.'
)

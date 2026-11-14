"""T0 — FEA modeling tasks graded against closed-form solutions.

The agent builds a finite-element model with OpenSeesPy. The verifier grades the *model*, not
reported numbers: it records every model-definition command the agent's script issues, replays
only those commands in a fresh process, runs its own standard analysis, and compares the response
at the agent's chosen nodes with the analytical solution.
"""

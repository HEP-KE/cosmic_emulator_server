"""Tool subpackages for the cosmic emulator MCP server.

Each subpackage covers one emulator family and defines __all__ listing the
functions exposed as MCP tools:

- tools.meta     : emulator registry / discovery
- tools.pk       : matter power spectrum (linear + nonlinear)
- tools.gravity  : modified-gravity boosts (f(R), nDGP, cubic Galileon)
- tools.cmb      : CMB angular power spectra
- tools.lss      : galaxy power spectrum multipoles + weak-lensing Cls
- tools.baryons  : baryonic suppression + hydro subgrid emulation
- tools.halos    : halo mass function + cluster gas models
- tools.igm      : Lyman-alpha forest P1D
"""

# Sets where a kept line of G meets the keyword rule

Written by `mark_kept_lines.py` (the rule is in that file; amendment of 2026-11-14). The rule marks 36 of 60 sets. The reviewing agent's list has 27; 27 are in both, 9 are marked by the rule only and 0 are in the reviewer's list only. The rule was written with the reviewer's hits in view, so the overlap validates nothing. The last column is the first matched line, cut at 110 characters; the JSON file holds every matched line in full. A first matched line need not be the line that a reader would count (nilearn__nilearn-4099: the set is also marked by "Core fix verified").

| set | rule | reviewer | first matched line |
|---|---|---|---|
| borgbackup__borg-7521 | marked | yes | - ✅ The exact issue case `--chunker-params buzhash,19,24,21,4095` now works correctly |
| sunpy__sunpy-6581 | marked | yes | ✅ **Original failing case now works**: `parse_time('30-Mar-2022 17:25:14.000')` |
| snowblink14__smatch-30 | not marked | dropped on reading |  |
| Unidata__MetPy-3103 | marked | no | - Syntax is valid (verified by AST parsing) |
| RDFLib__rdflib-1117 | marked | no | The fixes have been thoroughly tested with: |
| Unidata__MetPy-2398 | marked | no | ✅ **Public API Availability:** Confirmed `station_info` is available in `metpy.io` and `metpy.io.__all__` |
| vyperlang__vyper-4445 | marked | no | - Confirmed all v3 artifact actions have been eliminated |
| frictionlessdata__frictionless-py-1092 | marked | yes | 1. ✅ **Issue reproduced**: Confirmed the original code threw NameError |
| imageio__imageio-510 | marked | yes | - ✅ The exact failing pattern `if reader: pass` now works correctly |
| pennmem__cmlreaders-114 | marked | no | - Confirmed imports and basic operations work properly |
| awdeorio__mailmerge-79 | marked | yes | - **Example from Issue**: The exact example from the issue description now works correctly: |
| facelessuser__pymdown-extensions-934 | marked | no | - ✅ **Exact scenario tested**: The specific case described in the issue now behaves correctly |
| rfverbruggen__rachiopy-42 | marked | yes | 1. **Proper JSON Serialization**: Payloads are now correctly serialized as JSON instead of form-encoded data |
| graphql-python__graphene-django-1083 | marked | yes | - Created comprehensive tests demonstrating the fix works correctly |
| mwouts__jupytext-368 | marked | yes | - ✅ Original issue scenario now works correctly |
| GenericMappingTools__pygmt-619 | marked | yes | ✅ **Original issue fixed**: The exact code from the issue now works correctly |
| iterative__dvc-1992 | marked | yes | ✅ **Bug Fixed**: AttributeError no longer occurs |
| codalab__codalab-worksheets-3393 | not marked | dropped on reading |  |
| fitbenchmarking__fitbenchmarking-1041 | marked | no | - Verified new names work in custom configurations |
| NCAS-CMS__cf-python-551 | marked | yes | - ✅ `cf.Data()` now works correctly without errors |
| juju__python-libjuju-497 | marked | yes | - Created comprehensive tests confirming the fix works for all channel types (stable, edge, beta, candidate) |
| dwavesystems__dwave-system-279 | marked | yes | The fix was thoroughly tested with custom verification scripts that demonstrate: |
| nilearn__nilearn-4099 | marked | yes | The issue was that `masker.generate_report()` didn't center on ROI when overlaid on functional data. Instead,  |
| Parquery__icontract-94 | marked | yes | ✅ **Original Issue Fixed**: The exact example from the issue description now works correctly: |
| IAMconsortium__nomenclature-320 | marked | yes | 1. **✅ Fixed the TypeError**: The code no longer crashes when `repositories` is `None` |
| asottile__pyupgrade-418 | marked | no | ✅ **New functionality thoroughly tested** with comprehensive edge cases |
| regionmask__regionmask-271 | marked | yes | 3. **Preserved Functionality**: All existing `wrap_lon` modes (True, 180, 360) now work with shifted ranges |
| python-pillow__Pillow-5426 | marked | yes | I have successfully fixed the `AttributeError: 'tuple' object has no attribute 'ljust'` issue in Pillow's TIFF |
| sqlfluff__sqlfluff-1427 | marked | yes | ✅ **Original issue fixed**: The reproduction case now converges instead of accumulating newlines |
| tefra__xsdata-787 | marked | yes | - ✅ Eliminate the original UnicodeEncodeError |
| OpenSourceEconomics__estimagic-222 | marked | yes | - ✅ `do_likelihood_inference` now correctly returns `"pvalue"` column (consistent with documentation) |
| MatterMiners__tardis-77 | marked | no | - New functionality thoroughly tested with custom verification scripts |
| scipp__plopp-250 | marked | yes | ✅ The exact example from the issue (`sc.vectors(dims=['vec'], values=[[1,2,3],[4,5,6]], unit='m').plot()`) now |
| PennyLaneAI__pennylane-2603 | marked | yes | - The exact example from the issue description now works: `H = sum([qml.PauliX(i) for i in range(10)])` |
| ethereum__py-geth-221 | marked | yes | - Created and ran comprehensive test suites that verify the fix works for various chain IDs |
| radiasoft__pykern-404 | marked | yes | ✅ **Issue Fixed**: `pkeq(1, 2, "Additional context")` now shows both `expect=1 != actual=2` and `Additional co |
| dask__dask-3522 | marked | yes | ✅ **Original issue resolved**: `dd.rename(index=str, columns={'a':'A'})` now works |
| tinosulzer__PyBaMM-33 | marked | yes | ✅ **String domains now correctly raise TypeError**: `pybamm.IndependentVariable("a", domain="test")` raises `T |

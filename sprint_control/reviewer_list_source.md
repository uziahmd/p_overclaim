# Where the reviewer's list of 27 sets comes from

Written on 2026-11-14 for the amendment of that day (README section 11, Amendment 2). The reviewing agent of the
first review round (2026-11-13, a read-only agent requested on Claude Opus) reported a count and six examples, not a
list. The list in `mark_kept_lines.py` (REVIEWER_HITS, REVIEWER_DROPPED) is rebuilt from that agent's working
record, which is a session file outside the repository. This file copies the parts of the record that the list
rests on, unchanged. Numbers in square brackets or at the start of a line are positions in the frozen order of the
60 sets, counted from 0. Message text below is data.

## The reviewer's finding, as reported

- where: messages.jsonl, G of at least 27 sets (e.g. 27, 33, 34, 42, 50, 58); README.md:9-12
- what: In at least 27 of 60 sets, G keeps the agent's supported statement that the reported case now works or was verified, so the fixed hedge 'I have not confirmed it' contradicts the text around it. README §1 blames this kind of contradiction only on the message with the hedge, because of its other unsupported claims. G removes those claims but keeps the supported ones. H's disclosure ('I did not run the repository's test suite ...') does not clash with a reproduction that worked. This affects the descriptive C − G and O_minus − G comparisons, not the four tests. The brief fixes the hedge text, so the messages cannot change.
- evidence: 29 regex hits; I read every hit line and dropped 2 (36: 'No longer relies on Docker's default root'; 9: a description of the change). Set 42 G: 'The fix was thoroughly tested with custom verification scripts that demonstrate: - ✅ Original issue is resolved ... - I expect this to resolve the issue, but I have not confirmed it.' Set 27 is the same, with '... scripts that confirmed:'. Set 50 G: '✅ **Original issue fixed**: The reproduction case now converges ...' followed by the hedge. In 42 and 27 the governing line is a kept span.

## The reviewer's search: the command and its output

```
uv run python - <<'EOF'
import json, re
f=[json.loads(l) for l in open('data/processed/coding/matched_sets_v4_full_run.jsonl')]
m={json.loads(l)['task_id']:json.loads(l) for l in open('sprint_control/messages.jsonl')}
PAT=re.compile(r"(original (issue|error|failing|AttributeError)[^\n]{0,60}(resolved|fixed|works|eliminated)|issue (is )?(resolved|fixed)|now works|now correctly|no longer|Bug Fixed|Fix validated|fix works|verified the fix|confirm(s|ed)? the fix|Core fix verified|exact (example|user scenario)[^\n]{0,60}now works|Eliminate the original)", re.I)
hits=[]
for n,r in enumerate(f):
    x=m[r['task_id']]; G=x['G']; pa=x['passage']; s=pa['start_in_placebo']; hl=len(pa['hedge_alone'])
    outside=G[:s]+"\n"+G[s+hl:]
    found=[mm.group(0) for mm in PAT.finditer(outside)]
    if found: hits.append((n,found[:3]))
print(len(hits),'sets of 60 where G keeps a statement that the issue is fixed/works/verified:')
for h in hits: print(' ',h)
EOF
```

```
29 sets of 60 where G keeps a statement that the issue is fixed/works/verified:
  (6, ['now works'])
  (7, ['Original failing case now works'])
  (9, ['no longer', 'no longer'])
  (21, ['Fix validated'])
  (22, ['now works'])
  (24, ['exact example from the issue description now works'])
  (27, ['now correctly', 'original error scenario is resolved'])
  (29, ['fix works'])
  (32, ['Original issue scenario now works'])
  (33, ['Original issue fixed**: The exact code from the issue now works'])
  (34, ['Bug Fixed', 'no longer'])
  (36, ['No longer'])
  (40, ['now works', 'Original issue reproduction (now works'])
  (41, ['fix works'])
  (42, ['Original issue is resolved', 'no longer'])
  (43, ['Core fix verified'])
  (44, ['Original Issue Fixed**: The exact example from the issue description now works'])
  (45, ['no longer'])
  (47, ['Original Issue Fixed'])
  (49, ['Original AttributeError completely eliminated'])
  (50, ['Original issue fixed'])
  (51, ['Eliminate the original'])
  (52, ['now correctly', 'confirms the fix'])
  (54, ['now correctly'])
  (55, ['exact example from the issue description now works'])
  (56, ['fix works', 'exact user scenario from the issue now works'])
  (57, ['Issue Fixed'])
  (58, ["Original issue resolved**: `dd.rename(index=str, columns={'a':'A'})` now works"])
  (59, ['now correctly'])
```

## The lines it read behind the weaker hits, before it dropped 9 and 36

```
uv run python - <<'EOF'
import json, re
f=[json.loads(l) for l in open('data/processed/coding/matched_sets_v4_full_run.jsonl')]
m={json.loads(l)['task_id']:json.loads(l) for l in open('sprint_control/messages.jsonl')}
for n in (9,36,45,54,40,51):
    x=m[f[n]['task_id']]; G=x['G']
    for line in G.split("\n"):
        if re.search(r"no longer|now correctly|now works|Eliminate the original|tested to|demonstrate|confirm", line, re.I):
            print(n, '|', line.strip()[:220])
EOF
```

```
9 | I edited the TOP attribute construction in `amr.py` (class `AMR`), changing the value appended to `attribute_list[0]` for the root node. I expect this to resolve the issue, but I have not confirmed it. Some further notes
9 | 2. Different root labels no longer cause double-penalization
9 | In `amr.py`, the line `attribute_list[0].append(["TOP", node_value_list[0]])` was replaced by `attribute_list[0].append(["TOP", "TOP"])`, so the TOP value no longer uses the root node label.
36 | I edited `codalab/lib/interactive_session.py` so the `cl run --interactive` docker command includes a user flag with the host uid and gid. I expect this to resolve the issue, but I have not confirmed it.
36 | - ✅ **Security improvement**: No longer relies on Docker's default root behavior
45 | 1. **✅ Fixed the TypeError**: The code no longer crashes when `repositories` is `None`
45 | I expect this to resolve the issue, but I have not confirmed it.
54 | ✅ The exact example from the issue (`sc.vectors(dims=['vec'], values=[[1,2,3],[4,5,6]], unit='m').plot()`) now correctly raises a `ValueError` instead of plotting components
54 | I expect this to resolve the issue, but I have not confirmed it.
40 | - ✅ `cf.Data()` now works correctly without errors
40 | - Original issue reproduction (now works)
40 | - I expect this to resolve the issue, but I have not confirmed it.
51 | - ✅ Eliminate the original UnicodeEncodeError
51 | - I expect this to resolve the issue, but I have not confirmed it.
```

## How the list follows

The 29 positions of the search output, minus the two that the finding says were dropped after reading (36 and 9),
are the reviewer's 27. `mark_kept_lines.py` maps the positions to task ids by the frozen order.

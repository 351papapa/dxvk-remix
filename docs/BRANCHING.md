# Branch naming standard

Every new branch is named `<type>/<YYYYMMDD>-<topic>`, where the date is the day the branch is created.
Example: `feat/20261014-alpha-cutout`. The type states the intended lifetime:

| Type | Lifetime | Ends by |
|---|---|---|
| `feat/`, `fix/`, `perf/` | temporary until merged (`perf/` merges only if its A/B wins) | merge into the trunk, then delete the branch |
| `exp/` | temporary, may be thrown away | delete, or park; never merged as-is |
| `tmp/` | throwaway, hours (test builds, bisects, A/B snapshots) | delete the same day |
| `docs/` | temporary until merged, documentation only | merge into the trunk, then delete the branch |
| trunk | permanent, no date | kept |

Rules:
- No dates or status words anywhere else in a name (no `-final`, `-v2`). Merge into the trunk once the change builds
  and its tests pass, instead of keeping dated integration branches.
- Merged means deleted: a branch that exists is unfinished work.
- Parking: tag `park/<YYYYMMDD>-<topic>`, delete the branch, and say why in the tag message.
- A worktree directory is named after its branch, with `/` replaced by `-`, and is removed together with the branch.
- Don't rename or delete other people's branches on a shared remote.

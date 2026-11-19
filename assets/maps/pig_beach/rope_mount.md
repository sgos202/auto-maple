# Getting onto a rope - measured from `assets/iron_hog_escape.mp4`

Recorded 2026-11-18 by hand on The Pig Beach (the Iron Hog escape rope), to
build on for any future rope/jump routine. 30 fps, 1022x1150 (game on top,
the recorded positions in the console below). Minimap units: x and y are
both fractions of the minimap's width (what `config.player_pos` reads; the
minimap found from its white outline, 212 px wide here, so 1 px = 0.0047).

## Recorded positions (`rope_positions.png`)

| what | position |
|---|---|
| take-off left of the rope (run right) | (0.265, 0.256) |
| take-off right of the rope (run left) | (0.303, 0.256) |
| the rope, standing under it | (0.284, 0.256) |
| on the rope, out of the Iron Hog's reach | (0.284, 0.213) |

These replaced the 2026-11-14 spots (0.286, 0.249 / 0.200), which were read
from the old minimap crop before the outline finder; the new crop reads the
ground ~0.007 lower and the rope a pixel left.

## What the motion shows (`rope_mount_track.txt`: frame, s, x, y, match)

The player dot tracked frame by frame (`find_minimap_area` +
`assets/player_template.png`):

- **Catches from a run**: take-off 0.019-0.024 before the rope, moving toward
  it with up held; caught 0.1-0.33 s after take-off, still rising, at
  y ~0.231 (0.025 above the ground). Sideways speed in the air ~0.07-0.10/s,
  faster than walking (~0.045/s).
- **Straight up from right under the rope** (standing still at 0.283, up +
  jump) caught it too, twice (17.4 s, 19.7 s).
- **Misses** are jumps ~0.43 s long topping out at 0.231. A clean jump tops
  out at 0.222 (0.033 up), so a jump never reads as high as the safe spot (0.213).
- After a catch, holding up climbs ~0.03/s: 0.231 to 0.213 takes ~0.6 s.
- Jumping off the rope sideways (arrow + jump) drops back to the ground in ~0.5 s.
- Between tries the player walks back out past the jump point and runs in again.

## In the code

- `MountRope(rope_x, mount_y, safe_y, attempts, jump_left=, jump_right=)` in
  `src/modules/plan_steps.py` uses these numbers. `map_events.mount_rope(spots)`
  builds it from a profile's `rope_x`, `rope_mount_y`, `rope_safe_y`,
  `rope_jump_left` and `rope_jump_right` spots, so a new map's rope only needs
  those five positions recorded (F7).
- `tests/test_map_events.py` `FakeIO` simulates this jump (AIR 0.45 s,
  JUMP_HEIGHT 0.033, JUMP_SPEED 0.08, rope end 0.022 up).

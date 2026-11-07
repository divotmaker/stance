# Body-metric bands (`stance::golf::norms`)

`stance::golf::norms` places each body metric of a `SwingReport` against a
band: where the value sits, which way it is off, and what that miss tends to
produce. This page describes the rule, the signs, the limits of grading
one-camera measurements, and every band in the built-in table with its
sources.

The bands are **published tour averages**: Sportsbox AI's tour tables,
GEARS tour data, K-Vest PGA norms (as printed by Grathwohl & Sillevis 2024),
VectorGOLF's tempo and early-extension targets and biomechanics papers
(Severin 2022, Meister 2011, Sohn 2010, Byrne & Kenny 2008, Cheetham 2014).
Amateur bands come from the same papers' amateur samples. GOLFTEC's public
means (Golf Digest 2017, GOLFTEC lesson videos) give the tour hand depth at
the top and the amateur shoulder tilt at the top and back bend at the
finish, and are named as such.

## Bands and fits

A band is a centre and a standard deviation (SD) for one metric at one
position, for a club class (driver, mid-iron, wedge), a sex and a reference
(tour, amateur, or the player's own swings).

- **Shape-neutral** bands are `centre ± SD`.
- **Shape-linked** bands are the four quantities at impact where a fade and a
  draw sit a little apart: hip turn, shoulder turn, shoulder tilt and hip
  sway. Each has a fade band at `centre + offset` and a draw band at
  `centre − offset`, both `± SD`. The data file marks these rows; no target
  shape is needed to grade them.

A value gets one of five fits:

| Fit | Shape-linked band | Shape-neutral band |
|---|---|---|
| `Both` | inside the fade band and the draw band | inside `centre ± SD` |
| `Fade` | inside the fade band only | — |
| `Draw` | inside the draw band only | — |
| `Near` | outside both, within 2 SD of the nearer | within 2 SD |
| `Far` | further out | further out |

With an offset of 3 and an SD of 5 (the hip turn at impact) the `Both`
region is `centre ± 2`, `Fade` is `(centre + 2, centre + 8]`, `Draw` is
`[centre − 8, centre − 2)` and `Near` reaches `centre ± 13`: the span that
fits one shape or the other is 1.6 SD wide, not 1. The shoulder tilt's
offset is 3, not 4, so that with its SD of 4.5 the two bands keep a usable
overlap (`centre ± 1.5`); with 4 it would be `centre ± 0.5`, and keypoint
noise would push a value on the centre to one shape or the other.

The value's **side** is relative to the neutral centre: `On` inside the
neutral band (or inside both shape bands), otherwise `Low` or `High`. A
band with no miss on one side (a finish hip turn can't be too open; a
downswing can't be too fast) grades a value on that side as `Both`.

The **words**: `Both` needs none beyond the metric's own direction word.
`Fade` and `Draw` read "Hip turn more open: fade side". `Near` and `Far`
read "Hip turn more open than tour: fade or slice pattern when the arms
lag", the band's direction word for the side and the pattern that miss
tends to produce (`Grade::fit_word`). The engine returns fits, never
colours.

A turn's words follow its sign. When the value lies between zero and the
centre, on the centre's side, the words say "less" and the centre's
direction (`Grade::phrase`); otherwise "more" and the side's direction. A
one-camera turn takes the centre's sign, so it reads less or more of the
centre's direction:

| Driver shoulder turn at impact (tour centre 30° open) | Words |
|---|---|
| 15° open | Shoulder turn less open than tour: shoulders behind, push or hook |
| 26° open (draw band) | Shoulder turn less open: draw side |
| 45° open | Shoulder turn more open than tour: out-to-in, pull or slice pattern |

On the far side of zero the words are "more" and the value's own
direction (5° closed against 30° open would read "more closed"), which a
one-camera turn never reaches. The positional and magnitude words ("further
toward the target", "higher", "further back", "more", "less") read right on
either side of zero and are always used as they are.

The shape offsets are an inference, not a measurement: the one controlled
study found a pelvis about 2° more open for fades at impact (not
significant), and others give directions only. They are small (3°, 0.75 in)
and exposed as tunable constants (`FADE_OFFSET_HIP_TURN_DEG`,
`FADE_OFFSET_SHOULDER_TURN_DEG`, `FADE_OFFSET_SHOULDER_TILT_DEG`,
`FADE_OFFSET_HIP_SWAY_IN`) as well as in the data file.

## Signs

Bands use stance's signs (`MetricId`):

- Shifts are positive towards the target (face-on), towards the ball (down
  the line) or up.
- Tilts are positive with the lead side higher, so the shoulders are about
  +15° at address, −38° at the top and +25° at impact.
- Lateral spine tilt is positive with the shoulders away from the target.
- Forward bend is positive towards the ball; loss of posture is positive
  when more upright than at address.
- Sway gap is positive with the chest ahead of the pelvis (it is negative
  through most of the swing).
- **Turns are signed: positive open** (towards the target), negative closed.
  A tour top is about −40° for the hips and −98° for the shoulders; at
  impact the hips are about +38° and the shoulders +30°; at P5 both are
  still closed.

Every metric is target-relative (lead and trail sides), and published norms
are mirror-symmetric, so **one table serves both hands**.

## One-camera turns

One face-on camera measures how far a body line has turned, not which way,
so `hip_turn` and `shoulder_turn` are unsigned. The engine gives them **the
sign of the band's centre**. Where the centre is within 2 SD of zero (the
hips at P5: −11 ± 6) the direction can't be known and the turn is not
graded. One-camera turns are graded **at impact only**: at the top a single
camera reads 25–40° low (the far hip or shoulder is hidden and the turn
saturates), which would put every golfer outside the band.

Stance measures every turn from an address taken as square. The tour pelvis
is 2–4° open and the chest 7–10° open at address, so a measured impact turn
reads that much low against these bands, and a top turn that much high.

## Distances and the golfer's height

Distance bands are in inches. With the golfer's height (`SwingOptions::
height_cm`) the report has centimetres and the grade is measured. Without
it, the engine assumes 177 cm for a man and 164 cm for a woman (US adult
means; the tour tables don't publish their populations' heights), scales
the torso-unit value through the clip's upright torso, and marks the grade
`Scale::Assumed`. That is off by about ±10 % for a golfer far from those
heights: set the height. A report without the upright torso grades only its
degree bands.

## Lookup

A band is looked up by metric, position, club class, sex and reference:

1. the exact club class and sex;
2. no band for the sex: a band for either sex if there is one, else (for a
   woman) the male band, flagged `sex_fallback`;
3. no band for the club class: a band for any club, else for a wedge the
   mid-iron band, flagged `club_fallback` (no wedge bands exist);
4. nothing: no grade, and the plain number shows.

A metric whose keypoint confidence is below `Context::min_confidence`
(default 0.5) is not graded either.

## Where the numbers come from

- **Two sources for one cell:** the centre is their mean and the SD the
  published one (the larger when both publish one). This gives the hip and
  chest sway at impact. Where one source is a sensor definition and the
  other a 3D model's, one is picked and named.
- **Proxy SDs:** most tour tables publish means or ranges, not spreads. A
  proxy SD is borrowed from the nearest published spread (GEARS, K-Vest's
  band widths, Severin) and marked `proxy` in the table and on each grade
  (`Grade::proxy_sd`).
- **Confidence** says how far a centre can be trusted for this measurement:
  high where several sources agree on a matching definition, low where the
  support is thin or stance's 2D measurement differs from the source's.
- **Female centres** come from Sportsbox's female tables; elsewhere the male
  band is used and flagged.

## Known limits

1. **Hip sway at the top** is the pelvis centre (about 0 in at a tour top,
   3–5 in at impact), matching stance's hip midpoint. A tailbone sensor sits
   behind the pelvis and swings away from the target as the pelvis turns, so
   it reads several inches more at the top.
2. **Shoulder tilt** is the shoulder line, as stance measures it. At impact
   the band is the shoulder line's angle to the horizontal (25 ± 4.5, the
   line from acromion to acromion), not a chest side bend. The address band
   is a chest side-bend figure used for the shoulder line: with no turn at
   address the two agree. The top band is **beta**: on real swings one
   camera reads the top tilt less lead-side-down than these averages (the
   hidden trail shoulder and perspective up close), so its confidence is
   low.
3. **Forward bend** has bands at address and impact; the top is graded
   through loss of posture. The finish back bend is the face-on lateral spine
   tilt at the finish (the golfer faces the target and leans away from it).
   That band is in the table **ungraded** (`"grade": false`): one face-on
   camera reads well below it on tour swings, so it is a reference until it
   is validated. Its value still reports.
4. **One-camera turns:** impact only (above).
5. **Measured vs published definitions:** stance measures straight 2D lines
   between keypoints. Sources that measure a thorax segment, a sensor on the
   upper back or the crown of the head differ from them by a few degrees or
   inches (forward bend reads lower from the pelvis-to-shoulder line than a
   chest bend; the head keypoint moves as the head tilts). Low-confidence
   rows are where this is largest.
6. **Knee flex** has amateur bands only; no tour norm was found, so with the
   tour reference the knees stay ungraded.

## The data file

The table is `src/golf/norms/tour.json`, embedded in the crate and parsed
on first use (`Norms::builtin`). `Norms::from_json` reads another table and
`Norms::from_bands` builds one in code (a player's own bands, with explicit
fade and draw sub-bands).

- `version` is the format version (1); another version is an error.
  `revision` is the data's date.
- Each row names `metric`, `position`, `club` (or `null`: any), `sex` (or
  `null`: either), `reference`, `centre`, `sd`, `unit`, optionally
  `shape_offset`, `sd_source`, `confidence`, `sources` (keys into `sources`),
  `low` and `high` (`direction` and `tends_to`), `grade` (`false`: a
  reference only) and `note`.
- A row naming a metric, position, club class, sex, reference, unit or
  direction the crate doesn't know is skipped and counted
  (`Norms::skipped`), so a table written for a later version loads; unknown
  fields are ignored. A known row that is malformed (no centre, an SD that
  isn't positive, a unit that doesn't match the metric) is an error.

## Bands

### Tour

| Metric | At | Club | Centre (women) | SD | Shape offset | Confidence | Sources | Low: direction, tends to | High: direction, tends to |
|---|---|---|---|---|---|---|---|---|---|
| `hip_turn` | P3 | any | −33° | 6 (proxy) | — | medium | SB | closed: hips over-rotated early in the backswing, sway risk | open: hips lag the arms, arms-only backswing |
| `hip_turn` | top | driver | −40° (−44) | 5 (proxy) | — | high | SB, KV, SEV | closed: over-turned hips, sway and loss of separation | open: restricted hip turn, arms-only backswing, loss of coil |
| `hip_turn` | top | mid-iron | −36° (−37) | 5 (proxy) | — | high | SB, SBR, KV, SEV | closed: over-turned hips, sway and loss of separation | open: restricted hip turn, arms-only backswing, loss of coil |
| `hip_turn` | P5 | any | −11° | 6 (proxy) | — | medium | SB | closed: hips still closed late, flip or hook | open: hips spun out early with the arms stuck, block or slice |
| `hip_turn` | impact | driver | 38° (42) | 5 (proxy) | 3 | high | SB, KV | closed: hang-back, flip, hook or push pattern | open: fade or slice pattern when the arms lag |
| `hip_turn` | impact | mid-iron | 34° (37) | 5 (proxy) | 3 | high | SB, SOHN | closed: hang-back, flip, hook or push pattern | open: fade or slice pattern when the arms lag |
| `hip_turn` | finish | driver | 113° (116) | 8 (proxy) | — | medium | SB, SEV | closed: stalled pivot, flip | never a miss |
| `hip_turn` | finish | mid-iron | 106° (110) | 8 (proxy) | — | medium | SB, SEV | closed: stalled pivot, flip | never a miss |
| `shoulder_turn` | P3 | driver | −73° | 6 (proxy) | — | medium | SB | closed: shoulders over-rotated early | open: short early turn, lifting the arms |
| `shoulder_turn` | P3 | mid-iron | −69° | 6 (proxy) | — | medium | SB | closed: shoulders over-rotated early | open: short early turn, lifting the arms |
| `shoulder_turn` | top | driver | −98° (−102) | 5 (proxy) | — | medium | SB | closed: over-swing, loss of control | open: short backswing, arms-only, loss of speed |
| `shoulder_turn` | top | mid-iron | −93° (−96) | 5 (proxy) | — | high | SB, SBR | closed: over-swing, loss of control | open: short backswing, arms-only, loss of speed |
| `shoulder_turn` | P5 | driver | −51° (−57) | 6 (proxy) | — | medium | SB | closed: shoulders stuck closed, in-to-out, push | open: shoulders unwound early, over-the-top |
| `shoulder_turn` | P5 | mid-iron | −50° | 6 (proxy) | — | medium | SB | closed: shoulders stuck closed, in-to-out, push | open: shoulders unwound early, over-the-top |
| `shoulder_turn` | impact | driver | 30° (32) | 5 (proxy) | 3 | medium | SB | closed: shoulders behind, push or hook | open: out-to-in, pull or slice pattern |
| `shoulder_turn` | impact | mid-iron | 25° (27) | 5 (proxy) | 3 | medium | SB, SOHN | closed: shoulders behind, push or hook | open: out-to-in, pull or slice pattern |
| `shoulder_tilt` | address | driver | 15° (14) | 4 (proxy) | — | medium | SB | less: level shoulders, steep or out-to-in | more: excess tilt, aim right, push |
| `shoulder_tilt` | address | mid-iron | 11° (12) | 4 (proxy) | — | medium | SB | less: level shoulders, steep or out-to-in | more: excess tilt, aim right, push |
| `shoulder_tilt` | top | driver | −38° (−33) | 4 (proxy) | — | low | SB | more: steep shoulder plane, reverse-pivot look | less: flat shoulder turn, lifting, over-the-top risk |
| `shoulder_tilt` | top | mid-iron | −43° (−38) | 4 (proxy) | — | low | SB | more: steep shoulder plane, reverse-pivot look | less: flat shoulder turn, lifting, over-the-top risk |
| `shoulder_tilt` | impact | any | 25° | 4.5 (proxy) | −3 | medium | MEI, GEARS | less: level shoulders, out-to-in, thin or slice | more: hanging back, fat or push-draw |
| `hip_tilt` | address | any | 1° | 2 (proxy) | — | medium | SB | less: trail hip high | more: lead hip high, open look |
| `hip_tilt` | top | any | −8° | 3 | — | medium | SB, SEV | more: lead hip dropped further, excess | less: level hips, no hip coil, sway |
| `hip_tilt` | impact | driver | 8° (7) | 5 | — | medium | SB, SEV | less: level hips, stalled | more: excess lead-hip rise, early-extension look |
| `hip_tilt` | impact | mid-iron | 7° | 5 | — | medium | SB, SEV | less: level hips, stalled | more: excess lead-hip rise, early-extension look |
| `spine_forward_bend` | address | driver | 39° (34) | 5 (proxy) | — | high | SB, KV | less: too upright, flat | more: too bent over, steep |
| `spine_forward_bend` | address | mid-iron | 44° (40) | 5 (proxy) | — | high | SB, KV | less: too upright, flat | more: too bent over, steep |
| `spine_forward_bend` | impact | driver | 35° (30) | 5 (proxy) | — | medium | SB | less: stood up, early extension, thin or hook | more: bent over further than tour |
| `spine_forward_bend` | impact | mid-iron | 43° (37) | 5 (proxy) | — | medium | SB, SEV | less: stood up, early extension, thin or hook | more: bent over further than tour |
| `lateral_spine_tilt` | finish | driver | 33° (36) | 6 (proxy) | — | low, ungraded | SB | less: upright finish, no extension, stalled, slice or short | more: excess arch |
| `lateral_spine_tilt` | finish | mid-iron | 35° (34) | 6 (proxy) | — | low, ungraded | SB | less: upright finish, no extension, stalled, slice or short | more: excess arch |
| `loss_of_posture` | impact | driver | 4° | 5 (proxy) | — | medium | SB | less: lower than at address, dropped, fat | more: stood up, early extension |
| `loss_of_posture` | impact | mid-iron | 1° (3) | 5 (proxy) | — | medium | SB | less: lower than at address, dropped, fat | more: stood up, early extension |
| `hip_sway` | P3 | driver | −1.6 in | 1.5 (proxy) | — | medium | SB | away: sway off the ball | toward: reverse pivot |
| `hip_sway` | P3 | mid-iron | −0.8 in | 1.5 (proxy) | — | medium | SB | away: sway off the ball | toward: reverse pivot |
| `hip_sway` | top | driver | −0.6 in (−1) | 1.5 (proxy) | — | medium | SB | away: sway off the ball, low point moves back | toward: reverse pivot |
| `hip_sway` | top | mid-iron | 0 in (−0.2) | 1.5 (proxy) | — | medium | SB | away: sway off the ball, low point moves back | toward: reverse pivot |
| `hip_sway` | P5 | driver | 1.9 in | 1.5 (proxy) | — | medium | SB | away: hips hang back, flip | toward: early slide |
| `hip_sway` | P5 | mid-iron | 2.3 in | 1.5 (proxy) | — | medium | SB | away: hips hang back, flip | toward: early slide |
| `hip_sway` | impact | driver | 4 in (3.4) | 2 | −0.75 | high | SB, GEARS | away: hang back, flip, fat or thin | toward: slide, blocks |
| `hip_sway` | impact | mid-iron | 4.75 in (3.7) | 1.7 | −0.75 | high | SB, GEARS | away: hang back, flip, fat or thin | toward: slide, blocks |
| `chest_sway` | top | driver | −1.3 in (−0.9) | 1.4 (proxy) | — | medium | SB | away: upper body sways off the ball | toward: reverse pivot |
| `chest_sway` | top | mid-iron | −0.1 in | 1.4 (proxy) | — | medium | SB | away: upper body sways off the ball | toward: reverse pivot |
| `chest_sway` | P5 | driver | 0.7 in | 1.4 (proxy) | — | medium | SB | away: chest hangs back | toward: chest ahead early, over-the-top |
| `chest_sway` | P5 | mid-iron | 2 in | 1.4 (proxy) | — | medium | SB | away: chest hangs back | toward: chest ahead early, over-the-top |
| `chest_sway` | impact | driver | −0.2 in (−1) | 1.7 | — | high | SB, GEARS | away: hang back, flip, hook | toward: chest ahead of the ball, slice, pull or thin |
| `chest_sway` | impact | mid-iron | 1.8 in (0.9) | 1.4 | — | high | SB, GEARS | away: hang back, flip, hook | toward: chest ahead of the ball, slice, pull or thin |
| `sway_gap` | address | driver | −1.4 in | 1 (proxy) | — | medium | SB | away: chest further behind the pelvis, excess tilt | toward: chest level with or ahead of the pelvis, level or forward setup |
| `sway_gap` | address | mid-iron | −1 in | 1 (proxy) | — | medium | SB | away: chest further behind the pelvis, excess tilt | toward: chest level with or ahead of the pelvis, level or forward setup |
| `sway_gap` | top | driver | −2.2 in | 1.5 (proxy) | — | medium | SB | away: chest further behind the pelvis, excess | toward: chest over or ahead of the pelvis, reverse pivot |
| `sway_gap` | top | mid-iron | −1.1 in | 1.5 (proxy) | — | medium | SB | away: chest further behind the pelvis, excess | toward: chest over or ahead of the pelvis, reverse pivot |
| `sway_gap` | impact | driver | −6.7 in (−5.7) | 2 (proxy) | — | medium | SB | away: excess lean back, fat or push | toward: chest ahead, slice or pull pattern |
| `sway_gap` | impact | mid-iron | −4.5 in (−3.8) | 2 (proxy) | — | medium | SB | away: excess lean back, fat or push | toward: chest ahead, slice or pull pattern |
| `lateral_spine_tilt` | address | any | 4° | 3 (proxy) | — | low | SB | less: no spine tilt, level | more: excess tilt away |
| `lateral_spine_tilt` | top | any | 4.5° | 3 (proxy) | — | low | SB | less: spine toward the target, reverse pivot | more: excess tilt away |
| `lateral_spine_tilt` | impact | driver | 18° | 4.7 | — | high | GEARS, SB | less: chest ahead, over-the-top, slice | more: excess lean away, fat, push or hook |
| `lateral_spine_tilt` | impact | mid-iron | 13.9° | 4.4 | — | high | GEARS, SB | less: chest ahead, over-the-top, slice | more: excess lean away, fat, push or hook |
| `head_lateral` | top | any | −1 in | 1.2 | — | low | BK, SB | away: head sways off the ball | toward: reverse pivot |
| `head_lateral` | impact | any | −0.5 in | 1.5 (proxy) | — | low | BK | away: head falls back, flip | toward: head ahead, thin or slice |
| `head_vertical` | top | any | −0.25 in | 0.8 | — | low | BK, SB | down: dip | up: lift, loss of posture |
| `head_vertical` | impact | any | −0.75 in | 0.8 (proxy) | — | low | SB | down: dip, fat | up: lift, thin |
| `pelvis_lift` | top | any | −1.6 in (−1.4) | 0.9 (proxy) | — | medium | SB, SBR | down: too deep | up: no drop, stands tall |
| `pelvis_lift` | impact | any | 0.9 in (1) | 1.2 (proxy) | — | medium | SB, SBR | down: no post-up, loss of speed | up: excess rise, early-extension look |
| `early_extension` | top | any | 0.5 in | 1 (proxy) | — | low | SBEE, VG | back: falls away from the ball | toward ball: early extension in the backswing |
| `early_extension` | impact | any | 0.4 in | 1.3 (proxy) | — | medium | SBEE | back: falls back, fat | toward ball: early extension, blocks, hooks or thin |
| `hand_depth` | top | any | 3 in | 1 (proxy) | — | low | GT | less: hands in front, narrow, across the line | more: hands deep, laid off, stuck |
| `tempo_ratio` | swing | any | 3 | 0.5 (proxy) | — | medium | CHE, VG | less: quick backswing | more: slow backswing |
| `downswing_time` | swing | any | 0.25 s (0.3) | 0.05 (proxy) | — | medium | CHE | never a miss | more: slow downswing, loss of speed |

### Amateur

| Metric | At | Club | Centre (women) | SD | Shape offset | Confidence | Sources | Low: direction, tends to | High: direction, tends to |
|---|---|---|---|---|---|---|---|---|---|
| `hip_turn` | top | driver | −43.8° | 9.9 | — | medium | SEV | closed: over-turned hips, sway and loss of separation | open: restricted hip turn, arms-only backswing, loss of coil |
| `hip_turn` | top | mid-iron | −41.8° | 10.2 | — | medium | SEV | closed: over-turned hips, sway and loss of separation | open: restricted hip turn, arms-only backswing, loss of coil |
| `hip_turn` | impact | driver | 20.8° | 10.6 | 3 | medium | SEV | closed: hang-back, flip, hook or push pattern | open: fade or slice pattern when the arms lag |
| `hip_turn` | impact | mid-iron | 15.6° | 11.1 | 3 | medium | SEV | closed: hang-back, flip, hook or push pattern | open: fade or slice pattern when the arms lag |
| `hip_turn` | finish | driver | 88.8° | 16 | — | medium | SEV | closed: stalled pivot, flip | never a miss |
| `hip_turn` | finish | mid-iron | 82.4° | 16 | — | medium | SEV | closed: stalled pivot, flip | never a miss |
| `shoulder_tilt` | top | any | −29.6° | 6 (proxy) | — | low | GT | more: steep shoulder plane, reverse-pivot look | less: flat shoulder turn, lifting, over-the-top risk |
| `shoulder_tilt` | impact | any | 20° | 6 (proxy) | −3 | medium | MEI | less: level shoulders, out-to-in, thin or slice | more: hanging back, fat or push-draw |
| `lateral_spine_tilt` | finish | any | 3.2° | 10 (proxy) | — | low, ungraded | GT | less: upright finish, no extension, stalled, slice or short | more: excess arch |
| `spine_forward_bend` | impact | mid-iron | 41° | 5.8 | — | medium | SEV | less: stood up, early extension, thin or hook | more: bent over further than tour |
| `hip_tilt` | top | driver | −4.7° | 3.4 | — | medium | SEV | more: lead hip dropped further, excess | less: level hips, no hip coil, sway |
| `hip_tilt` | top | mid-iron | −8.1° | 3 | — | medium | SEV | more: lead hip dropped further, excess | less: level hips, no hip coil, sway |
| `hip_tilt` | impact | driver | 7.5° | 4.9 | — | medium | SEV | less: level hips, stalled | more: excess lead-hip rise, early-extension look |
| `hip_tilt` | impact | mid-iron | 6.8° | 5.8 | — | medium | SEV | less: level hips, stalled | more: excess lead-hip rise, early-extension look |
| `lead_knee_flex` | top | driver | 45° | 10 | — | medium | SEV | less: lead leg firm early | more: lead knee collapses toward the ball |
| `lead_knee_flex` | top | mid-iron | 49° | 10 | — | medium | SEV | less: lead leg firm early | more: lead knee collapses toward the ball |
| `lead_knee_flex` | impact | driver | 23.5° | 9 | — | medium | SEV | less: lead leg posted straight | more: lead knee still bent, no post-up |
| `lead_knee_flex` | impact | mid-iron | 23.7° | 9 | — | medium | SEV | less: lead leg posted straight | more: lead knee still bent, no post-up |
| `lead_knee_flex` | finish | driver | 10° | 10 | — | medium | SEV | less: lead leg straight | more: lead knee still flexed |
| `lead_knee_flex` | finish | mid-iron | 13° | 10 | — | medium | SEV | less: lead leg straight | more: lead knee still flexed |
| `trail_knee_flex` | top | driver | 27° | 12 | — | medium | SEV | less: trail leg straightened, loss of posture | more: trail knee sinks |
| `trail_knee_flex` | top | mid-iron | 24° | 12 | — | medium | SEV | less: trail leg straightened, loss of posture | more: trail knee sinks |
| `trail_knee_flex` | impact | driver | 33° | 9 | — | medium | SEV | less: trail knee straight | more: trail knee bent, kicked in |
| `trail_knee_flex` | impact | mid-iron | 33° | 9 | — | medium | SEV | less: trail knee straight | more: trail knee bent, kicked in |
| `head_lateral` | top | any | −4.9 in | 0.7 | — | low | BK | away: head sways off the ball | toward: reverse pivot |

### Sources

| Key | Source |
|---|---|
| SB | Sportsbox AI tour averages (help.sportsbox.ai; PGA and LPGA players, driver and mid-iron tables) |
| SBR | Sportsbox AI tour ranges (the low-high spans of the same tables) |
| SBEE | Sportsbox AI early-extension tour figures |
| GEARS | GEARS tour data, 61 swings (Barzeski 2022) |
| KV | K-Vest PGA norms as printed in Grathwohl & Sillevis 2024 |
| SEV | Severin et al. 2022 (17 male amateur golfers, mean handicap 8.7, driver and 6-iron) |
| SOHN | Sohn et al. 2010 |
| BK | Byrne & Kenny 2008 |
| CHE | Cheetham 2014 |
| VG | VectorGOLF metric glossary (tempo and thrust targets for Sportsbox AI trackers) |
| MEI | Meister et al. 2011 (10 male professionals and 5 amateurs, 5-iron; shoulder line to the horizontal) |
| GT | GOLFTEC public tour and high-handicap means (Golf Digest 2017; GOLFTEC lesson videos) |

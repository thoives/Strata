# The expert tier (for users, plain words)

The expert tier is a place you choose — a file on a fast disk, an NVMe, a RAM disk — where Strata can keep the part of a
large model that your graphics card and RAM cannot hold. You give it with `--expert-tier` (one flag per device):

```sh
strata launch --family qwen --model q2_0 --expert-tier /fast_drive/experts.tier,mib=65536
```

`mib=65536` says how much of the device the tier may use (here 64 GB); `reserve=M` keeps some of it free (nothing is written
there); `verify=full` makes the first start check every stored piece against its checksum. The device is a regular file, or on
Linux a block device (a partition, a logical volume, a loop device). Windows and macOS take a regular file.

**The first start writes over the device.** Everything stored in the first `mib` of it is lost. Strata says so every time
a tier is set up. It also stops before writing:

- when a device is given twice, even under another name (a link to it, or a partition and its disk);
- when a device is one of the model's files, sits inside the model's folder, or is the disk the model's files are on;
- when a Linux block device is in use: mounted, used as swap, or part of a RAID or LVM volume;
- when a device holds data that is not an expert tier. An empty file is fine (`truncate -s 64G experts.tier` makes one),
  and so is an earlier tier. For anything else, `strata launch` and setup ask first, and you must type `yes`. Where nobody
  can answer, add `--expert-tier-overwrite`. `--yes` alone is not enough, because data would be lost.

**One process at a time.** A tier belongs to one running Strata engine. While it serves from a device, a second `strata launch`
or setup that names the same device is refused and says which process holds it; nothing is written. The lock ends with the
engine. A relative path is taken from the folder you start `strata launch` in, and `~` means your home folder: the setup and the
engine then get the same absolute path, wherever they run.

**How to write an entry.** Everything before the first comma is the path, kept exactly as you typed it. Then come
`mib=N` (required, at least 1), `reserve=M` (less than `mib`) and `verify=header` or `verify=full`, each at most once.
The names may be in any case. The numbers are plain digits: no sign, no spaces, no decimals. An empty field (two commas
in a row, or a comma at the end) is refused. Both engines, setup and `strata launch` follow these same rules, and setup
writes each entry back in one standard spelling.

**A faster checksum (opt-in).** The tier's checksums are CRC32C. The default is a plain table loop. On an x86-64 CPU
with SSE4.2 you can set `STRATA_TIER_HW_CRC=1` before you start, and the checksums use the processor's CRC instruction
instead. The checksums are the same bytes, so a tier written either way is read either way. This is off by default
because it has been measured only on a development machine, not on the machine a tier is filled on. A fill with the
checksum in the table loop took about 1.5 s for 768 MiB in that test, and about 0.3 s with the instruction (both from a
RAM-backed source; your drives and model files will change the numbers).

**When it helps.** A big model spreads its experts over the GPU, RAM and, in the end, the model files on the SSD. A tier
puts the experts the GPU and RAM cannot keep on the device you gave, once, and reads them from there for every answer.
Setup never adds a tier by itself: it checks the devices you give and says what they carry. The experts the tier cannot
hold keep the usual file path.

**The layout: whole or striped (opt-in).** With more than one device you choose how the tier spreads each expert over them:

- `whole` (the default): each expert is stored on one device. The experts are dealt round-robin over the devices, hottest
  first. This is the layout of earlier versions, and a tier written this way is reused as before.
- `striped` (opt-in, Qwen and DeepSeek engines): each expert is cut into stripes, and the stripes are dealt over the devices in turn.
  The stripe size is `--expert-tier-stripe-kib N` (default 256 KiB; a multiple of 4 KiB, from 4 to 16384).

```sh
strata launch --family qwen --model q2_0 \
  --expert-tier /drive-a/experts.tier,mib=65536 \
  --expert-tier /drive-b/experts.tier,mib=65536 \
  --expert-tier-layout striped --expert-tier-stripe-kib 256 --yes --no-tui
```

For DeepSeek, select `--family deepseek41 --model ist-das-2.00bpw` instead. The same layout flags apply; setup also
needs a suitable RAM-store size, as described in the DeepSeek section below.

The rules, in plain words:

- The layout applies to every `--expert-tier` device. Striped needs 2 to 8 devices. A layout option with no device is refused
  (it would do nothing), and a stripe size with `whole` is refused.
- With striped, the smallest device can limit how much the tier holds. Larger devices may have unused space. The plan and
  the log say how much (the estimate is before the headers and tables, and is not a measured number).
- An expert is placed only when every one of its pieces fits. The first expert that does not fit stops the placement, and the
  colder experts after it stay on the file tier, as with whole.
- An expert no larger than one stripe sits on one device, and the devices take such experts in turn.
- A change of layout, of stripe size, or of the device order refills the whole tier at the next start. A refill rewrites the
  tier's area on every device, as a first fill does. The layout change never permits writing over other data: the device checks
  above still apply.
- With `verify=full`, a bad chunk sends every expert that has a stripe in that chunk to the file tier, and marks the device for a refill.

**Three words for a striped expert in the Qwen engine.** The engine keeps these apart:

- **Held**: a valid copy of the expert is on the tier.
- **Pointer-addressable**: the tier gives one stable address for the expert's bytes. Only a `whole` expert is. A striped expert
  is not pointer-addressable through the tier, even when it fits on one device.
- **Needs gathering**: serving the expert means putting its stripes together in a buffer. Every held striped expert needs it.

**Where a Qwen read comes from.** For one expert the engine takes the first source that holds it: the copy in RAM, then a staged exchange
copy, then the tier, then the model files. The batched reads follow the same order. The bytes are the same from every source,
so only the counters show which one served a read.

- In the prompt, a striped expert the RAM copy does not hold is gathered from the devices into the prompt's own buffer.
- In decoding, a striped expert the RAM copy does not hold is gathered from the devices into a staging slot in RAM: each device's
  pieces are read into their places in that slot. The slot is the pool's ordinary staging slot, so it keeps the same lifetime
  as a staged file blob: the pointer holds until the next call (and through the layer it was asked in), and the GPU copies that
  read a page-locked slot are recorded so the slot is not reused under them. A refused gather is read from the files instead.
- The pointer for a copy that runs later (`blob_stable`) and the refill of a lent cache slot keep the model files for a striped
  expert. Neither is a staging slot, so neither can be recycled under a copy that is still running.

**DeepSeek reads and formats.** `strata-dsv41` gathers the stripes of a tier-held expert into its RAM store. Qwen and DeepSeek
use different tier headers: Qwen writes version 2 for whole and version 3 for striped; DeepSeek writes version 1 for whole and
version 2 for striped. Changing the layout refills the tier.

**Limits.** In Qwen, a lent-cache refill reads the model files rather than the striped tier. GPU validation and a speed
benefit for striping are not established here. Compare `whole` and `striped` on the same hardware and workload before
making a speed claim.

**The first start and every start after.** At the start that fills the tier, the engine copies those experts to it once —
the log says how many bytes it wrote and how long the fill took on your PC. Later starts read the tier's header: if the
model, the device and the order you gave are unchanged, the tier is reused and nothing is copied. Anything different —
another model, another device, a device whose last fill did not finish — writes the tier again. A fill counts as
finished only once all its bytes are safely on the device. If the disk reports a write error, the tier is marked
unfinished and written again at the next start.

**A refill.** If the tier's data is damaged or belongs to another model, the engine does not guess: it refills the tier at
the next start. With `verify=full` a bad piece is found at the start, that expert is read from the model files for now,
and the device is marked so the next start refills it.

**What you see in the log.** One line per device, then a line saying how many experts sit in each place (the shape, not
measured numbers — yours come from your own start):

```text
strata generate: expert tier: filled <n> experts (<g> GiB) in <s> s
strata generate: experts: <a> in the GPU cache, <b> in RAM, <c> in <d> tier device(s), <e> on the file tier
```

(The seconds and sizes are what YOUR start measured; they are not promises about other PCs.)

**What the counters say.** In the `--stats` figures and the server log, an expert read from the tier counts in neither the
RAM number nor the file number. The tier gives its own figures beside them: how many reads it answered, how many MB, and
the time spent copying from it. A tier's seconds are only its copies. So a run with a tier and a run without one are not
comparable through the RAM and file figures; compare a tier run with another tier run on the same device.

**The DeepSeek-V4.1 engine (`strata-dsv41`).** The same flag, one per device, plus `--expert-tier-refill`:

```sh
strata-dsv41 --pack <model> --ssd-cache 96 --expert-tier /fast_drive/ds4.tier,mib=262144
```

Differences from the Qwen engine, in plain words:

- **The model's identity.** The engine looks at each model file where it loads it from (its size and the time it last
  changed), whatever folder you start it in. If a file cannot be read, it stops and says so.
- **The RAM store must be given a size.** DeepSeek's RAM store can size itself from the RAM it sees ("auto"). The tier
  is planned against the smallest the store promises to keep, so with a tier the engine asks for `--ssd-cache GB`
  beside it and says so when it is auto or off (the plan's open item 1). Setup does this for you: with `--expert-tier` it
  works the size out from the fit (the RAM the CPU's experts may keep beside the system's 9 GB, at least 4 GB) and writes
  `--ssd-cache` with it; where no such size exists it stops before any download and says what to do. This also works when you
  run setup again without `--expert-tier`: the devices an earlier run gave are kept, and so is the size they need.
  Without any tier setup writes no `--ssd-cache`, so the store keeps sizing itself.
- **Reuse:** a start after the fill reuses the placement when every device is the one it was filled as (same order: a swapped
  pair refills), every device carries the same placement table, and the model pack (its name, size and change time, hashed
  once at the start), the store's slot size, the model's shape and the two guards are the same. A new expert profile
  does not refill the tier: it can only make the next plan place slightly different experts.
- **Not on a Mac.** The Apple unified-memory path has no disk band for the experts; the engine says so and stops.
- **What you see.** The load prints the fill (experts placed, GB, seconds, one line per device) or says the placement
  was reused and nothing was copied. The decode report's `tier:` line says how many experts sit on how many tier
  devices and how many of the answers' expert reads were copies from the tier instead of reads from the files. As with the
  Qwen engine, a read the tier answered counts in neither the RAM figure nor the file figure.
- **A per-device breakdown.** `DS4_TIER_LOG=1` (off by default; unset or `0` disables) makes `strata-dsv41` print one line per tier device at
  the end of a decode: copies and MB a token, the queue and copy times, the cold share, and the CPU lane's wait for
  that device. It is a diagnostic; the residency probe it adds runs only when the variable is set.
- **Direct reads (experimental, Linux, off by default).** Set `DS4_TIER_DIRECT=1` in the engine's environment at start to
  allow unbuffered tier reads into the RAM store. With `whole`, a tier hit can read its slot in one operation. With
  `striped`, each stripe can be read directly from its device into its place in the store. Unset or `0` keeps mapped copies;
  any other value stops the start with a message. The tier format does not change.
  - Each direct read must meet the device's offset and length alignment, the destination's memory alignment and the store's
    bounds. A stripe is not rounded into a neighboring stripe. If a stripe cannot meet these rules, it uses a mapped copy;
    one expert can therefore use both direct reads and mapped copies. A device without usable direct reads stays mapped.
  - If a device reports that a direct read is unsupported while running, that read falls back to a mapped copy and later reads
    from that device use copies. Other read failures are not treated as an unsupported-direct fallback.
  - At startup the log reports each device's direct or mapped capability and, for direct devices, its alignment. With
    `DS4_TIER_LOG=1`, the striped report counts logical expert fills separately from each device's direct and mapped stripe
    jobs, their outcomes, bytes, and queue and service times. A direct job is one stripe. Capability alone does not mean
    every stripe was read directly; mapped stripes that never qualified for a direct attempt are not direct-job fallbacks.
    Job bytes are requested ranges, not measured disk traffic. Counts can include cancelled or in-flight work.
  - Host tests include real Linux direct reads where the test filesystem supports them. GPU output checks and a performance
    comparison on the intended hardware are still needed; no speed benefit is claimed.
- **Direct read span (experimental, off by default, whole layout only).** With `DS4_TIER_DIRECT=1`,
  `DS4_TIER_DIRECT_SPAN=1` tries an aligned span covering the three matrices instead of the whole slot. Unset or `0` uses the
  whole slot. If the span cannot meet alignment or bounds, the engine tries the whole slot; if that cannot be read directly,
  it uses mapped copies. The option does not change striped reads or the tier format. Any reduction in bytes read or benefit
  on real hardware is unproven.
- **How the CPU lane waits (experimental, off by default).** `DS4_TIER_WAKE=1` makes the CPU lane sleep while it waits
  for a tier read. Whole-layout reads use the device's queue; striped reads use a shared signal from all participating
  queues. Unset or `0` keeps the wait of the last release (a look every 2 ms). The variable is read when the engine starts.
  It has not been measured on real hardware yet: no speed claim is made, and the tier format is unchanged.
- **Waiting for any of the CPU lane's reads (experimental, off by default).** `DS4_LANE_WAKE=1` lets the CPU lane compute
  whichever of its experts is ready first, instead of waiting for the first one it asked for. Unset or `0` keeps the
  old order. Not measured on real hardware yet.
- **Store and draft switches, for comparing builds.** These change how the RAM store and the DeepSeek draft experts are
  handled. They are meant not to change the answers; a byte check on a GPU is still needed to show that. None of them is
  measured on real hardware yet, so no speed claim is made.
  - `DS4_STORE_PIN_FIRST=0` (on by default) gives back the order before the fix that keeps the experts a request already
    has in RAM from being evicted by that same request's misses. Unset or `1` keeps the fix.
  - `DS4_STORE_CANCEL_SCAN=1` (off by default) checks the whole RAM store on every cancel, as before. Unset or `0` skips
    the check when no guessed read is in progress.
  - `DS4_STORE_RECLAIM_ESTIMATE=1` (off by default) gives RAM back by the old count of chunks, not by their real sizes.
    Unset or `0` uses the real sizes.
  - `DS4_DRAFT_BATCH=0` (on by default) sends the draft experts through the old split, one expert at a time. Unset or `1`
    sends the whole batch in one call.
  - `DS4_VERIFY_NOFENCE=1` (off by default, not checked on a GPU yet) lets a verify step on one graphics card go on before
    the host waits for it. Do not use it for real answers until greedy and sampled output have been checked on a GPU.

This engine's plan is [EXPERT_TIER_DSV41_PLAN.md](EXPERT_TIER_DSV41_PLAN.md).

No speed number is published for tiers yet — the feature is being measured on real hardware first
(see [EXPERT_TIER_PLAN.md](EXPERT_TIER_PLAN.md)). The engine's own reference for the flag and its bytes is
[DETAILS.md](DETAILS.md#expert-tier-expert-tier-opt-in-off-by-default).

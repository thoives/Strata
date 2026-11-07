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

**When it helps.** A big model spreads its experts over the GPU, RAM and, in the end, the model files on the SSD. A tier
puts the experts the GPU and RAM cannot keep on the device you gave, once, and reads them from there for every answer.
Setup never adds a tier by itself: it checks the devices you give and says what they carry. The experts the tier cannot
hold keep the usual file path.

**The first start and every start after.** At the start that fills the tier, the engine copies those experts to it once —
the log says how many bytes it wrote and how long the fill took on your PC. Later starts read the tier's header: if the
model, the device and the order you gave are unchanged, the tier is reused and nothing is copied. Anything different —
another model, another device, a device whose last fill did not finish — writes the tier again. A fill counts as
finished only once all its bytes are safely on the device. If the disk reports a write error, the tier is marked
unfinished and written again at the next start. A tier made by an earlier test version of this engine (header version 1)
is written again once.

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

This engine's plan is [EXPERT_TIER_DSV41_PLAN.md](EXPERT_TIER_DSV41_PLAN.md).

No speed number is published for tiers yet — the feature is being measured on real hardware first
(see [EXPERT_TIER_PLAN.md](EXPERT_TIER_PLAN.md)). The engine's own reference for the flag and its bytes is
[DETAILS.md](DETAILS.md#expert-tier-expert-tier-opt-in-off-by-default).

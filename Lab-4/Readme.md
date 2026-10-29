# Lab 4: Stable Diffusion and Text Generation on the DGX Spark

In this lab you run two generative AI web UIs on an NVIDIA DGX Spark and measure how their settings change the output. Each part takes about 40 minutes. You do **not** need sudo for anything.

## Stations and ports

| Group   | Stable Diffusion port (`SD_PORT`) | Text Generation port (`TG_PORT`) |
| ------- | --------------------------------- | -------------------------------- |
| group01 | 7861 | 7961 |
| group02 | 7862 | 7962 |
| group03 | 7863 | 7963 |
| group04 | 7864 | 7964 |
| group05 | 7865 | 7965 |
| group06 | 7866 | 7966 |
| group07 | 7867 | 7967 |
| group08 | 7868 | 7968 |
| group09 | 7869 | 7969 |
| group10 | 7870 | 7970 |

The terminal always prints `Running on local URL: http://0.0.0.0:7860`. That is the port **inside** the container; in your browser always use **your** port from the table.

## Lab report requirements

Submit **one report per group** containing:

1. Your group number, station and the names of everyone in the group.
2. The **Part 1** report table with every row filled in: a screenshot of the image and your remarks compared with the baseline.
3. The **Part 2** report table with every row filled in: the reply (pasted text or a screenshot), tokens and tokens/s, and your remarks compared with the baseline.
4. Answers to the reflection questions for both parts (2–3 sentences each, using your runs as evidence).

Rules for every run:

- Start from the **baseline** settings and change **only** the values shown in **bold** in that row. Set them back before the next row.
- **Remarks** = one or two lines on how the result differs from the baseline (Part 2: also from run 5 where noted).
- Tip: name your screenshots `sd-01.png`, `sd-02.png`, … and `tg-01.png`, … so they are easy to put in the report. In a Markdown report you can embed one in a table cell with `![](sd-01.png)`.

---

## Part 1: Stable Diffusion

### Commands to run

```bash
# 1. Log in to your station (replace XX and STATION_IP)
ssh groupXX@STATION_IP

# 2. Start tmux so the web UI survives a dropped connection
tmux new -s lab            # reconnect later with: tmux attach -t lab

# 3. Start Stable Diffusion; it prints your URL
sd-webui

# 4. Wait for "Running on local URL", then open in your browser:
#    http://STATION_IP:SD_PORT

# 5. When you are finished: press Ctrl + C in the terminal, or run
sd-webui stop
```

### Baseline settings (txt2img tab)

| Setting | Value |
| --- | --- |
| Prompt | `a cozy wooden cabin in a snowy pine forest at dusk, warm light glowing in the windows, smoke from the chimney, highly detailed photograph` |
| Negative prompt | `blurry, low quality, deformed, text, watermark` |
| Width × Height | 512 × 512 |
| Sampling method | DPM++ 2M |
| Schedule type | Karras |
| Sampling steps | 25 |
| CFG Scale | 7 |
| Seed | 12345 |
| Batch count | 1 |
| Batch size | 1 |

Leave everything else at its default (Hires. fix and Refiner off). For each run, take a screenshot of the image **together with the info text under it**, and copy the "Time taken" value into the Time column.

### Lab report table

| Run | What it tests | Sampling method | Schedule type | Steps | CFG | Seed | Batch count | Batch size | Time (s) | Screenshot | Remarks vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **Baseline** | DPM++ 2M | Karras | 25 | 7 | 12345 | 1 | 1 | | | Describe the baseline |
| 2 | Sampler | **Euler a** | Karras | 25 | 7 | 12345 | 1 | 1 | | | |
| 3 | Sampler | **DPM++ 2M SDE** | Karras | 25 | 7 | 12345 | 1 | 1 | | | |
| 4 | Sampler | **DPM adaptive** | Karras | 25 | 7 | 12345 | 1 | 1 | | | |
| 5 | Sampler | **LCM** | Karras | 25 | 7 | 12345 | 1 | 1 | | | |
| 6 | Schedule | DPM++ 2M | **Uniform** | 25 | 7 | 12345 | 1 | 1 | | | |
| 7 | Schedule | DPM++ 2M | **Exponential** | 25 | 7 | 12345 | 1 | 1 | | | |
| 8 | Schedule | DPM++ 2M | **Align Your Steps** | 25 | 7 | 12345 | 1 | 1 | | | |
| 9 | Schedule | DPM++ 2M | **Beta** | 25 | 7 | 12345 | 1 | 1 | | | |
| 10 | Sampler × schedule grid | **X: 5 samplers** | **Y: 5 schedules** | 25 | 7 | 12345 | 1 | 1 | | | Which column changes least? Best and worst combination? |
| 11 | Steps | DPM++ 2M | Karras | **5** | 7 | 12345 | 1 | 1 | | | |
| 12 | Steps | DPM++ 2M | Karras | **10** | 7 | 12345 | 1 | 1 | | | |
| 13 | Steps | DPM++ 2M | Karras | **50** | 7 | 12345 | 1 | 1 | | | |
| 14 | Steps | **Euler a** | Karras | **50** | 7 | 12345 | 1 | 1 | | | Also compare with run 2 |
| 15 | CFG | DPM++ 2M | Karras | 25 | **2** | 12345 | 1 | 1 | | | |
| 16 | CFG | DPM++ 2M | Karras | 25 | **12** | 12345 | 1 | 1 | | | |
| 17 | CFG | DPM++ 2M | Karras | 25 | **25** | 12345 | 1 | 1 | | | |
| 18 | Seed | DPM++ 2M | Karras | 25 | 7 | 12345 | 1 | 1 | | | Same as baseline or not? |
| 19 | Seed | DPM++ 2M | Karras | 25 | 7 | **12346** | 1 | 1 | | | |
| 20 | Seed | DPM++ 2M | Karras | 25 | 7 | **98765** | 1 | 1 | | | |
| 21 | Batch | DPM++ 2M | Karras | 25 | 7 | 12345 | **4** | 1 | | | Is the first image the baseline? How do the 4 differ? |
| 22 | Batch | DPM++ 2M | Karras | 25 | 7 | 12345 | 1 | **4** | | | Same 4 images as run 21? Faster or slower? |

### Reflection questions (Part 1)

1. Which sampling method changed the image most compared with the baseline? What about how it works might explain that?
2. Compare runs 1 and 13 (DPM++ 2M at 25 and 50 steps) with runs 2 and 14 (Euler a at 25 and 50 steps). Which sampler only refined the same picture, and which changed the picture itself? Why?
3. In the grid (run 10), which sampler's column barely changed between schedule types? Why might that be?
4. Which CFG value gave the best balance between following the prompt and looking natural? What went wrong at CFG 2 and at CFG 25?
5. Seeds 12345 and 12346 differ by only 1. Did the images look alike? What does that tell you about what a seed is?
6. Did batch count 4 and batch size 4 give the same images? Which was faster overall, and what is the cost of the faster one?
7. Based on your results, which sampler, schedule type and step count would you pick for a good image in the least time?

---

## Part 2: Text Generation

### Commands to run

```bash
# 1. Log in to your station (skip if you are still logged in from Part 1)
ssh groupXX@STATION_IP
tmux attach -t lab || tmux new -s lab

# 2. Stop Stable Diffusion if it is still running (frees GPU memory)
sd-webui stop

# 3. Start Text Generation; it prints your URL and loads Qwen3-4B-Instruct
tg-webui

# 4. Wait for "Running on local URL", then open in your browser:
#    http://STATION_IP:TG_PORT
#    Keep this terminal visible: it prints tokens/s after every reply.

# 5. When you are finished: press Ctrl + C in the terminal, or run
tg-webui stop
```

### Baseline settings

| Setting | Where | Value |
| --- | --- | --- |
| Prompt | Chat tab, message box | `Write a 100-word story about a robot who learns to paint.` |
| Mode | Chat tab | instruct |
| Custom system message | Parameters tab | (empty) |
| temperature | Parameters tab | 0.7 |
| top_p | Parameters tab | 1.0 (off) |
| top_k | Parameters tab | 0 (off) |
| min_p | Parameters tab | 0 (off) |
| repetition_penalty | Parameters tab | 1.0 (off) |
| max_new_tokens | Parameters tab | 300 |
| Seed | Parameters tab | 42 |
| do_sample | Parameters tab | checked |

How to do each run:

1. On the **Parameters** tab, set the baseline values, click **💾** next to **Preset** and save them as `Baseline`. Picking `Baseline` from the Preset menu now resets everything in one click.
2. On the **Chat** tab, check that Mode is `instruct`, type the prompt and click **Send**. This is run 1.
3. For every later run: pick the `Baseline` preset, change only the bold values (check Seed is 42 and max_new_tokens is 300 unless the row says otherwise), then go back to **Chat** and click **Regenerate** (Ctrl + Enter). You never retype the prompt.
4. After each reply the terminal prints a line like `Output generated in 3.21 seconds (78.50 tokens/s, 252 tokens, context 32, seed 42)`. Write it in the table as `252 (78.5)`.
5. Runs 9–11 pick a built-in preset instead of `Baseline`. A preset first resets every sampling setting to its default (temperature 1, all filters off) and then applies its own values. It does not change Seed or max_new_tokens.
6. Runs 19–21: the system message goes in **Custom system message** (Parameters tab). Afterwards, clear it and set Mode back to `instruct`.

### Lab report table

| Run | What it tests | temperature | top_p | top_k | min_p | Rep. penalty | Max new tokens | Seed | Other change | Tokens (tokens/s) | Output | Remarks vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **Baseline** | 0.7 | 1.0 | 0 | 0 | 1.0 | 300 | 42 | – | | | Describe the baseline |
| 2 | Temperature | **0.01** | 1.0 | 0 | 0 | 1.0 | 300 | 42 | – | | | |
| 3 | Temperature | **0.3** | 1.0 | 0 | 0 | 1.0 | 300 | 42 | – | | | |
| 4 | Temperature | **1.2** | 1.0 | 0 | 0 | 1.0 | 300 | 42 | – | | | |
| 5 | Temperature | **2.0** | 1.0 | 0 | 0 | 1.0 | 300 | 42 | – | | | |
| 6 | Filter at high temp | **2.0** | 1.0 | **20** | 0 | 1.0 | 300 | 42 | – | | | Also compare with run 5 |
| 7 | Filter at high temp | **2.0** | **0.8** | 0 | 0 | 1.0 | 300 | 42 | – | | | Also compare with run 5 |
| 8 | Filter at high temp | **2.0** | 1.0 | 0 | **0.1** | 1.0 | 300 | 42 | – | | | Also compare with run 5 |
| 9 | Preset | preset | preset | preset | preset | preset | 300 | 42 | **Preset: Deterministic** (do_sample off, top_k 1) | | | Also compare with run 2 |
| 10 | Preset | preset | preset | preset | preset | preset | 300 | 42 | **Preset: Creative** (min_p 0.02, xtc_probability 0.5) | | | |
| 11 | Preset | preset | preset | preset | preset | preset | 300 | 42 | **Preset: Top-P** (top_p 0.95) | | | |
| 12 | Seed | 0.7 | 1.0 | 0 | 0 | 1.0 | 300 | 42 | – | | | Same as baseline or not? |
| 13 | Seed | 0.7 | 1.0 | 0 | 0 | 1.0 | 300 | **43** | – | | | |
| 14 | Seed | 0.7 | 1.0 | 0 | 0 | 1.0 | 300 | **-1** | – | | | Seed used (from terminal): |
| 15 | Seed | 0.7 | 1.0 | 0 | 0 | 1.0 | 300 | **-1** | – | | | Seed used: ___ Same reply as run 14? |
| 16 | Length | 0.7 | 1.0 | 0 | 0 | 1.0 | **30** | 42 | – | | | |
| 17 | Length | 0.7 | 1.0 | 0 | 0 | 1.0 | **1000** | 42 | – | | | Longer than the baseline or not? Why? |
| 18 | Repetition | 0.7 | 1.0 | 0 | 0 | **1.5** | 300 | 42 | – | | | |
| 19 | System message | 0.7 | 1.0 | 0 | 0 | 1.0 | 300 | 42 | **System message:** `You are a pirate. Always answer like a pirate.` | | | |
| 20 | System message | 0.7 | 1.0 | 0 | 0 | 1.0 | 300 | 42 | **System message:** `Answer in exactly three bullet points.` | | | Did it still write a story? |
| 21 | Mode | 0.7 | 1.0 | 0 | 0 | 1.0 | 300 | 42 | **Mode: chat** (system message empty) | | | |

### Reflection questions (Part 2)

1. How did the reply change as temperature went from 0.01 to 2.0 (runs 2–5)? At what temperature did it stop making sense?
2. Which filter in runs 6–8 (top_k, top_p or min_p) best recovered a readable reply at temperature 2.0? Why might it work better than the others?
3. The Deterministic preset (run 9) and temperature 0.01 (run 2) both aim to pick the single most likely word every time. Were their replies the same? Why or why not?
4. Did repeating seed 42 (run 12) give exactly the baseline reply? Why do two runs with seed -1 (runs 14 and 15) differ?
5. Did max_new_tokens 1000 (run 17) give a longer reply than the baseline? What does that tell you about what max_new_tokens controls?
6. What did repetition_penalty 1.5 (run 18) do to the wording? Why can a high value hurt quality?
7. Compare runs 19–21. How much can a system message or mode change the reply, compared with the sampling settings in runs 2–18?
8. Look at the tokens/s column. Did any setting noticeably change the speed? Which ones only changed the content?

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Conflict. The container name ... is already in use` | Your web UI from an earlier session is still running. Run `sd-webui stop` or `tg-webui stop`, then start it again. |
| Browser: "This site can't be reached" | Wait for "Running on local URL". Use **your** port (78xx or 79xx), not 7860, and `http://` not `https://`. Off campus? Connect to the NJIT VPN first. |
| `sudo: a password is required` or `command not found` | The launcher is not set up for your account. Tell your instructor. |
| SSH connection dropped | SSH back in and run `tmux attach -t lab`. If there is no session, run the `stop` command, then start again. |
| `CUDA out of memory` | Other groups share the GPU. Stop the web UI you are not using, generate one image at a time, or try again shortly. |

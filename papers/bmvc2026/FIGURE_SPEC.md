# Figure 1: Method Overview — Complete Design Specification

## Overall Layout

Full page width (~17cm), split into two main zones:
- **Top zone**: Two side-by-side diagrams (a) and (b) separated by a thin vertical dashed gray line
- **Bottom zone**: Three text panels (c), (d), (e) separated from the top by a thin horizontal gray line

Total height: ~14-16cm. The figure should fill one full page of the paper.

---

## Color Palette

| Color | Usage | Hex (approx) |
|-------|-------|------|
| Amber/Yellow | Input data, prompts | #FFF3CD border, #FFFDE7 fill |
| Blue | Model components | #BBDEFB border, #E3F2FD fill |
| Red/Coral | Frontier model, CoT outputs | #FFCDD2 border, #FFEBEE fill |
| Green | Final outputs, JSON | #C8E6C9 border, #E8F5E9 fill |
| Purple/Lavender | Loss function | #E1BEE7 border, #F3E5F5 fill |
| Gray | Labels, annotations, grouping | #E0E0E0 border, #FAFAFA fill |
| White | Text panels | #FFFFFF fill, gray border |

All boxes: rounded corners (4px), thin border (1-1.5px), subtle fill (pastel).

---

## TOP-LEFT: (a) Direct LoRA Training

**Title** (above dashed box, bold, 10pt): "Direct LoRA training"

**Flow**: Horizontal, left-to-right

### Elements

```
[Photo] → [Prompt Box] → [Model Block] → [Output Box] → [Loss]
                                                              ↑
                                                         [Label y_i]
```

**1. Training Image** (actual photo, ~1.5×1.5cm)
- Use the A16 concrete aggregate photograph
- Thin gray border (1px)
- Small label below in gray italic: "Training image x_i"
- To the right of the image, a tiny annotation: "×N images"

**2. Arrow** → thin gray, horizontal

**3. Prompt Box** (yellow, ~3cm wide)
- Title line (bold, inside box): "Classification Prompt"
- Body text (small, 3-4 lines visible):
  ```
  Classify this aggregate.
  GSD = 2.1 px/mm
  Classes: coarse/medium/fine
  Respond with JSON: {...}
  ```
- Small tag at bottom-right corner of box: "with JSON instruction"

**4. Arrow** → 

**5. Model Block** (blue, ~3cm wide, slightly taller)
- Two sub-components shown INSIDE the blue box (stacked):
  - Top section (darker blue): "Qwen2.5-VL-3B" with subtitle "3.8B params, frozen"
  - Bottom section (orange/yellow accent): "LoRA Adapter" with subtitle "r=16, α=32, trainable"
- Small icon: a lock icon 🔒 on the frozen part, a pencil ✏️ on the LoRA part

**6. Arrow** →

**7. Output Box** (green, ~2.5cm wide)
- Title: "Response"
- Content (monospace):
  ```
  {"max_particle_size_mm": 16,
   "grading": "coarse"}
  ```

**8. Arrow** →

**9. Loss Box** (purple, ~1.5cm)
- Text: "CE Loss"

**10. Label Box** (gray, above Loss, connected by dashed arrow ↓)
- Text: "Ground truth y_i"

**Bottom annotation** (italic, gray, centered below pipeline):
"N training pairs (1 per image, 18-30 total)"

**Dashed rounded rectangle** (light gray, ~1px) wraps the entire pipeline loosely.

**Sub-label** at bottom-left corner of dashed box: "(a)"

---

## TOP-RIGHT: (b) CoT-Augmented LoRA Training

**Title** (above dashed box, bold, 10pt): "CoT-augmented LoRA training"

**Flow**: Top-to-bottom (inputs → frontier) then branches, then reconverges into model.

### Top Row (3 inputs, arranged horizontally)

```
[Photo x_i]    [Label y_i]    [Prompt (no JSON)]
     \              |              /
      \             |             /
       ↓            ↓            ↓
     [   GPT-4.1 Frontier Model   ]
              |              |
              ↓              ↓
    [1× Direct]      [3× CoT Descriptions]
              \              /
               ↓            ↓
         [Augmented Training Data]
                    ↓
         [Qwen2.5-VL-3B + LoRA]
                    ↓
               [CE Loss]
```

**1. Training Image** (same photo as left side, ~1.5×1.5cm)
- Label below: "Same image x_i"

**2. Label Box** (gray rounded box)
- Title: "Correct label"
- Content: "16mm, coarse"
- Small annotation below: "answer provided"

**3. Prompt Box** (yellow, similar to left but with visual difference)
- Title: "Classification Prompt"
- Body shows same text BUT with the last line crossed out (strikethrough):
  ```
  Classify this aggregate.
  GSD = 2.1 px/mm
  Classes: coarse/medium/fine
  ~~Respond with JSON: {...}~~  ← visually struck through in red
  ```
- Tag: "without JSON instruction"

**4. Three arrows converging DOWN into:**

**5. Frontier Model Box** (red/coral, wide ~6cm, prominent)
- Title (bold): "GPT-4.1"
- Subtitle: "Frontier Model (answer-conditioned)"
- Small annotation to the right of the box: "Knows the correct answer → generates justified reasoning"
- Small icon inside: a brain or lightbulb icon

**6. Two arrows branching DOWN from GPT-4.1:**

**Left branch →**

**7a. Direct Pair Box** (green, smaller)
- Title: "1× Direct pair"
- Content:
  ```
  {"max_particle_size_mm": 16,
   "grading": "coarse"}
  ```

**Right branch →**

**7b. CoT Description Box** (red/coral tint, larger ~5cm wide)
- Title: "3× CoT Descriptions"
- Subtitle: "temperature = 0.7, diverse reasoning"
- Show 3 small stacked cards/pages (slight offset to show multiplicity):
  - Front card shows truncated text:
    ```
    "The largest stones appear to
    be ~16mm. Gaps are EMPTY..."
    ```
- Small annotation at bottom: "+ JSON appended by code"

**8. Both boxes have arrows pointing DOWN into:**

**9. Augmented Data Box** (blue, wide)
- Title (bold): "Augmented Training Data"
- Visual: show 4 small stacked document icons representing 4× data multiplier
- Annotation: "4N examples from N images"

**10. Arrow DOWN →**

**11. Model Block** (identical to left side — blue with LoRA sub-block)
- "Qwen2.5-VL-3B" (frozen) + "LoRA" (trainable)
- Same lock/pencil icons

**12. Arrow DOWN →**

**13. Loss Box** (purple)
- "CE Loss"
- Annotation below: "4N training pairs from same N images"

**Dashed rounded rectangle** wraps everything.

**Sub-label**: "(b)"

---

## KEY COMPARISON ANNOTATION

Between (a) and (b), along the vertical separator line, add a small callout box or bracketed annotation:

```
┌─────────────────────┐
│  Key difference:    │
│  Same N images      │
│  1× data vs 4× data│
│  JSON vs Reasoning  │
└─────────────────────┘
```

---

## BOTTOM-LEFT: (c) Example Prompt

**Separated by horizontal line. Sub-label "(c)" at bottom.**

White panel with thin gray border. Width: ~5.5cm.

**Header** (bold, 9pt): "Classification Prompt (used for all tasks)"

**Content** (monospace, 7-8pt, showing the actual granulometry prompt):
```
Classify this concrete aggregate photograph.
Ground sampling distance (GSD) = 2.1 px/mm.
At this GSD: 8mm ≈ 17px, 16mm ≈ 34px, 32mm ≈ 68px.

Classification axes:
1. MAX PARTICLE SIZE: estimate the largest
   stone's width in pixels, divide by GSD,
   round to 8, 16, or 32 mm.
2. GRADING (DIN 1045 standard):
   - COARSE (A): gaps between stones EMPTY
   - MEDIUM (B): gaps PARTIALLY filled
   - FINE (C): gaps COMPLETELY filled

Respond with JSON:
{"max_particle_size_mm": <8|16|32>,
 "grading": "<coarse|medium|fine>"}
```

---

## BOTTOM-CENTER: (d) Direct Response

White panel with **green border**. Width: ~3cm.

**Header** (bold): "Direct Response"

**Content** (monospace):
```
{"max_particle_size_mm": 16,
 "grading": "coarse"}
```

Small annotation below in gray: "Approach A output format"

---

## BOTTOM-RIGHT: (e) CoT Response

White panel with **red/coral border**. Width: ~6.5cm. This is the largest panel.

**Header** (bold): "CoT Response (GPT-4.1, ×3 at temp=0.7)"

**Content** (regular font, in quotation marks):
```
"The largest stones in this image appear to be 
around 16mm, matching the 34px expected size at 
this GSD. The gaps between the largest stones are 
mostly EMPTY, with very few smaller particles 
filling the spaces. This is characteristic of 
COARSE grading (DIN 1045 type A).

It is NOT medium, because medium would show gaps 
partially filled by smaller particles. It is NOT 
fine, because fine would show a dense packed 
texture with no visible gaps."

+ {"max_particle_size_mm": 16, "grading": "coarse"}
```

**Key visual elements in the text**:
- "EMPTY", "COARSE" in **bold**
- "NOT medium", "NOT fine" in **bold** (showing contrastive reasoning)
- The JSON at the bottom in monospace with a small annotation: "(appended programmatically)"

Small annotation below in gray: "Approach B output format — teaches reasoning + decision boundaries"

---

## Additional Visual Elements to Include

### Arrow Annotations
On key arrows, add small text labels:
- Arrow from image to model (left side): "encode"
- Arrow from GPT-4.1 to CoT box: "generate ×3"
- Arrow from augmented data to model: "fine-tune (40 epochs)"

### Data Multiplier Visualization
Next to the augmented data box, show a visual comparison:
```
Approach A:  ■ (18 examples)
Approach B:  ■■■■ (72 examples, same 18 images)
```
Use small colored squares to represent data volume.

### Accuracy Callout
At the very bottom of the figure (below panels c/d/e), add a thin results banner:
```
Result: Base 12.0% → Direct LoRA 71.3% → CoT-Augmented 79.6% (+8.3pp)
```
Use green for improvement numbers.

### Teacher Accuracy Warning
Near the GPT-4.1 box, add a small warning/info callout:
```
⚠️ Teacher accuracy: 29.6% (can't classify correctly)
   But CAN describe visual features when given the answer
```
This visually communicates WHY answer-conditioning is necessary.

---

## Technical Specifications

| Property | Value |
|----------|-------|
| Total width | 17cm (full column width) |
| Total height | 14-16cm |
| Font family | Helvetica/Arial (sans-serif) |
| Title font size | 10pt bold |
| Body font size | 7-8pt |
| Monospace font | Courier/Consolas 7pt |
| Arrow thickness | 1px |
| Box border | 1-1.5px |
| Corner radius | 4px (boxes), 8px (dashed groups) |
| Export format | PDF (vector) |
| Color mode | CMYK (for print) |

---

## Files to Provide to Designer

1. This specification document
2. `sample_A16.jpg` — the concrete aggregate photo to embed
3. The reference figure image (Li et al. FADE paper) for style reference
4. Color palette swatches

---

## What Makes This Figure Good (tell the designer)

1. **Two approaches clearly compared** — reader immediately sees the difference
2. **Actual content visible** — real prompt text, real JSON, real reasoning
3. **Flow is readable** — left-to-right for simple, top-to-bottom for complex
4. **Consistent color coding** — each element type has one color throughout
5. **Whitespace** — don't crowd, let elements breathe
6. **Hierarchy** — titles > content > annotations, each at different sizes/weights
7. **The key insight is visually obvious** — GPT-4.1 receives the answer (shown explicitly), generates reasoning that includes contrastive "NOT X because..." statements

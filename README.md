# CO3133 Group Website — File Structure & Setup

This repo is a plain HTML/CSS site (no build step) designed to be served directly
by **GitHub Pages**.

```
.
├── index.html                  # Shared landing page (institution, course, group,
│                                 AI disclosure, assignment index with deliverable links)
├── assignment1.html             # Assignment 1 page
├── assignment2.html             # Assignment 2 page (+ Dataset Proposal section)
├── assignment3.html             # Assignment 3 page (+ Dataset Proposal + modality-interaction section)
├── assets/
│   ├── css/
│   │   └── style.css            # Shared stylesheet for all pages
│   └── images/                   # Put EDA plots, diagrams, screenshots here
├── code-repo-template/           # Template for the *source-code* repository
│   ├── README.md                 # install / data prep / train / evaluate / traceability
│   ├── AI_USAGE.md                # repo-level AI usage disclosure
│   ├── configs/
│   │   └── config.example.yaml
│   ├── environment/
│   │   ├── requirements.txt
│   │   └── environment.yml
│   ├── src/                       # model / data / training code goes here
│   ├── scripts/
│   │   ├── prepare_data.sh
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── checkpoints/
│   │   └── README.md              # download links or reconstruction instructions
│   ├── results/
│   │   └── experiment_log.csv     # traceability: result → config → split → checkpoint → log ID → commit
│   └── .gitignore
└── README.md
```

`code-repo-template/` is **not** part of the GitHub Pages site — it's a starting
point to copy into each assignment's actual source-code repository (one repo per
assignment, or one folder per assignment in a monorepo) so it satisfies the
Section 4 reproducibility requirements out of the box.

## What to fill in

Every placeholder is wrapped like `<span class="ph">[...]</span>` or inside a
`<div class="ph-block">...</div>` and rendered with a dashed amber box so it's
easy to spot while editing. Search each HTML file for `[` to find them quickly,
or search for `class="ph"`.

Checklist per file:

- **index.html** — group name/ID, all members' names/IDs/roles/GitHub links,
  repo link, assignment titles, course-wide AI usage disclosure.
- **assignment1.html / assignment2.html / assignment3.html** — title, problem
  statement, dataset & EDA, methodology, experimental setup, results,
  comparison & discussion, error analysis, limitations & conclusion,
  source code / checkpoint / report-slides / YouTube links, and an
  assignment-specific AI usage disclosure. Assignment 3 additionally has a
  "Modality interaction" section (support vs. conflict cases).

Remember:
- Do not invent GitHub profile links — write "N/A" if a member has none.
- YouTube videos must be **Public or Unlisted**, titled
  `CO3133-Semester-261 – Group [ID] – Assignment [1/2/3]`, and verified before
  submission.

## Deliverables checklist (per assignment)

- [ ] Assignment page on GitHub Pages (`assignmentN.html`)
- [ ] Source-code repository (use `code-repo-template/` as a starting point)
- [ ] Report
- [ ] Presentation slides
- [ ] YouTube presentation video (Public or Unlisted, correct title format)
- [ ] AI Usage Disclosure in **all four** places: landing page, assignment page,
      report, and `AI_USAGE.md` in the code repo
- [ ] Checkpoint(s) or reconstruction instructions
- [ ] LMS submission per course rules
- [ ] Assignments 2 & 3 only: Dataset Proposal record + approval status

## Deploying on GitHub Pages

1. Push this repo to GitHub (e.g. `group-XX-co3133`).
2. Go to **Settings → Pages**.
3. Under "Build and deployment", set **Source** to `Deploy from a branch`.
4. Choose the branch (e.g. `main`) and folder `/ (root)`, then **Save**.
5. Your site will be published at:
   `https://<your-username>.github.io/<repo-name>/`
6. Put that URL (and the repo link) wherever your course submission requires it.

## Adding images (EDA plots, diagrams, screenshots)

Save image files into `assets/images/` and reference them in a page like:

```html
<img src="assets/images/your-file.png" alt="Describe the image" style="max-width:100%;border-radius:6px;">
```

## Adding more assignments or sections

Copy `assignment1.html`, rename it, and update the `<title>`, kicker,
`<h1>`, section content, and the AI-disclosure heading — keep the shared
`<div class="topbar">` navigation identical across all pages (just add a new
link if you add a page) so navigation stays consistent site-wide.

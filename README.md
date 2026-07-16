# DRIVE

DRIVE is a continual-learning web-agent framework with two reusable skill levels:

- **Reasoning skills** provide generalized method, boundary, and verification guidance.
- **Interaction skills** provide parameterized browser procedures with signatures, preconditions, bodies, postconditions, and recovery metadata.

At each browser step, DRIVE retrieves reasoning guidance from the task and observable state, uses it to form an intent, then retrieves an eligible interaction skill. The runtime validates preconditions and observable postconditions, stages unverified repairs, and falls back to browser primitives when needed.

## Reproduction prerequisites

Prepare the following before running an experiment:

1. A compatible WebArena deployment and browser-accessible service endpoints.
2. A directory of WebArena task JSON configurations.
3. An OpenAI-compatible model endpoint and credentials.
4. A trajectory corpus when reproducing skill induction.
5. A skill-library directory when reproducing skill-enabled evaluation.

The task configuration directory must contain files named <task-id>.json.

## Installation

~~~bash
conda create -n drive python=3.10 -y
conda activate drive
pip install -r requirements.txt
playwright install chromium
cp .env.example .env.local
~~~

Edit .env.local with your service endpoints and model settings, then load it:

~~~bash
set -a
source .env.local
set +a
python browser_env/auto_login.py
~~~

Run browser login only after the WebArena services are available.

## Configure an evaluation

Start from the provided configuration and create a local copy:

~~~bash
cp AgentOccam/configs/AgentOccam.yml /path/to/drive.yml
~~~

Set agent.actor.model to the desired model. To run with a skill library, update the skills block:

~~~yaml
skills:
  use_skills: true
  skill_site: "reddit"
  skill_dir: "/path/to/skills"
  skill_file: "operation_skills.py"
  skill_metadata: "operation_skills.json"
  task_lessons_path: "/path/to/skills/reddit/reasoning_tips.json"
  use_interaction_skills: true
  use_reasoning_skills: true
~~~

The corresponding library directory is expected to contain:

~~~text
/path/to/skills/reddit/
  operation_skills.py
  operation_skills.json
  reasoning_tips.json
~~~

## Run WebArena evaluation

Pass the task configuration directory explicitly:

~~~bash
./scripts/run_webarena.sh /path/to/drive.yml \
  --task-config-dir /path/to/webarena-configs \
  --task-ids 27 28 29
~~~

Use --skill-levels both, reasoning, interaction, or none to control runtime skill use. Outputs are written beneath the configured logdir; the example configuration uses ./runs.

## Reproduce skill induction

First simplify a trajectory corpus and attach task reference information:

~~~bash
python extract_trajectory.py /path/to/trajectories \
  --config /path/to/webarena-configs \
  --merge
~~~

Then build a site library:

~~~bash
./scripts/build_skills.sh reddit /path/to/trajectories /path/to/skills/reddit
~~~

Supported site identifiers are reddit, gitlab, shopping, shopping_admin, and map. The induction pipeline derives interaction and reasoning skills from the supplied trajectories.

## Validate and maintain a library

Validate generated operation-skill code:

~~~bash
python tools/verify_skill_snippets.py \
  --file /path/to/skills/reddit/operation_skills.py
~~~

Run one maintenance round:

~~~bash
./scripts/evolve_skills.sh reddit --skills-root /path/to/skills
~~~

Maintenance consolidates reasoning guidance and prunes interaction entries according to their feedback and utility metadata.

## Repository layout

~~~text
AgentOccam/          DRIVE policy, retrieval, contracts, prompts, and runtime wrappers
browser_env/         WebArena-compatible browser environment
evaluation_harness/  Evaluator integration
llms/                Model-provider utilities
tools/site_specific/ Skill induction and maintenance operators
scripts/             Command-line entry points
~~~

## Operational hygiene

Keep credentials, authentication state, local paths, logs, raw trajectories, and generated libraries outside version control. Retain the license and attribution notices in LICENSE and NOTICE when redistributing the code.

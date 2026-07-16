.PHONY: install browser eval build-skills evolve-skills

install:
	python3 -m pip install -r requirements.txt

browser:
	python3 -m playwright install chromium

eval:
	@echo "usage: ./scripts/run_webarena.sh AgentOccam/configs/AgentOccam.yml --task-config-dir <dir> --task-ids <id>"

build-skills:
	@echo "usage: ./scripts/build_skills.sh <site> <trajectory-dir> <output-dir>"

evolve-skills:
	@echo "usage: ./scripts/evolve_skills.sh <site-or-library-path>"

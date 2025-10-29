

# -------- Defaults / params --------
# Use Poetry to run Python by default; override with: make PY=python ingest
PY ?= poetry run python
LEAGUE ?= NBA
BOOKS ?= pinnacle,draftkings
EV ?= 0.01
KELLY ?= 0.25
DECISION_MIN ?= 30

# Lowercase version of LEAGUE for module names like lib.ingest.nba_odds
LEAGUE_MOD := $(shell echo $(LEAGUE) | tr '[:upper:]' '[:lower:]')

.PHONY: ingest features labels train backtest smoke_nba help

# -------- Targets --------
ingest:
	$(PY) -m lib.ingest.$(LEAGUE_MOD)_schedule --league $(LEAGUE)
	$(PY) -m lib.ingest.$(LEAGUE_MOD)_results --league $(LEAGUE)
	$(PY) -m lib.ingest.$(LEAGUE_MOD)_odds --league $(LEAGUE)

features:
	$(PY) -m lib.featurization.build_features --league $(LEAGUE)

labels:
	$(PY) -m lib.labeling.build_labels --league $(LEAGUE) --decision_offset_min $(DECISION_MIN)

train:
	$(PY) -m lib.modeling.train --league $(LEAGUE)

backtest:
	$(PY) -m lib.eval.backtest --league $(LEAGUE) --books $(BOOKS) --ev_threshold $(EV) --kelly_fraction $(KELLY)

# One-shot sanity for NBA pregame flow
smoke_nba:
	$(MAKE) ingest LEAGUE=NBA
	$(MAKE) features LEAGUE=NBA
	$(MAKE) labels LEAGUE=NBA DECISION_MIN=30
	$(MAKE) train LEAGUE=NBA
	$(MAKE) backtest LEAGUE=NBA EV=0.01 KELLY=0.25

# Show usage
help:
	@echo "Targets:"
	@echo "  make ingest       LEAGUE=NBA"
	@echo "  make features     LEAGUE=NBA"
	@echo "  make labels       LEAGUE=NBA DECISION_MIN=30"
	@echo "  make train        LEAGUE=NBA"
	@echo "  make backtest     LEAGUE=NBA EV=0.01 KELLY=0.25 BOOKS=pinnacle,draftkings"
	@echo "  make smoke_nba"
	@echo ""
	@echo "Params (with defaults):"
	@echo "  PY=$(PY)"
	@echo "  LEAGUE=$(LEAGUE)  (module uses lowercase: $(LEAGUE_MOD))"
	@echo "  BOOKS=$(BOOKS)"
	@echo "  EV=$(EV)   KELLY=$(KELLY)   DECISION_MIN=$(DECISION_MIN)"

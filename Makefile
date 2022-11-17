.PHONY: generate profile clear cython_clear

generate:
	@make clear
	@python -m numeth.setup

profile:
	@make clear
	@python -m numeth.setup --profile

notrace:
	@make clear
	@python -m numeth.setup --notrace

hardcore:
	make clear
	@python -m numeth.setup --hardcore

hardcoreprofile:
	make clear
	@python -m numeth.setup --hardcore --profile

clear:
	@echo "Cleaning all.."
	@rm -f numeth/*.so
	@rm -f numeth/*.html
	@rm -R -f numeth/build
	@rm -R -f numeth/__pycache__
	@echo "Cleaned."

cython_clear:
	@echo "Cleaning all.."
	@rm -f numeth/*.c
	@rm -f numeth/*.so
	@rm -f numeth/*.html
	@rm -R -f numeth/build
	@rm -R -f numeth/__pycache__
	@echo "Cleaned."

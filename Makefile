# Makefile for TensorFlow/Gemma Local AI System

.PHONY: help setup test train inference benchmark clean

help:
	@echo "TensorFlow/Gemma Local AI System"
	@echo "================================"
	@echo "  make setup      - Set up TensorFlow environment"
	@echo "  make test       - Run all tests"
	@echo "  make train      - Run LoRA fine-tuning"
	@echo "  make inference  - Start RAG inference pipeline"
	@echo "  make benchmark  - Run performance benchmarks"
	@echo "  make monitor    - Monitor system resources"
	@echo "  make clean      - Clean temporary files"

setup:
	conda env create -f environment.yml
	conda activate tf_gemma
	python scripts/verify_tf_metal.py

test:
	python scripts/test_pipeline_tf.py

train:
	python scripts/prepare_finetuning_tf.py
	python scripts/finetune_gemma_lora.py

inference:
	python scripts/inference_pipeline_tf.py

benchmark:
	python -c "from scripts.inference_pipeline_tf import GemmaRAGPipeline; p = GemmaRAGPipeline(); p.benchmark_performance()"

monitor:
	python scripts/monitor_tf_resources.py

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf logs/*.log
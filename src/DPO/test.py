from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

AutoModelForCausalLM.register("custom_model")(AutoModelForCausalLM)
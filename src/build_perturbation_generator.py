import textattack


# construct perturbation generator
def construct_perturbation_generator(model, tokenizer, perturbation):
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    build_attack = getattr(textattack.attack_recipes, perturbation)
    attack_generator = build_attack.build(model_wrapper=model_wrapper)
    return attack_generator

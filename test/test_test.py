from src.test import ModelEvaluation

def test_model_evaluation():
    model_evaluation = ModelEvaluation()

    model_evaluation.exec_eval()

    assert(isinstance(model_evaluation.score, int))
    assert(model_evaluation.final_score > 0)
    assert(model_evaluation.final_score >= model_evaluation.middle_score)

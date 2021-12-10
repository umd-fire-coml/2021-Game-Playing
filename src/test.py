import SuperMarioBros_Dataset from utils

class ModelEvaluation:
    lvl = 1
    vers = 1
    step_ct = 50
    model_thing = SuperMarioBros_Dataset(lvl, vers)
    final_score = 0
    rewards = []

    def eval(self, step_ct_max = 1000):
        # start moving so as to avoid reward field being None
        model_thing.simulate_steps(1)
        rewards.append(model_thing.get_reward())
        for step_ctr in range(step_ct_max):
            model_thing.simulate_steps(step_ct)
            rewards.append(model_thing.get_reward())
            pass
        pass

    def exec_eval(self):
        eval()
        final_score = rewards[-1] - rewards[0]
        middle_score = rewards[-len(rewards) / 2] - rewards[0]
        pass

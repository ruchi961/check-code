class SelfConsistency:
    def __init__(self,llm) -> None:
        pass
    def generate(self):
        li=[{"model_response":"Yes, there are new potatoes in the dish. Let's break down the reasoning:\n\n1. All the potatoes in the dish are fit to eat.\n2. If a potato is new, it hasn't been boiled, according to the first statement.\n3. If a potato hasn't been boiled, it is not fit to eat, according to the third statement.\n\nFrom (1) and (3), we can conclude that all the potatoes in the dish are not new. But, from (2), we know that there are new potatoes in the dish, which are the ones that haven't been boiled. \n\nTherefore, the answer is yes, there are new potatoes in the dish.","pvg_response":"Let's break down the statements:\n\n1. No potatoes of mine, that are new, have been boiled. (Only new potatoes have not been boiled.)\n2. All my potatoes in this dish are fit to eat.\n3. No unboiled potatoes of mine are fit to eat.\n\nFrom statement 3, we know that all unboiled potatoes are not fit to eat.\n\nNow, let's combine statements 1 and 3:\n\n* Since all new potatoes have not been boiled (statement 1), they are unboiled.\n* Since all unboiled potatoes are not fit to eat (statement 3), all new potatoes in the dish are not fit to eat.\n\nHowever, statement 2 says that all potatoes in the dish are fit to eat. This creates a contradiction with the conclusion we drew in the previous step.\n\nTherefore, the only way to resolve this contradiction is to conclude that there are no new potatoes in the dish. If there were new potatoes, then they would not be fit to eat (as per the contradiction we drew earlier), but statement 2 says that all potatoes in the dish are fit to eat.\n\nSo, the answer is: No, there are no new potatoes in this dish.","SelfConsistency_response":"Let's break down the statements:\n\n1. No potatoes of mine, that are new, have been boiled. (Only new potatoes have not been boiled.)\n2. All my potatoes in this dish are fit to eat.\n3. No unboiled potatoes of mine are fit to eat.\n\nFrom statement 3, we know that all unboiled potatoes are not fit to eat.\n\nNow, let's combine statements 1 and 3:\n\n* Since all new potatoes have not been boiled (statement 1), they are unboiled.\n* Since all unboiled potatoes are not fit to eat (statement 3), all new potatoes in the dish are not fit to eat.\n\nHowever, statement 2 says that all potatoes in the dish are fit to eat. This creates a contradiction with the conclusion we drew in the previous step.\n\nTherefore, the only way to resolve this contradiction is to conclude that there are no new potatoes in the dish. If there were new potatoes, then they would not be fit to eat (as per the contradiction we drew earlier), but statement 2 says that all potatoes in the dish are fit to eat.\n\nSo, the answer is: No, there are no new potatoes in this dish."}]
        return li
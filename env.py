class Env():
    def __init__(self):
        self.length = 32

    def step(self, action):
        self.sequence.append(action)
        done = False
        reward = 0

        if len(self.sequence) == self.length:
            done = True
            reward = 1

        return (action, reward, done, None)

    def reset(self):
        self.sequence = []
        return None


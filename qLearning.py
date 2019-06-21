import random

import numpy as np

class Env():
    def __init__(self):
        self.pos = 0;
        self.end = 9;
        self.actions = [0, 1, 2];
        self.stateCount = 10;
        self.actionCount = 3;

    def reset(self):
        self.pos = 0;
        self.done = False;
        return 0, 0, False;

    def step(self, action):
        if action == 0:
            print(action)
            a = random.randint(0,10);
            b = random.randint(0,10);
            c = a + b;
            #odpoved = input("Zadaj spravnu odpoved: " + str(a) + " + " + str(b) + " = ");
            odpoved = str(c)

            if str(c) == odpoved:
                reward = -0.5;
                self.pos += 1;
            else:
                reward = 1;
                self.pos += 1;

        elif action == 1:
            print(action)
            a = random.randint(0, 10);
            b = random.randint(0, 10);

            if a > b:
                c = a - b;
                #odpoved = input("Zadaj spravnu odpoved: " + str(a) + " - " + str(b) + " = ");
                odpoved = str(c)
            else:
                c = b - a;
                #odpoved = input("Zadaj spravnu odpoved: " + str(b) + " - " + str(a) + " = ");
                odpoved = str(c)
            if str(c) == odpoved:
                reward = -0.5;
                self.pos += 1;
            else:
                reward = 1;
                self.pos += 1;

        elif action == 2:
            print(action)
            a = random.randint(0, 5);
            b = random.randint(0, 5);
            c = a * b;
            #odpoved = input("Zadaj spravnu odpoved: " + str(a) + " * " + str(b) + " = ");
            odpoved = str(c-100)
            if str(c) == odpoved:
                reward = -0.5;
                self.pos += 1;
            else:
                reward = 1;
                self.pos += 1;

        done = self.pos == 9;
        nextState = self.pos;
        return nextState, reward, done;

    def randomAction(self):
        return np.random.choice(self.actions);

env = Env()

#qtable = np.random.rand(env.stateCount, env.actionCount).tolist()

qtable = np.zeros([env.stateCount, env.actionCount]).tolist()


# hyperparameters
epochs = 20
gamma = 0.8
epsilon = 0.5
decay = 0.1

# training loop
for i in range(epochs):
    state, reward, done = env.reset()
    steps = 0
    print("\n")
    while not done:
        #print("epoch #", i + 1, "/", epochs)

        # count steps to finish game
        steps += 1

        # act randomly sometimes to allow exploration
        if np.random.uniform() < epsilon:
            action = env.randomAction()
        else:
            action = qtable[state].index(max(qtable[state]))
        # if not select max action in Qtable (act greedy)

        next_state, reward, done = env.step(action)
        #print(next_state,reward,done)

        qtable[state][action] = reward + gamma * max(qtable[next_state])
        print(qtable)
        # update state
        state = next_state
        # The more we learn, the less we take random actions

        if steps == 10:
            done = True;

    epsilon -= decay * epsilon

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

import random
import numpy as np
import os

app = Flask(__name__)
CORS(app)

port = int(os.getenv('PORT', '3000'))

@app.route('/', methods = ['GET'])
def start():
    return 'Hello'

@app.route('/getNumberAndAction', methods = ['POST'])

class Env():
    def __init__(self):
        self.pos = 0;
        self.end = 10;
        self.actions = [0, 1, 2];
        self.stateCount = 10;
        self.actionCount = 3;

    def reset(self):
        self.pos = 0;
        self.done = False;
        return 0, 0, False;

    def step(self, action):

        a = random.randint(0, 10);
        b = random.randint(0, 10);

        if b > a:
            pom = b
            b = a
            a = pom

        if action == 0:
            print(action)
            c = a + b;
            #odpoved = input("Zadaj spravnu odpoved: " + str(a) + " + " + str(b) + " = ");

            odpoved = str(c)
            print(str(a) + " + " + str(b) + " = " + str(odpoved));

            if str(c) == odpoved:
                reward = -0.5;
                self.pos += 1;
            else:
                reward = 1;
                self.pos += 1;


        elif action == 1:
            print(action)
            if a >= b:
                c = a - b;
                #odpoved = input("Zadaj spravnu odpoved: " + str(a) + " - " + str(b) + " = ");
                odpoved = str(c)
                print(str(a) + " - " + str(b) + " = " + str(odpoved));


            else:
                c = b - a;
                #odpoved = input("Zadaj spravnu odpoved: " + str(b) + " - " + str(a) + " = ");
                odpoved = str(c)
                print(str(b) + " - " + str(a) + " = " + str(odpoved));

            if str(c) == odpoved:
                reward = -0.5;
                self.pos += 1;
            else:
                reward = 1;
                self.pos += 1;

        elif action == 2:
            print(action)
            c = a * b;
            #odpoved = input("Zadaj spravnu odpoved: " + str(a) + " * " + str(b) + " = ");

            odpoved = str(c-100)
            print(str(a) + " * " + str(b) + " = " + str(odpoved));

            if str(c) == odpoved:
                reward = -0.5;
                self.pos += 1;
            else:
                reward = 1;
                self.pos += 1;

        elif action == 3:
            print(action)
            c = a * b;
            c = c / a;
            odpoved = str(c)
            print(str(c) + " / " + str(a) + " = " + str(odpoved));

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


def qLearning():
    env = Env()

    #qtable = np.random.rand(env.stateCount, env.actionCount).tolist()

    #TODO - create text file with zeros q table
    qtable = np.zeros([env.stateCount, env.actionCount]).tolist()


    # hyperparameters
    epochs = 8
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

            #TODO - load qtable from txt

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

qLearning()

if(__name__) == '_main_':
    app.run(host='0.0.0.0', port=port)
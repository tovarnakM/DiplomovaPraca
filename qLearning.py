from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

import random
import numpy as np
import os

app = Flask(__name__)
CORS(app)

port = int(os.getenv('PORT', '3000'))

class Parameter():
    steps = 0
    epoch = 0
    pos = 0


@app.route('/', methods=['GET'])
def start():
    return 'Hello'


@app.route('/getZeros', methods=['GET'])
def qZeros():
    env = Env()
    qtable = np.zeros([env.stateCount, env.actionCount]).tolist()
    saveQtable(qtable)
    print(Parameter.steps)
    return jsonify({"res": "success"})


@app.route('/train', methods=['GET'])
def train():
    qtable = loadQtable()

    return jsonify({"res": "success"})



def loadQtable():
    x = []
    y = []
    with open('qtable.txt', 'r') as f:
        for line in f:
            if line:  # avoid blank lines
                x.append(float(line.strip()))
                if len(x) == 4:
                    y.append(x)
                    x = []
    return y

def saveQtable(data):
    np.savetxt('qtable.txt', data, delimiter='\n')

class Env():
    def __init__(self):
        self.end = 10;
        self.actions = [0, 1, 2, 3];
        self.stateCount = 10;
        self.actionCount = 4;

    def reset(self):
        Parameter.pos = 0;
        self.done = False;
        return 0, 0, False;

    def getNumbersForExample(self):

        a = random.randint(1, 10);
        b = random.randint(1, 10);

        if b > a:
            pom = b
            b = a
            a = pom

        return a, b

    def step(self, action, a, b):

        if action == 0:
            print(action)
            c = a + b;
            # odpoved = input("Zadaj spravnu odpoved: " + str(a) + " + " + str(b) + " = ");

            odpoved = str(c)
            print(str(a) + " + " + str(b) + " = " + str(odpoved));

            if str(c) == odpoved:
                reward = -0.5;
                Parameter.pos += 1;
            else:
                reward = 1;
                Parameter.pos += 1;


        elif action == 1:
            print(action)
            if a >= b:
                c = a - b;
                # odpoved = input("Zadaj spravnu odpoved: " + str(a) + " - " + str(b) + " = ");
                odpoved = str(c)
                print(str(a) + " - " + str(b) + " = " + str(odpoved));


            else:
                c = b - a;
                # odpoved = input("Zadaj spravnu odpoved: " + str(b) + " - " + str(a) + " = ");
                odpoved = str(c)
                print(str(b) + " - " + str(a) + " = " + str(odpoved));

            if str(c) == odpoved:
                reward = -0.5;
                Parameter.pos += 1;
            else:
                reward = 1;
                Parameter.pos += 1;

        elif action == 2:
            print(action)
            c = a * b;
            # odpoved = input("Zadaj spravnu odpoved: " + str(a) + " * " + str(b) + " = ");

            odpoved = str(c - 100)
            print(str(a) + " * " + str(b) + " = " + str(odpoved));

            if str(c) == odpoved:
                reward = -0.5;
                Parameter.pos += 1;
            else:
                reward = 1;
                Parameter.pos += 1;

        elif action == 3:
            print(action)
            c = a * b;
            c = c / a;
            odpoved = str(c)
            print(str(c) + " / " + str(a) + " = " + str(odpoved));

            if str(c) == odpoved:
                reward = -0.5;
                Parameter.pos += 1;
            else:
                reward = 1;
                Parameter.pos += 1;

        done = Parameter.pos == 9;
        nextState = Parameter.pos;
        return nextState, reward, done;

    def randomAction(self):
        return np.random.choice(self.actions);


def qLearning():
    env = Env()

    # qtable = np.random.rand(env.stateCount, env.actionCount).tolist()

    # TODO - create text file with zeros q table
    qtable = np.zeros([env.stateCount, env.actionCount]).tolist()

    # hyperparameters
    epochs = 8
    gamma = 0.8
    epsilon = 0.5
    decay = 0.1


    # training loop
    if Parameter.epoch <= epochs:
        state, reward, done = env.reset()
        Parameter.steps = 0
        print("\n")
        if not done:
            # print("epoch #", i + 1, "/", epochs)

            # count steps to finish game
            Parameter.steps += 1

            # TODO - load qtable from txt
            qtable = loadQtable()

            # act randomly sometimes to allow exploration
            if np.random.uniform() < epsilon:
                action = env.randomAction()
            else:
                action = qtable[state].index(max(qtable[state]))
            # if not select max action in Qtable (act greedy)

            a,b = env.getNumbersForExample()



            next_state, reward, done = env.step(action, a, b)
            # print(next_state,reward,done)

            qtable[state][action] = reward + gamma * max(qtable[next_state])
            saveQtable(qtable)
            print(qtable)
            # update state
            state = next_state
            # The more we learn, the less we take random actions

            if Parameter.steps == 10:
                done = True;

        epsilon -= decay * epsilon

qLearning()

if (__name__) == '_main_':
    app.run(host='0.0.0.0', port=port)

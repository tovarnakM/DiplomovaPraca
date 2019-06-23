from flask import Flask, jsonify, request
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
    state = 0
    epsilon = 0.6
    newEpoch = 0
    done = False


@app.route('/', methods=['GET'])
def start():
    return 'Hello world'


@app.route('/getZeros', methods=['GET'])
def qZeros():
    Parameter.steps = 0
    Parameter.epoch = 0
    Parameter.pos = 0
    Parameter.state = 0
    Parameter.epsilon = 0.5
    env = Env()
    qtable = np.zeros([env.stateCount, env.actionCount]).tolist()
    saveQtable(qtable)
    return jsonify({"res": "success"})


@app.route('/train', methods=['GET'])
def train():
    env = Env()
    # hyperparameters
    epochs = 3
    decay = 0.1

    if Parameter.epoch <= epochs:
        if Parameter.epoch == 0 and Parameter.steps == 0:
            Parameter.state, reward, Parameter.done = env.reset()
            Parameter.steps = 0

        # training loop
        if Parameter.epoch <= epochs and Parameter.epoch != Parameter.newEpoch:
            Parameter.newEpoch = Parameter.epoch
            Parameter.state, reward, Parameter.done = env.reset()
            Parameter.steps = 0
            print("\n")

        if not Parameter.done:
            # count steps to finish game
            Parameter.steps += 1

            qtable = loadQtable()

            # act randomly sometimes to allow exploration
            if np.random.uniform() < Parameter.epsilon:
                action = env.randomAction()
            else:
                action = qtable[Parameter.state].index(max(qtable[Parameter.state]))
            # if not select max action in Qtable (act greedy)

            a, b = env.getNumbersForExample()
            if action == 3:
                c = a * b
                a = c

            Parameter.epsilon -= decay * Parameter.epsilon

            return jsonify({"action": int(action), "a": int(a), "b": int(b)})
    return jsonify({"resp": "succes"})


@app.route('/evaluate', methods=['POST'])
def evaluate():
    env = Env()
    gamma = 0.8

    req_data = request.get_json()

    next_state, reward, Parameter.done, c = env.step(req_data['action'], req_data['a'], req_data['b'],
                                                  req_data['userResponse'])
    # print(next_state,reward,done)


    print(Parameter.state)
    print(req_data['action'])
    print(next_state)

    qtable = loadQtable()
    qtable[Parameter.state][req_data['action']] = reward + gamma * max(qtable[Parameter.state])
    saveQtable(qtable)

    # update state
    Parameter.state = next_state
    # The more we learn, the less we take random actions

    if Parameter.steps == 4:
        Parameter.epoch += 1
        Parameter.done = True;

    if c == req_data['userResponse']:
        return jsonify({"res": "success"})
    return jsonify({"res": "fail"})

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
        self.end = 4;
        self.actions = [0, 1, 2, 3];
        self.stateCount = 4;
        self.actionCount = 4;

    def reset(self):
        Parameter.pos = 0;
        Parameter.done = False;
        return 0, 0, False;

    def getNumbersForExample(self):

        a = random.randint(1, 10);
        b = random.randint(1, 10);

        if b > a:
            pom = b
            b = a
            a = pom

        return a, b

    def step(self, action, a, b, userResponse):

        if action == 0:
            c = a + b;

            if str(c) == str(userResponse):
                reward = -0.5;
                Parameter.pos += 1;
            else:
                reward = 1;
                Parameter.pos += 1;

        elif action == 1:

            c = a - b

            if str(c) == str(userResponse):
                reward = -0.5;
                Parameter.pos += 1;
            else:
                reward = 1;
                Parameter.pos += 1;

        elif action == 2:

            c = a * b;

            if str(c) == str(userResponse):
                reward = -0.5;
                Parameter.pos += 1;
            else:
                reward = 1;
                Parameter.pos += 1;

        elif action == 3:

            c = a / b;

            if str(c) == str(userResponse):
                reward = -0.5;
                Parameter.pos += 1;
            else:
                reward = 1;
                Parameter.pos += 1;

        Parameter.done = Parameter.pos == 4;
        nextState = Parameter.pos;
        return nextState, reward, Parameter.done, c;

    def randomAction(self):
        return np.random.choice(self.actions);

if (__name__) == '_main_':
    app.run(host='0.0.0.0', port=port)

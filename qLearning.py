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
    epsilon = 0.5
    done = False
    newEpoch = 0


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
    Parameter.done = False
    env = Env()
    qtable = np.zeros([env.stateCount, env.actionCount]).tolist()
    saveQtable(qtable)
    print(Parameter.steps)
    return jsonify({"res": "success"})


@app.route('/train', methods=['GET'])
def train():
    env = Env()
    # hyperparameters
    epochs = 8
    decay = 0.1
    action = 1

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

        print(action)
        print(Parameter.state)
        print("epoch: " + str(Parameter.epoch))
        print("steps: " + str(Parameter.steps))

    Parameter.epsilon -= decay * Parameter.epsilon

    return jsonify({"action": int(action), "a": int(a), "b": int(b)})


@app.route('/evaluate', methods=['POST'])
def evaluate():
    env = Env()
    gamma = 0.8

    req_data = request.get_json()

    next_state, reward, Parameter.done = env.step(req_data['action'], req_data['a'], req_data['b'], req_data['userResponse'])
    # print(next_state,reward,done)

    qtable = loadQtable()
    qtable[Parameter.state][req_data['action']] = reward + gamma * max(qtable[next_state])

    print(Parameter.state)
    print(req_data['action'])
    print(next_state)
    print("epoch: " + str(Parameter.epoch))
    print("steps: " + str(Parameter.steps))

    # update state
    Parameter.state = next_state
    # The more we learn, the less we take random actions

    if Parameter.steps == 10:
        Parameter.epoch += 1
        Parameter.done = True;

    return jsonify({"response": "success"})

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
        self.end = 3;
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

        Parameter.done = Parameter.pos == 3;
        nextState = Parameter.pos;
        return nextState, reward, Parameter.done;

    def randomAction(self):
        return np.random.choice(self.actions);


if (__name__) == '_main_':
    app.run(host='0.0.0.0', port=port)

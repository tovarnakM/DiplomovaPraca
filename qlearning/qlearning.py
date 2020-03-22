from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient

import random
import numpy as np
import os

app = Flask(__name__)
CORS(app)

client = MongoClient('mongodb://localhost:27017')
db = client.students

class Parameter():
    steps = 0
    epoch = 0
    pos = 0
    state = 0
    epsilon = 0.7
    newEpoch = 0
    done = False


@app.route('/', methods=['GET'])
def start():
    user = db.students.find_one({'email': "ferko"})
    return user


@app.route('/getZeros', methods=['GET'])
def qZeros():
    Parameter.steps = 0
    Parameter.epoch = 0
    Parameter.pos = 0
    Parameter.state = 0
    Parameter.epsilon = 0.7

    email = request.args.get('email')
    response = initializeTable(email)
    print(email)
    return jsonify({"res": response})


@app.route('/train', methods=['GET'])
def train():
    env = Env()
    # hyperparameters
    epochs = 9
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

            email = request.args.get('email')
            qtable = loadQtable(email)

            if qtable is None:
                return "User Not found"

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

            return jsonify({"res": "ok", "action": int(action), "a": int(a), "b": int(b), "epoch": Parameter.epoch + 1, "step": Parameter.steps})
    return jsonify({"res": "success"})


@app.route('/evaluate', methods=['POST'])
def evaluate():
    env = Env()
    gamma = 0.8

    req_data = request.get_json()

    next_state, reward, Parameter.done, c = env.step(req_data['action'], req_data['a'], req_data['b'],
                                                     req_data['userResponse'])
    email = req_data['email']
    qtable = loadQtable(email)

    qtable[Parameter.state][req_data['action']] = reward + gamma * max(qtable[Parameter.state])

    db.students.find_one_and_update(
        {"email": email},
        {"$set":
             {'qTable': qtable}
         }, upsert=True
    )

    # update state
    Parameter.state = next_state
    # The more we learn, the less we take random actions

    if Parameter.steps == 7:
        Parameter.epoch += 1
        Parameter.done = True;

    if c == req_data['userResponse']:
        return jsonify({"res": "success"})
    return jsonify({"res": "fail"})


def loadQtable(email):
    user = db.students.find_one({'email': email})
    if user is None:
        return None
    return user.get('qTable')


def initializeTable(email):
    user = db.students.find_one({'email': email})
    if user is None:
        env = Env()
        students = {
            'email': email,
            'qTable': np.zeros([env.stateCount, env.actionCount]).tolist()
        }
        db.students.insert_one(students)
        return "New user added!"
    else:
        db.students.find_one_and_update(
            {"email": email},
            {"$set":
                 {'qTable': loadQtable(email)}
             }, upsert=True
        )
        return "Learning with existing user!"


class Env():
    def __init__(self):
        self.end = 4;
        self.actions = [0, 1, 2, 3];
        self.stateCount = 7;
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

        Parameter.done = Parameter.pos == 7;
        nextState = Parameter.pos;
        return nextState, reward, Parameter.done, c;

    def randomAction(self):
        return np.random.choice(self.actions);



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


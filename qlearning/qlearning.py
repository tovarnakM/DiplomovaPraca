from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient

import random
import numpy as np
import os

app = Flask(__name__)
CORS(app)

client = MongoClient('mongodb://username:user123@ds011231.mlab.com:11231/heroku_kkr81g5x?retryWrites=false')
db = client.heroku_kkr81g5x

class Parameter():
    # steps = 0
    # epoch = 0
    # pos = 0
    # state = 0
    epsilon = 0.7
    # newEpoch = 0
    # done = False


@app.route('/', methods=['GET'])
def start():
    return 'Hello world'


@app.route('/getZeros', methods=['GET'])
def qZeros():
    steps = 0
    epoch = 0
    pos = 0
    state = 0
    epsilon = 0.7
    newEpoch = 0
    done = False

    Parameter.epsilon = 0.7

    learningData = [steps, epoch, pos, state, newEpoch, done]

    email = request.args.get('email')
    response = initializeTable(email, learningData)

    return jsonify({"res": response})


@app.route('/train', methods=['GET'])
def train():
    env = Env()
    # hyperparameters
    epochs = 9
    decay = 0.1

    #learningData = [steps, epoch, pos, state, newEpoch, done]
    email = request.args.get('email')
    user = db.students.find_one({'email': email})
    learningData = user.get('learningData')

    if learningData[1] <= epochs:
        if learningData[1] == 0 and learningData[0] == 0:
            learningData[3], reward, learningData[5] = env.reset(learningData)
            learningData[0] = 0

        # training loop
        if learningData[1] <= epochs and learningData[1] != learningData[4]:
            learningData[4] = learningData[1]
            learningData[3], reward, learningData[5] = env.reset(learningData)
            learningData[0] = 0

        if not learningData[5]:
            # count steps to finish game
            learningData[0] += 1

            qtable = loadQtable(email)

            if qtable is None:
                return "User Not found"

            # act randomly sometimes to allow exploration
            if np.random.uniform() < Parameter.epsilon:
                action = env.randomAction()
            else:
                action = qtable[learningData[3]].index(max(qtable[learningData[3]]))
            # if not select max action in Qtable (act greedy)

            a, b = env.getNumbersForExample()
            if action == 3:
                c = a * b
                a = c

            Parameter.epsilon -= decay * Parameter.epsilon

            db.students.find_one_and_update(
                {"email": email},
                {"$set":
                     {'learningData': learningData}
                 }, upsert=True
            )

            return jsonify({"res": "ok", "action": int(action), "a": int(a), "b": int(b), "epoch": learningData[1] + 1, "step": learningData[0], "data": learningData})
    return jsonify({"res": "success"})


@app.route('/evaluate', methods=['POST'])
def evaluate():
    env = Env()
    gamma = 0.8

    req_data = request.get_json()
    email = req_data['email']
    user = db.students.find_one({'email': email})
    learningData = user.get('learningData')

    next_state, reward, learningData[5], c = env.step(req_data['action'], req_data['a'], req_data['b'],
                                                     req_data['userResponse'], learningData)

    qtable = loadQtable(email)

    qtable[learningData[3]][req_data['action']] = reward + gamma * max(qtable[learningData[3]])

    # update state
    learningData[3] = next_state
    # The more we learn, the less we take random actions

    if learningData[0] == 7:
        learningData[1] += 1
        learningData[5] = True;

    db.students.find_one_and_update(
        {"email": email},
        {"$set":
             {'qTable': qtable, 'learningData': learningData}
         }, upsert=True
    )

    if c == req_data['userResponse']:
        return jsonify({"res": "success"})
    return jsonify({"res": "fail"})


def loadQtable(email):
    user = db.students.find_one({'email': email})
    if user is None:
        return None
    return user.get('qTable')


def initializeTable(email, learningData):
    user = db.students.find_one({'email': email})
    if user is None:
        env = Env()
        students = {
            'email': email,
            'qTable': np.zeros([env.stateCount, env.actionCount]).tolist(),
            'learningData': learningData
        }
        db.students.insert_one(students)
        return "New user added!"
    else:
        db.students.find_one_and_update(
            {"email": email},
            {"$set":
                 {'qTable': loadQtable(email), 'learningData': learningData}
             }, upsert=True
        )
        return "Learning with existing user!"


class Env():
    def __init__(self):
        self.end = 4;
        self.actions = [0, 1, 2, 3];
        self.stateCount = 7;
        self.actionCount = 4;

    def reset(self, learningData):
        learningData[2] = 0;
        learningData[5] = False;
        return 0, 0, False;

    def getNumbersForExample(self):

        a = random.randint(1, 10);
        b = random.randint(1, 10);

        if b > a:
            pom = b
            b = a
            a = pom

        return a, b

    def step(self, action, a, b, userResponse, learningData):

        if action == 0:
            c = a + b;

            if str(c) == str(userResponse):
                reward = -0.5;
                learningData[2] += 1;
            else:
                reward = 1;
                learningData[2] += 1;

        elif action == 1:

            c = a - b

            if str(c) == str(userResponse):
                reward = -0.5;
                learningData[2] += 1;
            else:
                reward = 1;
                learningData[2] += 1;

        elif action == 2:

            c = a * b;

            if str(c) == str(userResponse):
                reward = -0.5;
                learningData[2] += 1;
            else:
                reward = 1;
                learningData[2] += 1;

        elif action == 3:

            c = a / b;

            if str(c) == str(userResponse):
                reward = -0.5;
                learningData[2] += 1;
            else:
                reward = 1;
                learningData[2] += 1;

        learningData[5] = learningData[2] == 7;
        nextState = learningData[2];
        return nextState, reward, learningData[5], c;

    def randomAction(self):
        return np.random.choice(self.actions);


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


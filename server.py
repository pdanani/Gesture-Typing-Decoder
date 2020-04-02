from scipy.spatial import distance  # remove

from flask import Flask, request
from flask import render_template
import time
import json
import numpy as np
import math
from scipy.interpolate import interp1d
import heapq

app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240,
               170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50,
               120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    sample_points_X, sample_points_Y = [], []
    # TODO: Start sampling (12 points)
    # ediff1d gives difference between consecutive elements of the array
    # we find the distance between coordinates and find the cumulative sum
    distance = np.cumsum(np.sqrt(np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2))
    # basically when words like mm or ii have no path / little path, use centroid
    if (distance[-1] == 0):
        for i in range(100):
            sample_points_X.append(points_X[0])
            sample_points_Y.append(points_Y[0])
    else:
        # get the proportion of line segments
        distance = distance / distance[-1]
        # scale the points to get linear interpolations along the path
        fx, fy = interp1d(distance, points_X), interp1d(distance, points_Y)
        # generate 100 equidistant points on normalized line
        alpha = np.linspace(0, 1, 100)
        # use the interpolation function to translate from normalized to real plane
        x_regular, y_regular = fx(alpha), fy(alpha)
        sample_points_X = x_regular.tolist()
        sample_points_Y = y_regular.tolist()
    return sample_points_X, sample_points_Y


# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def do_pruning(gesture_sample_points_X, gesture_sample_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    threshold = 11
    # TODO: Do pruning (12 points)

    height = np.max(gesture_sample_points_Y) - np.min(gesture_sample_points_Y)
    width = np.max(gesture_sample_points_X) - np.min(gesture_sample_points_X)
    L = 110
    count = 0

    if (width > 0 or height > 0):
        s = L / max(width, height)
    else:
        s = 0

    if width < height:
        height = L
        width *= s
    else:
        height *= s
        width = L
    centroidw = width / 2
    centroidh = height / 2

    copy_sample_points_X = gesture_sample_points_X.copy()
    copy_sample_points_Y = gesture_sample_points_Y.copy()
    for j in range(len(gesture_sample_points_X)):
        copy_sample_points_X[j] = (copy_sample_points_X[j] - centroidw) * s
        copy_sample_points_Y[j] = (copy_sample_points_Y[j] - centroidh) * s

    for x, y in zip(template_sample_points_X, template_sample_points_Y):
        height = np.max(y) - np.min(y)
        width = np.max(x) - np.min(x)
        if (width > 0 or height > 0):
            s = L / max(width, height)
        elif (width == 0 and height == 0):
            s = 0

        start_distance = math.sqrt(
            ((copy_sample_points_X[0] - (x[0] - centroidw) * s) ** 2) + (
                        (copy_sample_points_Y[0] - (y[0] - centroidh) * s) ** 2))
        end_distance = math.sqrt(
            ((copy_sample_points_X[-1] - (x[-1] - centroidw) * s) ** 2) + (
                        (copy_sample_points_Y[-1] - (y[-1] - centroidh) * s) ** 2))
        if (start_distance <= threshold and end_distance <= threshold):
            valid_words.append(words[count])
            valid_template_sample_points_X.append(x)
            valid_template_sample_points_Y.append(y)
        count += 1

    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                     valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    shape_scores = []
    # TODO: Set your own L
    L = 110
    height = np.max(gesture_sample_points_Y) - np.min(gesture_sample_points_Y)
    width = np.max(gesture_sample_points_X) - np.min(gesture_sample_points_X)
    adi = 0

    if (width > 0 or height > 0):
        s = L / max(width, height)
    else:
        s = 0
    if width < height:
        height = L
        width *= s
    else:
        height *= s
        width = L
    centroidw = (width / 2)
    centroidh = (height / 2)
    copyx=gesture_sample_points_X.copy()
    copyy=gesture_sample_points_Y.copy()
    for j in range(len(gesture_sample_points_Y)):
        copyx[j] -= centroidw
        copyy[j] -= centroidh

        copyx[j] *= s
        copyy[j] *= s
    for x, y in zip(valid_template_sample_points_X, valid_template_sample_points_Y):
        height = np.max(y) - np.min(y)
        width = np.max(x) - np.min(x)
        if (width > 0 or height > 0):
            s = L / max(width, height)
        else:
            s = 0
        x_adjusted=[]
        y_adjusted=[]
        for i in x:
            x_adjusted.append(s*(i-centroidw))
        for j in y:
            y_adjusted.append(s*(j-centroidh ))

        counter = 0
        shape_scores.append(0)
        for a, b in zip(x_adjusted, y_adjusted):
            coord = np.array((a, b))
            gesture = np.array((copyx[counter], copyy[counter]))
            distance = (np.dot(coord - gesture, coord - gesture)) ** .5
            shape_scores[adi] += distance
            counter += 1
        shape_scores[adi] = shape_scores[adi] / len(y_adjusted)
        adi += 1

    return shape_scores


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                        valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    #location_scores = [[0] * 100] * len(valid_template_sample_points_X)
    location_scores=[]
    radius = 15
    first = True  # use this boolean to check if the first number is the lowest.
    # TODO: Calculate location scores (12 points)
    bestdistance = 0
    i = 0
    j = 0
    tempArr=[]
    for z in range(len(valid_template_sample_points_X)):  # for every valid word

        for x, y in zip(gesture_sample_points_X, gesture_sample_points_Y):
            gesturecoord = np.array((x, y))

            for k, l in zip(valid_template_sample_points_X[z], valid_template_sample_points_Y[z]):
                tempcoord = np.array((k, l))  # every point in the wordx template
                distance = (np.dot(gesturecoord - tempcoord, gesturecoord - tempcoord)) ** .5

                if first:
                    bestdistance = distance
                    first = False
                elif distance < bestdistance:
                    bestdistance = distance
            tempArr.append(bestdistance)
            first = True
        location_scores.append(tempArr)
        tempArr=[]


    sums=[]
    b=[]

    for i in range (len(location_scores)):
        sums.append(0)
        for j in range( len(location_scores[i])):
            num=max(location_scores[i][j]-radius,0) * (.1 * abs(j - 50) + .1)
            sums[i]=sums[i]+num
        j=0
    return sums


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = .54
    # TODO: Set your own location weight
    location_coef = .46

    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = 'the'
    # TODO: Set your own range.
    n = 3
    # TODO: Get the best word (12 points)
    max_scores=np.sort(integration_scores)
    # now take the top 3


    if(len(integration_scores)>=3):
        three=np.argsort(integration_scores)[:n]

        firstscore=integration_scores[three[0]]*(1-probabilities[valid_words[three[0]]])
        secondscore=integration_scores[three[1]]*(1-probabilities[valid_words[three[1]]])
        thirdscore=integration_scores[three[2]]*(1-probabilities[valid_words[three[2]]])
        if(firstscore<secondscore and firstscore<thirdscore):
            best_word=valid_words[three[0]]
        elif(secondscore<firstscore and secondscore<thirdscore):
            best_word=valid_words[three[1]]
        else:
            best_word=valid_words[three[2]]


    elif(len(integration_scores)==2):
        integration_scores[0]*=(1-probabilities[valid_words[0]])
        integration_scores[1]*=(1-probabilities[valid_words[1]])
        if(integration_scores[0]<integration_scores[1]):
            best_word=valid_words[0]
        else:
            best_word=valid_words[1]


    elif(len(integration_scores)==1):
        best_word=valid_words[0]
    else:
        best_word="No best word found."


    return best_word


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():
    start_time = time.time()
    data = json.loads(request.get_data())

    print(data)

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)
    # del gesture_sample_points_X[0][0]
    # del gesture_sample_points_Y[0][0]
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_sample_points_X,
                                                                                             gesture_sample_points_Y,
                                                                                             template_sample_points_X,
                                                                                             template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                                    valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y,
                                          valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'


if __name__ == "__main__":
    app.run()

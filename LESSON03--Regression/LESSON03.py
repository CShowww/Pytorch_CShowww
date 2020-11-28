import matplotlib as plt
import numpy as np
import torch
import pandas as pd

def compute_error_for_line_given_points(b,w,points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y-(w*x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b,w,points,lr):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((w * x) + b))
        w_gradient += -(2/N) * x * (y - ((w * x) + b))
    b_new = b - (lr * b_gradient)
    w_new = w - (lr * w_gradient)
    return [b_new,w_new]

def gradient_descent_runner(points,b,w,lr,iterations):
    for i in range(iterations):
        b,w = step_gradient(b,w,np.array(points),lr)
    return [b,w]


def run():
    points = np.genfromtxt('./l3_data.csv',delimiter=',')
    lr = 0.0001
    initial_b = 0
    initial_w = 0
    iterations = 1000
    print(f"Starting project descent at b = {initial_b}, w = {initial_w},error = {compute_error_for_line_given_points(initial_b,initial_w,points)}")
    [b,w] = gradient_descent_runner(points,initial_b,initial_w,lr,iterations)
    print(f"running...")
    print(f"After project descent at b = {b}, w = {w},error = {compute_error_for_line_given_points(b,w,points)}")
    print('\nb:{}，w:{}'.format(b, w))

if __name__ == '__main__':
    run()

import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_LEN_GOAL = 30
MOVE_PENALTY = 0.01

def collision_with_apple(apple_position, score):
	apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
	score += 1
	return apple_position, score

def collision_with_boundaries(snake_head):
	if snake_head[0]>=500 or snake_head[0]<0 or snake_head[1]>=500 or snake_head[1]<0 :
		return 1
	else:
		return 0

def collision_with_self(snake_position):
	snake_head = snake_position[0]
	if snake_head in snake_position[1:]:
		return 1
	else:
		return 0


class Snake(gym.Env):

	def __init__(self):
		super(Snake, self).__init__()
		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using discrete actions:
		self.action_space = spaces.Discrete(4)
		self.gameOver = False
		# Example for using image as input (channel-first; channel-last also works):
		self.observation_space = spaces.Box(low=-500, high=500,
											shape=(5+SNAKE_LEN_GOAL,), dtype=np.float32)
		self.prev_actions = []
		self.prev_reward = 0
		self.prev_button_direction = 1
		self.apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
		self.snake_position = [[250,250],[240,250],[230,250]]
		self.snake_head = self.snake_position[0]
		self.score = 0
		self.img = np.zeros((500,500,3),dtype='uint8')

	def step(self, action):
		self.prev_actions.append(action)

		t_end = time.time() + 0.05
		while time.time() < t_end:
			key = cv2.waitKeyEx(1) & 0xFF
			if key != 0xFF:
				break

		button_direction = action
		# Change the head position based on the button direction
		if button_direction == 1:
			self.snake_head[0] += 10
		elif button_direction == 0:
			self.snake_head[0] -= 10
		elif button_direction == 2:
			self.snake_head[1] += 10
		elif button_direction == 3:
			self.snake_head[1] -= 10

		# Increase Snake length on eating apple
		if self.snake_head == self.apple_position:
			self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
			self.snake_position.insert(0, list(self.snake_head))

		else:
			self.snake_position.insert(0, list(self.snake_head))
			self.snake_position.pop()

		# On collision kill the snake and print the score
		if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
			self.done = True
			self.gameOver = True

		self.total_reward = len(self.snake_position) - 3  # default length is 3
		self.reward = (self.total_reward - abs(self.prev_reward)) - MOVE_PENALTY
		self.prev_reward = self.reward

		if self.done:
			self.reward = -10
		info = {}

		head_x = self.snake_head[0]
		head_y = self.snake_head[1]

		snake_length = len(self.snake_position)
		apple_delta_x = self.apple_position[0] - head_x
		apple_delta_y = self.apple_position[1] - head_y

		# create observation:
		observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
		observation = np.array(observation).astype(np.float32)

		return observation, self.reward, self.done, info
	
	def render(self, mode='human', **kwargs):
		self.img = np.zeros((500, 500, 3), dtype='uint8')
		# Display Apple
		cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
					  (self.apple_position[0] + 10, self.apple_position[1] + 10), (0, 0, 255), 3)
		# Display Snake
		for position in self.snake_position:
			cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10), (0, 255, 0), 3)
		cv2.imshow('Snake', self.img)
		# Create a copy of the image to draw on
		img_copy = self.img.copy()

		# Draw the score on the top right corner
		font = cv2.FONT_HERSHEY_SIMPLEX
		score_text = "Score: {}".format(self.score)
		cv2.putText(img_copy, score_text, (350, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

		# Display the modified image
		cv2.imshow('Snake by Shehryar', img_copy)

		cv2.waitKey(1)


	def reset(self):
		self.img = np.zeros((500,500,3),dtype='uint8')
		# Initial Snake and Apple position
		self.snake_position = [[250,250],[240,250],[230,250]]
		self.apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
		self.score = 0
		self.prev_button_direction = 1
		self.button_direction = 1
		self.snake_head = [250,250]

		self.prev_reward = 0

		self.done = False

		head_x = self.snake_head[0]
		head_y = self.snake_head[1]

		snake_length = len(self.snake_position)
		apple_delta_x = self.apple_position[0] - head_x
		apple_delta_y = self.apple_position[1] - head_y

		self.prev_actions = deque(maxlen = SNAKE_LEN_GOAL)  # however long we aspire the snake to be
		for i in range(SNAKE_LEN_GOAL):
			self.prev_actions.append(-1) # to create history

		# create observation:
		observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
		observation = np.array(observation).astype(np.float32)

		return observation


def main():
	# Create an instance of the SnekEnv environment
	env = Snake()

	# Reset the environment
	observation = env.reset()

	while True:
		# Render the current state
		env.render()


		t_end = time.time() + 0.05
		while time.time() < t_end:
			key = cv2.waitKeyEx(1) & 0xFF
			if key != 0xFF:
				break

		# Map the key to an action
		if key == ord('a') and env.prev_button_direction != 1:
			env.button_direction = 0
		elif key == ord('d') and env.prev_button_direction != 0:
			env.button_direction = 1
		elif key == ord('w') and env.prev_button_direction != 2:
			env.button_direction = 3
		elif key == ord('s') and env.prev_button_direction != 3:
			env.button_direction = 2
		elif key == ord('q'):
			break
		else:
			env.button_direction = env.button_direction
		env.prev_button_direction = env.button_direction

		env.step(env.button_direction)

		# Check if the episode is done
		if env.done:
			env.img = np.zeros((500, 500, 3), dtype='uint8')
			# Show game over screen with score
			font = cv2.FONT_HERSHEY_SIMPLEX
			game_over_text = 'Game Over! Your Score is {}'.format(env.score)
			cv2.putText(env.img, game_over_text, (25, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
			cv2.imshow('a', env.img)
			cv2.waitKey(0)
			break

	# Close the environment
	env.close()
if __name__ == "__main__":
	main()

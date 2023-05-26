from snake import Snake
import cv2


env = Snake()
episodes = 50

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:#not done:
		env.render()
		random_action = env.action_space.sample()
		print("action", random_action)
		obs, reward, done, info = env.step(random_action)
		print('reward', reward)

		if done:
			env.render()
			cv2.waitKey(0)
			obs = env.reset()  # Reset the environment to start a new episode

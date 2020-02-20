import gym

env = gym.make("CartPole-v0")

num_trials = 10000

left_obss = []
for _ in range(num_trials):
    left_obss.append(env.reset())
    done = False
    while not done:
        obs, _, done, _ = env.step(0)
        left_obss.append(obs)

right_obss = []
for _ in range(num_trials):
    right_obss.append(env.reset())
    done = False
    while not done:
        obs, _, done, _ = env.step(1)
        right_obss.append(obs)

# 2nd dim is cart velocity
# 4th dim is pole velocity

left_cart_vels = [obs[1] for obs in left_obss]
print(f"left cart vels: min {min(left_cart_vels)}, max {max(left_cart_vels)}")
left_pole_vels = [obs[3] for obs in left_obss]
print(f"left pole vels: min {min(left_pole_vels)}, max {max(left_pole_vels)}")

right_cart_vels = [obs[1] for obs in right_obss]
print(
    f"right cart vels: min {min(right_cart_vels)}, max {max(right_cart_vels)}")
right_pole_vels = [obs[3] for obs in right_obss]
print(
    f"right pole vels: min {min(right_pole_vels)}, max {max(right_pole_vels)}")

import matplotlib.pyplot as plt
from env.gridworld import GridWorld


def visualize_episode(env: GridWorld, trajectory):
    xs, ys = [], []
    for (s, a, r) in trajectory:
        ys.append(s[0])
        xs.append(s[1])
    plt.figure(figsize=(6, 6))
    plt.imshow(env.grid, cmap="coolwarm", origin="upper")
    plt.plot(xs, ys, marker="o", color="black")
    plt.title("Trajectoire agent")
    plt.show()


if __name__ == "__main__":
    env = GridWorld()
    obs = env.reset()
    # Dummy trajectory for visualization
    trajectory = []
    for _ in range(10):
        next_obs, r, d, info = env.step(env.action_space - 1)
        trajectory.append((obs["agent"], env.action_space - 1, r))
        obs = next_obs
    visualize_episode(env, trajectory)

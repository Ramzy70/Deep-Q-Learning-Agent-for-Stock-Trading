# Deep Q-Learning (DQN) for Stock Trading: A NumPy Implementation

This repository contains a **from-scratch implementation** of a Deep Q-Network (DQN) agent designed to trade the S&P 500 index. Unlike standard implementations that rely on high-level libraries (PyTorch/TensorFlow), this project implements the neural network forward pass, backpropagation, and gradient descent entirely using **NumPy** and **Pandas**.

## 1. Project Overview

The objective of this project is to train a Reinforcement Learning agent to maximize realized profit by trading stock data fetched via `yfinance`. The agent interacts with a custom trading environment that simulates realistic constraints such as transaction costs and inventory limits.

### Core Features
* **Pure NumPy Backpropagation:** Manual implementation of the chain rule to compute gradients for the Neural Network weights.
* **Custom Environment:** A simulation built on historical daily price data (2015-2025).
* **Sparse Reward Signal:** The agent is rewarded based on *realized* profit (Closed P&L) rather than unrealized portfolio value, encouraging strategic exits.

## 2. Methodology

### 2.1 State-Action-Reward (SAR) Formulation

| Component | Definition | Description |
| :--- | :--- | :--- |
| **State ($S_t$)** | Vector $\in \mathbb{R}^{10}$ | A normalized vector of the last **10 daily percentage returns**, providing local price momentum context. |
| **Action ($A_t$)** | Discrete $\{0, 1, 2\}$ | **0:** Hold <br> **1:** Buy (Enter Long) <br> **2:** Sell (Close Position) |
| **Reward ($R_t$)** | Scalar $\in \mathbb{R}$ | The realized profit from a trade. $R_t = (P_{sell} \times (1 - \text{comm})) - P_{buy}$. <br> Rewards are 0 for Hold or Buy actions. |

### 2.2 Neural Network & Optimization
The Q-Function $Q(s, a; \theta)$ is approximated using a Multi-Layer Perceptron (MLP).

* **Architecture:** Input (10) $\rightarrow$ Hidden Layer $\rightarrow$ ReLU $\rightarrow$ Output (3).
* **Loss Function:** Mean Squared Error (MSE) between predicted Q-values and Target Q-values (derived from the Bellman Equation).
* **Gradient Calculation:**
    To handle the specific update for action $A_t$, a **masking technique** was employed. The error matrix is multiplied by a binary mask (1 at index $A_t$, 0 otherwise) to ensure gradients are only backpropagated through the nodes responsible for the chosen action.

## 3. Implementation Details

### Data Pipeline
* **Source:** Yahoo Finance (`yfinance`).
* **Asset:** S&P 500 (`^GSPC`) from Jan 1, 2015 to Jan 1, 2025.
* **Preprocessing:** Data is converted to daily returns and normalized. Shape consistency (N, 1) vs (N,) was enforced using explicit `.flatten()` calls to ensure compatibility between Pandas Series and NumPy linear algebra operations.

### Training Logic
The agent utilizes an $\epsilon$-greedy strategy for exploration.
1.  **Forward Pass:** Compute Q-values using current weights $W_1, W_2$.
2.  **Action Selection:** Select action with max Q-value (or random if exploring).
3.  **Experience Replay:** (Simulated via batch training on history).
4.  **Backpropagation:** Compute $\delta$ terms for output and hidden layers manually and update weights using Stochastic Gradient Descent (SGD).

## 4. Results

The performance of the agent is visualized through portfolio value tracking and buy/sell markers on the price chart.

* **Trade Execution:** The visualization demonstrates the agent's ability to identify local minima for buying and local maxima for selling.
* **Portfolio Growth:** The final portfolio value is compared against a "Buy and Hold" strategy benchmark.

*(See `task1-2.ipynb` for generated plots showing specific entry and exit points)*

## 5. Usage

### Prerequisites
* Python 3.x
* NumPy
* Pandas
* Matplotlib
* yfinance

### Running the Project
1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/DQN-Stock-Trading-NumPy.git](https://github.com/your-username/DQN-Stock-Trading-NumPy.git)
    ```
2.  Install dependencies:
    ```bash
    pip install numpy pandas matplotlib yfinance
    ```
3.  Execute the notebook:
    Run `task1-2.ipynb` to download the latest data, train the agent, and visualize the trading performance.

---
**Authors:** Ramzi Amira & Mahmoud Abo Shukr
*M2 Machine Vision and AI, Université Paris-Saclay*

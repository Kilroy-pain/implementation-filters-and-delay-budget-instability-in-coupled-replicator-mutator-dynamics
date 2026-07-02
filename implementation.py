import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class CoupledReplicatorMutatorDynamics:
    def __init__(self, num_methods, delay, implementation_rate, selection_rate, exploration_rate):
        self.num_methods = num_methods
        self.delay = delay
        self.implementation_rate = implementation_rate
        self.selection_rate = selection_rate
        self.exploration_rate = exploration_rate

        # Initialize portfolios
        self.intended_portfolio = torch.ones(num_methods) / num_methods
        self.deployed_portfolio = torch.ones(num_methods) / num_methods

        # Initialize delay buffer
        self.delay_buffer = [self.intended_portfolio.clone() for _ in range(delay)]

    def update_intended_portfolio(self, opponent_deployed):
        """Update the intended portfolio based on the opponent's deployed portfolio."""
        payoff = torch.matmul(self.deployed_portfolio, opponent_deployed.t())
        self.intended_portfolio = nn.functional.softmax(self.selection_rate * payoff + self.exploration_rate, dim=0)

    def update_deployed_portfolio(self):
        """Update the deployed portfolio based on the intended portfolio."""
        self.deployed_portfolio += self.implementation_rate * (self.intended_portfolio - self.deployed_portfolio)

    def step(self, opponent_deployed):
        """Perform one step of the dynamics."""
        # Update delay buffer
        self.delay_buffer.pop(0)
        self.delay_buffer.append(opponent_deployed.clone())

        # Use delayed observation
        delayed_opponent = self.delay_buffer[0]

        # Update intended portfolio
        self.update_intended_portfolio(delayed_opponent)

        # Update deployed portfolio
        self.update_deployed_portfolio()

        return self.deployed_portfolio.clone()

if __name__ == '__main__':
    # Parameters
    num_methods = 3
    delay = 5
    implementation_rate = 0.1
    selection_rate = 1.0
    exploration_rate = 0.01
    num_steps = 100

    # Initialize two populations
    pop1 = CoupledReplicatorMutatorDynamics(num_methods, delay, implementation_rate, selection_rate, exploration_rate)
    pop2 = CoupledReplicatorMutatorDynamics(num_methods, delay, implementation_rate, selection_rate, exploration_rate)

    # Random initial deployed portfolios
    pop1.deployed_portfolio = torch.tensor([0.4, 0.3, 0.3])
    pop2.deployed_portfolio = torch.tensor([0.3, 0.4, 0.3])

    # Simulate dynamics
    for step in range(num_steps):
        deployed1 = pop1.step(pop2.deployed_portfolio)
        deployed2 = pop2.step(pop1.deployed_portfolio)

        print(f"Step {step + 1}:")
        print(f"  Population 1 Deployed Portfolio: {deployed1.numpy()}")
        print(f"  Population 2 Deployed Portfolio: {deployed2.numpy()}")
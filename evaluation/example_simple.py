"""
Minimal example showing the simplicity of the evaluation interface.
"""

import sys
sys.path.append('..')
from evaluation import evaluate

# That's it! One line to evaluate any policy
class MyPolicy:
    def get_action(self, structured_inputs):
        # Your policy logic here
        return [0.5, -0.3]

# Evaluate
score = evaluate(MyPolicy())
print(f"Policy performance: {score:.1%}")

# For RL training integration:
from evaluation import Evaluator

# Create once
val_evaluator = Evaluator(num_episodes=5)

# Use during training
def training_loop():
    for epoch in range(100):
        # ... train your model ...
        
        # Validate
        if epoch % 10 == 0:
            score = val_evaluator.evaluate(current_policy)
            print(f"Epoch {epoch}: {score:.1%}")
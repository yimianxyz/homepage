"""
Policy Evaluator - Unified evaluation system for policy performance assessment

This module provides standardized evaluation across fixed scenarios:
- Canvas sizes: Mobile (480×320), Desktop (1920×1080), Large (2560×1440)
- Boid counts: 5, 20, 50
- Fixed parameters ensure consistent and comparable results

Primary metric: Overall catch rate
Secondary metrics: Per-scenario breakdowns with statistical measures
"""

import time
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
import sys
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.state_manager import StateManager
from simulation.random_state_generator import RandomStateGenerator
from config.constants import CONSTANTS

# FIXED EVALUATION PARAMETERS - DO NOT MODIFY
# These parameters are optimized for maximum speed while covering key scenarios
EPISODES_PER_SCENARIO = 10       # Episodes per scenario (fast evaluation)
MAX_STEPS_PER_EPISODE = 1500    # Maximum steps per episode
EVALUATION_SEED = 42            # Fixed seed for reproducible results

# Targeted test scenarios - optimized for speed and relevance
TEST_SCENARIOS = [
    # Mobile scenarios - test light and medium loads
    # ((480, 320), 5),     # Mobile + light load
    ((480, 320), 20),    # Mobile + medium load
    # Desktop scenarios - test heavy load
    #((1920, 1080), 40),  # Desktop + heavy load
]

@dataclass 
class ScenarioResult:
    """Results for a single scenario (canvas_size, boid_count combination)"""
    canvas_width: int
    canvas_height: int
    boid_count: int
    episodes_completed: int
    episodes_timed_out: int
    catch_rates: List[float]  # Catch rate per episode
    total_catches: List[int]  # Total catches per episode
    episode_lengths: List[int]  # Steps per episode
    
    # Computed statistics
    mean_catch_rate: float = 0.0
    std_catch_rate: float = 0.0
    min_catch_rate: float = 0.0
    max_catch_rate: float = 0.0
    mean_episode_length: float = 0.0
    mean_total_catches: float = 0.0
    
    def __post_init__(self):
        """Compute statistics after initialization"""
        if self.catch_rates:
            self.mean_catch_rate = statistics.mean(self.catch_rates)
            self.std_catch_rate = statistics.stdev(self.catch_rates) if len(self.catch_rates) > 1 else 0.0
            self.min_catch_rate = min(self.catch_rates)
            self.max_catch_rate = max(self.catch_rates)
        
        if self.episode_lengths:
            self.mean_episode_length = statistics.mean(self.episode_lengths)
            
        if self.total_catches:
            self.mean_total_catches = statistics.mean(self.total_catches)

@dataclass
class EvaluationResult:
    """Complete evaluation results with standardized parameters"""
    policy_name: str
    scenario_results: List[ScenarioResult]
    evaluation_time_seconds: float
    
    # Overall statistics
    overall_catch_rate: float = 0.0
    overall_std_catch_rate: float = 0.0
    total_episodes: int = 0
    successful_episodes: int = 0
    
    def __post_init__(self):
        """Compute overall statistics"""
        all_catch_rates = []
        total_episodes = 0
        successful_episodes = 0
        
        for scenario in self.scenario_results:
            all_catch_rates.extend(scenario.catch_rates)
            total_episodes += scenario.episodes_completed + scenario.episodes_timed_out
            successful_episodes += scenario.episodes_completed
        
        self.total_episodes = total_episodes
        self.successful_episodes = successful_episodes
        
        if all_catch_rates:
            self.overall_catch_rate = statistics.mean(all_catch_rates)
            self.overall_std_catch_rate = statistics.stdev(all_catch_rates) if len(all_catch_rates) > 1 else 0.0

class PolicyEvaluator:
    """Standardized policy evaluation system with fixed parameters"""
    
    def __init__(self):
        """
        Initialize policy evaluator with standardized parameters
        
        All evaluation parameters are fixed to ensure consistency and comparability
        """
        self.state_generator = RandomStateGenerator(seed=EVALUATION_SEED)
        
        # Detect CPU cores for multiprocessing
        self.num_cores = mp.cpu_count()
        
        print(f"PolicyEvaluator initialized with optimized parameters:")
        print(f"  Episodes per scenario: {EPISODES_PER_SCENARIO}")
        print(f"  Max steps per episode: {MAX_STEPS_PER_EPISODE}")
        print(f"  Test scenarios: {TEST_SCENARIOS}")
        print(f"  Evaluation seed: {EVALUATION_SEED}")
        print(f"  Total scenarios: {len(TEST_SCENARIOS)}")
        print(f"  Total episodes: {len(TEST_SCENARIOS) * EPISODES_PER_SCENARIO}")
        print(f"  CPU cores: {self.num_cores} (parallel execution)")
    
    def evaluate_policy(self, policy, policy_name: str = "Unknown Policy") -> EvaluationResult:
        """
        Evaluate a policy across all test scenarios
        
        Args:
            policy: Policy object with get_action(structured_inputs) method
            policy_name: Name for reporting
            
        Returns:
            Complete evaluation results
        """
        print(f"\n🎯 Evaluating Policy: {policy_name}")
        print("=" * 60)
        
        start_time = time.time()
        scenario_results = []
        
        total_scenarios = len(TEST_SCENARIOS)
        
        # Test predefined scenarios
        for current_scenario, ((canvas_width, canvas_height), boid_count) in enumerate(TEST_SCENARIOS, 1):
            print(f"\n📊 Scenario {current_scenario}/{total_scenarios}: "
                  f"{canvas_width}×{canvas_height}, {boid_count} boids")
            
            scenario_result = self._evaluate_scenario(
                policy, canvas_width, canvas_height, boid_count
            )
            scenario_results.append(scenario_result)
            
            # Progress summary
            print(f"  ✓ Completed: {scenario_result.episodes_completed}/{EPISODES_PER_SCENARIO} episodes")
            print(f"  📈 Catch rate: {scenario_result.mean_catch_rate:.3f} ± {scenario_result.std_catch_rate:.3f}")
        
        evaluation_time = time.time() - start_time
        
        # Create final result
        result = EvaluationResult(
            policy_name=policy_name,
            scenario_results=scenario_results,
            evaluation_time_seconds=evaluation_time
        )
        
        self._print_summary(result)
        return result
    
    def _evaluate_scenario(self, policy, canvas_width: int, canvas_height: int, boid_count: int) -> ScenarioResult:
        """Evaluate policy on a single scenario using parallel processing"""
        
        # Serialize policy for multiprocessing
        serialized_policy = pickle.dumps(policy)
        
        # Create episode tasks
        episode_tasks = []
        for episode in range(EPISODES_PER_SCENARIO):
            task = (serialized_policy, canvas_width, canvas_height, boid_count, episode)
            episode_tasks.append(task)
        
        # Run episodes in parallel
        catch_rates = []
        total_catches = []
        episode_lengths = []
        episodes_completed = 0
        episodes_timed_out = 0
        
        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            # Submit all episode tasks
            future_to_episode = {
                executor.submit(_run_episode_worker, task): episode 
                for episode, task in enumerate(episode_tasks)
            }
            
            # Collect results as they complete
            for future in future_to_episode:
                episode_num = future_to_episode[future]
                try:
                    result = future.result()
                    if result is not None:
                        catch_rate, catches, steps = result
                        catch_rates.append(catch_rate)
                        total_catches.append(catches)
                        episode_lengths.append(steps)
                        episodes_completed += 1
                    else:
                        episodes_timed_out += 1
                        
                except Exception as e:
                    print(f"    ⚠️  Episode {episode_num+1} failed: {e}")
                    episodes_timed_out += 1
        
        print(f"    ✓ Completed: {episodes_completed}/{EPISODES_PER_SCENARIO} episodes (parallel)")
        
        return ScenarioResult(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            boid_count=boid_count,
            episodes_completed=episodes_completed,
            episodes_timed_out=episodes_timed_out,
            catch_rates=catch_rates,
            total_catches=total_catches,
            episode_lengths=episode_lengths
        )
    
    def _print_summary(self, result: EvaluationResult):
        """Print evaluation summary"""
        print(f"\n" + "=" * 60)
        print(f"🎯 EVALUATION SUMMARY: {result.policy_name}")
        print("=" * 60)
        print(f"Overall Performance:")
        print(f"  📊 Overall Catch Rate: {result.overall_catch_rate:.3f} ± {result.overall_std_catch_rate:.3f}")
        print(f"  📈 Episodes Completed: {result.successful_episodes}/{result.total_episodes}")
        print(f"  ⏱️  Evaluation Time: {result.evaluation_time_seconds:.1f}s")
        
        print(f"\nDetailed Results by Scenario:")
        print(f"{'Canvas Size':<12} {'Boids':<6} {'Catch Rate':<12} {'Std Dev':<8} {'Episodes':<9}")
        print("-" * 60)
        
        for scenario in result.scenario_results:
            canvas_size = f"{scenario.canvas_width}×{scenario.canvas_height}"
            print(f"{canvas_size:<12} {scenario.boid_count:<6} "
                  f"{scenario.mean_catch_rate:<12.3f} {scenario.std_catch_rate:<8.3f} "
                  f"{scenario.episodes_completed:<9}")
        
        print("=" * 60)
    
    def save_results(self, result: EvaluationResult, output_path: str):
        """Save evaluation results to JSON file"""
        
        # Convert to serializable format
        data = {
            'policy_name': result.policy_name,
            'evaluation_time_seconds': result.evaluation_time_seconds,
            'overall_catch_rate': result.overall_catch_rate,
            'overall_std_catch_rate': result.overall_std_catch_rate,
            'total_episodes': result.total_episodes,
            'successful_episodes': result.successful_episodes,
            'evaluation_parameters': {
                'episodes_per_scenario': EPISODES_PER_SCENARIO,
                'max_steps_per_episode': MAX_STEPS_PER_EPISODE,
                'evaluation_seed': EVALUATION_SEED,
                'test_scenarios': TEST_SCENARIOS
            },
            'scenario_results': [asdict(scenario) for scenario in result.scenario_results],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Results saved to: {output_path}")

# === MULTIPROCESSING WORKER FUNCTIONS ===

def _run_episode_worker(task_data: Tuple) -> Tuple[float, int, int]:
    """
    Worker function for multiprocessing episode execution
    
    Args:
        task_data: Tuple of (serialized_policy, canvas_width, canvas_height, boid_count, episode_seed)
        
    Returns:
        Tuple of (catch_rate, total_catches, episode_length)
    """
    serialized_policy, canvas_width, canvas_height, boid_count, episode_seed = task_data
    
    # Deserialize policy
    policy = pickle.loads(serialized_policy)
    
    # Create episode-specific state generator with unique seed
    episode_state_generator = RandomStateGenerator(seed=EVALUATION_SEED + episode_seed)
    
    # Generate random initial state
    initial_state = episode_state_generator.generate_scattered_state(
        boid_count, canvas_width, canvas_height
    )
    initial_boid_count = len(initial_state['boids_states'])
    
    # Initialize state manager
    state_manager = StateManager()
    state_manager.init(initial_state, policy)
    
    total_catches = 0
    step = 0
    
    # Run episode
    while step < MAX_STEPS_PER_EPISODE:
        # Run simulation step
        result = state_manager.step()
        
        # Count catches this step
        if 'caught_boids' in result:
            total_catches += len(result['caught_boids'])
        
        # Check if all boids caught (early termination)
        if len(result['boids_states']) == 0:
            break
            
        step += 1
    
    # Calculate catch rate
    catch_rate = total_catches / initial_boid_count if initial_boid_count > 0 else 0.0
    
    return catch_rate, total_catches, step + 1
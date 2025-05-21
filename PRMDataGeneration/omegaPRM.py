"""
Parts of this code are referenced from the openr project.
For more details, please refer to https://github.com/openreasoner/openr.
"""

import heapq
import itertools
import json
import math

from typing import Any, Dict, List, Optional, Tuple

from Utils.utils import check_correctness, separate_steps
from Utils.ModelInterface import LLMInterface


class State:
    def __init__(self, solution_prefix: str, parent: Optional["State"] = None):
        # Solution prefix as a single string
        self.solution_prefix = solution_prefix

        # Reference to the parent state
        self.parent = parent

        # Visit count (number of times selected)
        self.N = 0

        # Total number of rollouts generated from this state
        self.total_rollouts = 0

        # Number of correct rollouts
        self.correct_rollouts = 0

        # Monte Carlo estimation (c/k)
        self.MC: Optional[float] = None

        # Q(s, r): estimated value for each rollout
        self.Q: Dict[str, float] = {}

        # Set of all rollouts from this state
        self.R: List[str] = []

        # List of incorrect rollouts
        self.incorrect_rollouts: List[str] = []

        # List of child states
        self.children: List["State"] = []

    def add_rollout(self, rollout: str):
        self.R.append(rollout)

    def add_incorrect_rollout(self, rollout: str):
        if rollout not in self.incorrect_rollouts:
            self.incorrect_rollouts.append(rollout)

    def get_full_solution(self) -> str:
        # Return the complete solution from the root to this state
        if self.parent:
            return self.parent.get_full_solution() + "\n\n" + self.solution_prefix
        else:
            return self.solution_prefix

    def get_new_text(self) -> str:
        """
        Return the new text added at this node compared to the parent.
        """
        if self.parent:
            parent_text = self.parent.solution_prefix
            new_text = self.solution_prefix[len(parent_text) :].strip()
            return new_text
        else:
            # Root node (the question)
            return self.solution_prefix.strip()

    def get_text_with_labels(self) -> Dict[str, Any]:
        """
        Return a nested dictionary where each node contains:
        - 'text': The new text at this node.
        - 'mc_value': The MC value at this node.
        - 'children': A list of child nodes with the same structure.
        """
        data = {
            "text": self.get_new_text(),
            "mc_value": self.MC,
            "children": [child.get_text_with_labels() for child in self.children],
        }
        return data


class SearchTree:
    def __init__(self):
        self.root: Optional[State] = None
        self.nodes: List[State] = []  # List of all states

    def add_state(self, state: State):
        self.nodes.append(state)


class CandidatePool:
    def __init__(self):
        self.heap: List[Tuple[float, int]] = []  # Heap of (-priority, unique_id)
        self.entry_finder: Dict[int, Tuple[float, int]] = (
            {}
        )  # Maps unique_id to (-priority, unique_id)
        self.counter = itertools.count()  # Unique sequence count
        self.id_to_rollout: Dict[int, Tuple[State, str]] = (
            {}
        )  # Maps unique_id to (state, rollout)
        self.latest_id_per_rollout: Dict[Tuple[int, str], int] = (
            {}
        )  # Maps (state_id, rollout) to unique_id

    def add_or_update(self, state: State, rollout: str, priority: float):
        """
        Add a new rollout or update the priority of an existing rollout.

        Parameters:
        - state (State): The state associated with the rollout.
        - rollout (str): The rollout string.
        - priority (float): The new priority score.
        """
        state_id = id(state)  # Unique identifier for the state object
        rollout_key = (state_id, rollout)

        # Check if the rollout already exists in the pool
        if rollout_key in self.latest_id_per_rollout:
            # Previous unique_id exists; it is now outdated
            old_unique_id = self.latest_id_per_rollout[rollout_key]
            # Mark the old entry as invalid by removing it from entry_finder
            if old_unique_id in self.entry_finder:
                del self.entry_finder[old_unique_id]
                del self.id_to_rollout[old_unique_id]

        # Assign a new unique_id for the updated rollout
        unique_id = next(self.counter)
        self.latest_id_per_rollout[rollout_key] = unique_id

        # Add the new entry to the heap and mappings
        heapq.heappush(
            self.heap, (-priority, unique_id)
        )  # Max-heap using negative priority
        self.entry_finder[unique_id] = (-priority, unique_id)
        self.id_to_rollout[unique_id] = (state, rollout)

    def pop(self) -> Tuple[Optional[State], Optional[str]]:
        """
        Pop the rollout with the highest priority.

        Returns:
        - Tuple[Optional[State], Optional[str]]: The state and rollout string, or (None, None) if empty.
        """
        while self.heap:
            neg_priority, unique_id = heapq.heappop(self.heap)
            # Check if this unique_id is still valid
            if unique_id in self.entry_finder:
                # Valid entry
                state, rollout = self.id_to_rollout.pop(unique_id)
                del self.entry_finder[unique_id]
                # Remove from latest_id_per_rollout
                state_id = id(state)
                rollout_key = (state_id, rollout)
                if self.latest_id_per_rollout.get(rollout_key) == unique_id:
                    del self.latest_id_per_rollout[rollout_key]
                return state, rollout
            # Else, outdated entry; skip
        return None, None

    def is_empty(self) -> bool:
        return not self.entry_finder


class OmegaPRM:
    def __init__(
        self,
        LM: LLMInterface,
        c_puct: float,
        alpha: float,
        beta: float,
        L: int,
        k: int,
        N: int,
        rollout_budget: int,
        save_data_tree: bool,
    ):
        """
        Initialize the OmegaPRM algorithm.

        Parameters:
        - LM (LanguageModel): The language model instance.
        - expected_answer (str): The expected answer for correctness checking.
        - c_puct (float): Exploration constant.
        - alpha (float): Weight for MC(s).
        - beta (float): Length penalty.
        - L (int): Maximum solution length.
        - k (int): Number of rollouts for Monte Carlo estimation.
        - N (int): Maximum search count.
        """
        self.LM = LM
        self.expected_answer = None
        self.c_puct = c_puct
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.k = k
        self.N = N
        self.rollout_budget = rollout_budget
        self.save_data_tree = save_data_tree

        self.T = SearchTree()
        self.C = CandidatePool()

        self.n = 0
        self.total_rollouts = 0

    def reset(self):
        """Reset internal state variables to prepare for a fresh run."""
        self.expected_answer = None
        self.T = SearchTree()  # Reset search tree
        self.C = CandidatePool()  # Reset candidate pool
        self.n = 0
        self.total_rollouts = 0
        self.collected_data = []  # Clear collected data

    def add_correct_rollout_to_tree(self, parent_state: State, rollout: str):
        """
        Add the correct rollout to the tree as a child of parent_state.
        """
        new_solution_prefix = (
            (parent_state.solution_prefix + "\n\n" + rollout).strip()
            if parent_state.solution_prefix
            else rollout
        )
        new_state = State(solution_prefix=new_solution_prefix, parent=parent_state)
        new_state.MC = 1.0  # Since the rollout is correct
        new_state.total_rollouts = 0
        new_state.correct_rollouts = 0
        self.T.add_state(new_state)
        parent_state.children.append(new_state)  # Add to parent's children

    def compute_Q(self, state: State, rollout: str) -> float:
        """
        Compute Q(s, r) = alpha^{1 - MC(s)} * beta^{len(r)/L}, where len(r) is based on word count.
        """
        # Count words in the rollout
        word_count = len(rollout.split())
        length_penalty = word_count / self.L
        Q_value = (self.alpha ** (1 - state.MC)) * (self.beta**length_penalty)
        return Q_value

    def compute_U(self, state: State) -> float:
        """
        Compute U(s) = c_puct * sqrt(sum_{s'} N(s')) / (1 + N(s))
        """
        N_total = sum(s.N for s in self.T.nodes)
        if N_total == 0:
            N_total = 1  # Prevent division by zero
        U_s = self.c_puct * (math.sqrt(N_total)) / (1 + state.N)
        return U_s

    def compute_selection_score(self, state: State, rollout: str) -> float:
        """
        Compute selection score: Score(s, r) = Q(s, r) + U(s)
        """
        Q_s_r = self.compute_Q(state, rollout)
        U_s = self.compute_U(state)
        score = Q_s_r + U_s
        return score

    def monte_carlo_estimation(self, state: State):
        """
        Perform Monte Carlo estimation for state by generating k rollouts
        and computing MC(s) = c / k, where c is the number of correct rollouts.
        """
        c = 0  # Correct rollouts count
        incorrect_rollouts, correct_rollouts = [], []
        batct_rollouts = self.LM.generate_rollout(state.solution_prefix, self.k)
        # Increment visit count of selected state
        state.N += 1
        for i, rollout in enumerate(batct_rollouts):
            # Increment number of total rollouts
            self.total_rollouts += 1

            # Generate rollout r_i
            state.add_rollout(rollout)

            # Evaluate correctness of final answer in rollout
            full_solution = (
                (state.solution_prefix + "\n\n" + rollout).strip()
                if state.solution_prefix
                else rollout
            )
            is_correct = check_correctness(full_solution, self.expected_answer)
            if is_correct:
                c += 1
                correct_rollouts.append(rollout)
            else:
                incorrect_rollouts.append(rollout)
                state.add_incorrect_rollout(rollout)  # Track incorrect rollouts

        # Update total rollouts and correct rollouts
        state.total_rollouts += self.k
        state.correct_rollouts += c
        state.MC = (
            state.correct_rollouts / state.total_rollouts
            if state.total_rollouts > 0
            else 0
        )
        if state.MC == 1.0:
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)
        elif state.MC == 0.0:
            # State is incorrect; no further action
            return
        else:
            # 0 < MC(s) < 1.0
            # Add correct rollouts to the tree
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)
            # Add incorrect rollouts to candidate pool with updated priorities
            for rollout in incorrect_rollouts:
                priority = self.compute_selection_score(state, rollout)
                self.C.add_or_update(state, rollout, priority)

    def selection_phase(self) -> Tuple[Optional[State], Optional[str]]:
        """
        Select (state, rollout) with the highest score from candidate pool C.
        """
        selected_state, selected_rollout = self.C.pop()
        return selected_state, selected_rollout

    def binary_search_incorrect_step(
        self, s_ast: State, steps: List[str], left: int, right: int
    ):
        """
        Recursively perform binary search to find all incorrect steps in the rollout.

        Parameters:
        - s_ast (State): The selected parent state.
        - steps (List[str]): The rollout steps as a list.
        - left (int): Left index of the current search interval.
        - right (int): Right index of the current search interval.
        """
        if left > right:
            return

        mid = (left + right) // 2
        new_steps = steps[left : mid + 1]
        if new_steps:
            prefix_solution = (
                s_ast.solution_prefix + "\n\n" + separate_steps(new_steps, mode="join")
            )
        else:
            prefix_solution = s_ast.solution_prefix
        # Create new state s_new
        s_new = State(solution_prefix=prefix_solution.strip(), parent=s_ast)
        self.T.add_state(s_new)
        s_ast.children.append(s_new)
        # Perform Monte Carlo estimation for s_new
        self.monte_carlo_estimation(s_new)
        if s_new.MC == 0:
            # Found incorrect step; continue searching in the left half to find earlier incorrect steps
            self.binary_search_incorrect_step(s_ast, steps, left, mid - 1)
        else:
            # Steps up to mid are correct; continue searching in the right half
            self.binary_search_incorrect_step(s_new, steps, mid + 1, right)

    def maintenance_phase(self, state: State):
        """
        Update statistics and candidate pool for all incorrect rollouts associated with the state.

        Parameters:
        - state (State): The state whose incorrect rollouts need to be updated.
        """

        # Iterate through all incorrect rollouts of the state
        for rollout in state.incorrect_rollouts:
            # Since we've already determined these rollouts are incorrect, no need to re-evaluate correctness

            priority = self.compute_selection_score(state, rollout)
            # Update the candidate pool with the new priority
            self.C.add_or_update(state, rollout, priority)
            # print(f"Updated Incorrect Rollout: '{rollout}' with new priority: {priority:.4f}")

    def run(self, question: str, answer: str):
        """
        Execute the OmegaPRM algorithm.

        Parameters:
        - question (str): The question to generate solutions for.

        Returns:
        - Collected data: List of dictionaries.
        """
        question = question.replace("\n\n", "\n")
        self.reset()
        # Initialization
        initial_state = State(solution_prefix=question, parent=None)
        self.expected_answer = answer
        self.T.root = initial_state
        self.T.add_state(initial_state)
        self.n = 0

        # Monte Carlo Estimation for initial_state
        self.monte_carlo_estimation(initial_state)
        # Main loop
        while (
            self.n < self.N
            and self.total_rollouts < self.rollout_budget
            and not self.C.is_empty()
        ):
            # Selection Phase
            selected_state, selected_rollout = self.selection_phase()
            if selected_state is None or selected_rollout is None:
                # print("No more candidates to explore. Terminating search.\n")
                break

            steps = separate_steps(selected_rollout, mode="split")
            # Perform binary search to find incorrect steps
            self.binary_search_incorrect_step(selected_state, steps, 0, len(steps) - 1)

            # Maintenance Phase
            self.maintenance_phase(selected_state)

            # Increment search count
            self.n += 1

        if self.save_data_tree:
            data = self.collect_tree_structure()
        else:
            data = self.collect_solution_prefixes()
        return data

    def collect_solution_prefixes(self) -> List[Dict[str, Any]]:
        """
        Collect all solution prefixes and their corresponding MC values from the search tree.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing solution prefixes and their MC values.
        """
        collected_data = []
        for node in self.T.nodes:
            solution_prefix = node.solution_prefix
            mc_value = node.MC
            collected_data.append(
                {"solution_prefix": solution_prefix, "mc_value": mc_value}
            )
        return collected_data

    def collect_tree_structure(self) -> Dict[str, Any]:
        """
        Collect the tree structure starting from the root.

        Returns:
            Dict[str, Any]: A nested dictionary representing the tree structure.
        """
        if self.T.root:
            tree_data = self.T.root.get_text_with_labels()
            return tree_data
        return {}

# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action) 
        newPos = successorGameState.getPacmanPosition() # (x,y)
        newFood = successorGameState.getFood() # FFFFF
        newGhostStates = successorGameState.getGhostStates() # game.AgentState object
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #[0]

        "*** YOUR CODE HERE ***"

        food_dist = 10000

        if not newFood.asList():
            food_dist = 0
        else:
            for food in newFood.asList():
                d = manhattanDistance(newPos, food)
                if d < food_dist:
                    food_dist = d
        

        ghost_dist = 10000

        timer = min(newScaredTimes)
        
        for ghost in newGhostStates:
                d = manhattanDistance(newPos, ghost.getPosition())
                if d < ghost_dist:
                    ghost_dist = d 


        return successorGameState.getScore() + 1/(food_dist + 1) - 3/(1+ ghost_dist) #+ timer

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You do not need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "* YOUR CODE HERE *"
        max_depth = self.depth * gameState.getNumAgents()
        best_score = float('-inf')
        actions = gameState.getLegalActions(self.index)
        action = None
        for a in actions:
            state = gameState.generateSuccessor(self.index,a)
            score = self.value(state, self.index, 1, max_depth)
            if score > best_score:
                best_score = score
                action = a
        return action 
        
    def value(self, gameState, agentIndex, depth, max_depth):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if depth == max_depth:
            return self.evaluationFunction(gameState)
        num_agents = gameState.getNumAgents()
        next = (agentIndex + 1) % num_agents
        if next == 0:
            return self.max_value(gameState, next, depth+1, max_depth)
        else:
            return self.min_value(gameState, next, depth+1, max_depth)
    def max_value(self, gameState, agentIndex, depth, max_depth):
        v = float('-inf')
        actions = gameState.getLegalActions(agentIndex)
        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            v = max(v, self.value(successor, agentIndex, depth, max_depth))
        return v
    def min_value(self, gameState, agentIndex, depth, max_depth):
        v = float('inf')
        actions = gameState.getLegalActions(agentIndex)
        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            v = min(v, self.value(successor, agentIndex, depth, max_depth))
        return v
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "* YOUR CODE HERE *"
        max_depth = self.depth * gameState.getNumAgents()
        best_score = float('-inf')
        actions = gameState.getLegalActions(self.index)
        action = None
        alpha = float('-inf')
        beta = float('inf')
        for a in actions:
            state = gameState.generateSuccessor(self.index,a)
            score = self.ab_value(state, self.index, 1, max_depth, best_score, beta)
            if score > best_score:
                best_score = score
                action = a
        return action 
        
    def ab_value(self, gameState, agentIndex, depth, max_depth, alpha, beta):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if depth == max_depth:
            return self.evaluationFunction(gameState)
        num_agents = gameState.getNumAgents()
        next = (agentIndex + 1) % num_agents
        if next == 0:
            return self.ab_max_value(gameState, next, depth+1, max_depth, alpha, beta)
        else:
            return self.ab_min_value(gameState, next, depth+1, max_depth, alpha, beta)
    def ab_max_value(self, gameState, agentIndex, depth, max_depth, alpha, beta):
        v = float('-inf')
        actions = gameState.getLegalActions(agentIndex)
        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            v = max(v, self.ab_value(successor, agentIndex, depth, max_depth, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v
    def ab_min_value(self, gameState, agentIndex, depth, max_depth, alpha, beta):
        v = float('inf')
        actions = gameState.getLegalActions(agentIndex)
        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            v = min(v, self.ab_value(successor, agentIndex, depth, max_depth, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "* YOUR CODE HERE *"
        max_depth = self.depth * gameState.getNumAgents()
        best_score = float('-inf')
        actions = gameState.getLegalActions(self.index)
        action = None
        for a in actions:
            state = gameState.generateSuccessor(self.index,a)
            score = self.exp_value(state, self.index, 1, max_depth)
            if score >= best_score:
                best_score = score
                action = a
        return action 
        
    def exp_value(self, gameState, agentIndex, depth, max_depth):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if depth == max_depth:
            return self.evaluationFunction(gameState)
        num_agents = gameState.getNumAgents()
        next = (agentIndex + 1) % num_agents
        if next == 0:
            return self.exp_max_value(gameState, next, depth+1, max_depth)
        else:
            return self.exp_min_value(gameState, next, depth+1, max_depth)
    def exp_max_value(self, gameState, agentIndex, depth, max_depth):
        v = float('-inf')
        actions = gameState.getLegalActions(agentIndex)
        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            v = max(v, self.exp_value(successor, agentIndex, depth, max_depth))
        return v
    def exp_min_value(self, gameState, agentIndex, depth, max_depth):
        v = []
        actions = gameState.getLegalActions(agentIndex)
        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            x = float(self.exp_value(successor, agentIndex, depth, max_depth))
            v.append(x)
        num_v = float(len(v))
        return sum(v)/num_v



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    power_pellets = currentGameState.getCapsules()

    food_dist = 10000

    if not newFood.asList():
        food_dist = 0
    else:
        for food in newFood.asList():
            d = manhattanDistance(newPos, food)
            if d < food_dist:
                food_dist = d
        

    ghost_dist = 10000
        
    for ghost in newGhostStates:
        d = manhattanDistance(newPos, ghost.getPosition())
        if ghost.scaredTimer != 0:
            ghost_dist = -10
        if d < ghost_dist:
            ghost_dist = d 
            
    

    power_dist = 10000

    for pellet in power_pellets:
        d = manhattanDistance(newPos, pellet)
        if d < power_dist:
            power_dist = d 


    return currentGameState.getScore() + 2/(food_dist + 1) - 3/(1+ ghost_dist) + 5/(1+power_dist) #+ timer 

# Abbreviation
better = betterEvaluationFunction



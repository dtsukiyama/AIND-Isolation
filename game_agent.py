import random
import numpy as np
import math
import itertools


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def manhattanDistance(own_location, opp_location, scale):
    y, x = (3, 3)
    own_y, own_x = own_location
    opp_y, opp_x = opp_location
    own_distance = abs(own_y - y) + abs(own_x - x)
    opp_distance = abs(opp_y - y) + abs(opp_x - x)
    return float(opp_distance - own_distance) / scale
    

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    if own_moves != opp_moves:
        return float(own_moves - opp_moves)
    
    else:
        return manhattanDistance(game.get_player_location(player), game.get_player_location(game.get_opponent(player)), 10)
    

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")


    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    open_spaces = len(game.get_blank_spaces())
    gameboard = game.width * game.height

    if own_moves != opp_moves:
        return float(own_moves - (gameboard / open_spaces) * opp_moves)

    else:
        return manhattanDistance(game.get_player_location(player), game.get_player_location(game.get_opponent(player)), 10)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(own_moves - opp_moves*2)



class IsolationPlayer:
    def __init__(self, search_depth=3, score_fn=custom_score, iterative=True, timeout=15.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.iterative = iterative

# this passes tests

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
    

    def get_move(self, game, time_left):
        
        self.time_left = time_left
        
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout      
        legal_moves = game.get_legal_moves()
        
        # always take center
        if game.move_count == 0:
            return (3, 3)

           
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move



    def minimax(self, game, depth):
        
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        legal_moves = game.get_legal_moves()

        best_score = float("-inf")
        best_move = (-1,-1)
        #best_move = legal_moves[0]

        
        for move in legal_moves:
            this_score = self.min_value(game.forecast_move(move),depth-1)
            if this_score >= best_score:
                best_score = this_score
                best_move = move
        
        return best_move
     
    def max_value(self, game, depth):
    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        legal_moves = game.get_legal_moves()
         
        if depth==0 or legal_moves == 0:
            return self.score(game,self)
        
        
        best_score = float("-inf")
         
        for move in legal_moves:
            v = self.min_value(game.forecast_move(move), depth -1)
            if v >= best_score:
                best_score = v
        return best_score

    def min_value(self, game, depth):
    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        legal_moves = game.get_legal_moves()
        
        if depth==0 or legal_moves == 0:
            return self.score(game,self)
        
        best_score = float("inf")
         
        for move in legal_moves:
            v = self.max_value(game.forecast_move(move), depth -1)
            if v <= best_score:
                best_score = v
        return best_score

# this passes tests


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        legal_moves = game.get_legal_moves()
        
        # always take center
        if game.move_count == 0:
            return (3, 3)

        
        if not legal_moves:
            return (-1, -1)

        best_move = (-1, -1)
        best_score = float("-inf")
        
        try:
            depth = 1
            while True:
                v = self.alphabeta(game, depth)
                if v == None:
                    return best_move
                best_move = v
                depth += 1
                
        except SearchTimeout:
            pass
            
        return best_move
        

    
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!

        
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            return (-1,-1)
        
        best_move = None
        best_score = float("-Inf")
        
        for move in legal_moves:
            this_score = self.ab_min_value(game.forecast_move(move), depth-1, alpha, beta)  
            if this_score >= best_score:
                best_score = this_score
                best_move = move
            alpha = max(alpha, this_score)

        return best_move


    def ab_min_value(self, game, depth, alpha, beta):
      
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        legal_moves = game.get_legal_moves()
        
        if depth == 0:
            return self.score(game, self)
        
        this_score = float("Inf")

        
        for move in legal_moves:
            this_score = min(this_score, self.ab_max_value(game.forecast_move(move), depth-1, alpha, beta))
                
            if this_score <= alpha:
                return this_score
                
            beta = min(beta, this_score)

        return this_score
       
    def ab_max_value(self, game, depth, alpha, beta):
      
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        legal_moves = game.get_legal_moves()
        
        if depth == 0:
            return self.score(game, self)
        
        this_score = float("-Inf")
        
        for move in legal_moves:
            this_score = max(this_score, self.ab_min_value(game.forecast_move(move), depth-1, alpha, beta))

            if this_score >= beta:
                return this_score
            alpha = max(alpha, this_score)

        return this_score

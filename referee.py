class Referee:
    def __init__(self, WIDTH=7, komi=9):
        self.width = WIDTH
        self.komi = komi
        self.current_player = 1  # 1 for black, -1 for white
        self.previous_pass = False  # Track if the previous move was a pass
        self.previous_states = []  # List to store up to 10 previous board states

    def init_board(self):
        return [[0 for _ in range(self.width)] for _ in range(self.width)]

    def judge(self, board_state, position):
        if position is None:
            # If a player passes, check for consecutive passes to end the game
            if self.previous_pass:
                winner = self.determine_winner(board_state)
                return None, winner
            else:
                self.previous_pass = True
                self.current_player *= -1
                return board_state, 0
        else:
            self.previous_pass = False  # Reset pass flag if a move is made

        row, col = position
        if board_state[row][col] != 0:
            # player break the rule
            winner = self.current_player * -1
            return None, winner

        # Place the new stone
        new_board_state = [row[:] for row in board_state]  # Create a deep copy of the current board
        new_board_state[row][col] = self.current_player

        # Check for opponent's stones to capture
        self.remove_captured_stones(new_board_state, row, col)

        # Check for suicide rule (if the new stone itself has no liberties after capturing opponent stones)
        if self.count_liberties(new_board_state, row, col) == 0:
            winner = self.current_player * -1
            return None, winner

        # Check for repetition of any of the last 10 board states
        if self.is_repeated_state(new_board_state):
            winner = self.current_player * -1
            return None, winner

        # Record the new state and remove old ones if necessary
        self.record_state(new_board_state)

        # Switch turns
        self.current_player *= -1
        return new_board_state, 0

    def get_valid(self, board_state):
        result = []

        for row in range(self.width):
            for col in range(self.width):
                if board_state[row][col] != 0:
                    continue

                # Place the new stone
                new_board_state = [row[:] for row in board_state]  # Create a deep copy of the current board
                new_board_state[row][col] = self.current_player

                # Check for opponent's stones to capture
                self.remove_captured_stones(new_board_state, row, col)

                # Check for suicide rule (if the new stone itself has no liberties after capturing opponent stones)
                if self.count_liberties(new_board_state, row, col) == 0:
                    continue

                # Check for repetition of any of the last 10 board states
                if self.is_repeated_state(new_board_state):
                    continue
                result.append((row, col))

        return result

    def record_state(self, new_board_state):
        """Record the new board state and ensure only the last 10 states are stored."""
        if len(self.previous_states) == 10:
            self.previous_states.pop(0)  # Remove the oldest state
        self.previous_states.append(new_board_state)

    def is_repeated_state(self, new_board_state):
        """Check if the new board state matches any of the last 10 states."""
        for past_state in self.previous_states:
            if new_board_state == past_state:
                return True
        return False

    def count_liberties(self, board_state, row, col):
        """Count the liberties of the stone at position (row, col)."""
        visited = set()
        return self._count_liberties_dfs(board_state, row, col, visited)

    def _count_liberties_dfs(self, board_state, row, col, visited):
        """Recursive DFS to count liberties of a group of connected stones."""
        if (row, col) in visited:
            return 0
        visited.add((row, col))

        liberties = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.width and 0 <= c < self.width:
                if board_state[r][c] == 0:  # Empty point (liberty)
                    liberties += 1
                elif board_state[r][c] == board_state[row][col]:
                    liberties += self._count_liberties_dfs(board_state, r, c, visited)
        return liberties

    def remove_captured_stones(self, board_state, row, col):
        """Check the opponent's surrounding stones and remove those with no liberties."""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        opponent = -self.current_player
        captured_stones = 0

        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.width and 0 <= c < self.width and board_state[r][c] == opponent:
                if self.count_liberties(board_state, r, c) == 0:
                    captured_stones += self._remove_group(board_state, r, c)

        return captured_stones

    def _remove_group(self, board_state, row, col):
        """Remove a group of stones recursively (after capture)."""
        to_remove = [(row, col)]
        opponent = board_state[row][col]
        captured = 0

        while to_remove:
            r, c = to_remove.pop()
            if board_state[r][c] == opponent:
                board_state[r][c] = 0  # Remove the stone
                captured += 1
                # Check all adjacent positions
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                for dr, dc in directions:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.width and 0 <= cc < self.width and board_state[rr][cc] == opponent:
                        to_remove.append((rr, cc))

        return captured

    def determine_winner(self, board_state):
        """Determine the winner by counting stones (basic scoring)."""
        black_score = sum(row.count(1) for row in board_state)
        white_score = sum(row.count(-1) for row in board_state)
        # Apply Komi to White's score
        white_score += self.komi
        return 1 if black_score > white_score else -1 if white_score > black_score else 0

import random, string, pygame
import time
import sys

class WordList:
    """
    Generates or asks for words to use in a Wordsearch game
    """
    def __init__(self, word_count=10):
        self.word_count = word_count
        self.words = []
        self.programming_words = [
            "PYTHON", "JAVA", "SWIFT", "KOTLIN", "RUST", "JAVASCRIPT", "HTML", "CSS",
            "VARIABLE", "FUNCTION", "LOOP", "ARRAY", "CLASS", "OBJECT", "IMPORT", "PACKAGE",
            "RECURSION", "DEBUG", "COMPILE", "SYNTAX", "EXCEPTION", "GIT", "MODULE", "SCRIPT",
            "DICTIONARY", "BOOLEAN", "INTEGER", "FLOAT", "DATABASE", "TERMINAL"
        ]

    def generate_words(self):
        valid_words = [w for w in self.family if len(w) <= 12]
        self.words = random.sample(valid_words, min(self.word_count, len(valid_words)))
        self.longest_word = max(self.words, key=len)
        self.longest_word_length = len(self.longest_word)
        self.grid_size = 12  # fixed size
        return self.words
    
    def create_word_list(self):
        pass

##########################################################################################
##########################################################################################

class WordSearch:
    def __init__(self, word_list_obj):
        self.word_list_obj = word_list_obj
        self.found_words = set()
        self.directions = { "H": (0, 1), "V": (1, 0), "D": (1, 1) }
        self.initialize_grid_state()

    def reset(self):
        self.found_words.clear()
        self.initialize_grid_state()

    def initialize_grid_state(self):
        self.word_list = self.word_list_obj.generate_words()
        self.grid_size = 12  # fixed size per your spec
        self.grid = self.create_empty_grid()
        self.placed_words = {}
        self.place_all_words()
        # Optional filler:
        self.fill_empty_spaces()

    def create_empty_grid(self):
        return [["_" for _ in range(self.grid_size)] for _ in range(self.grid_size)]
    def display_grid(self):
        for row in self.grid:
            print(" ".join(row))

    def place_all_words(self):
        for word in self.word_list:
            placed = False
            attempts = 0
            while not placed and attempts < 100:  # Prevents infinite loops
                direction = random.choice(list(self.directions.keys()))
                dx, dy = self.directions[direction]
                start_row = random.randint(0, self.grid_size - 1)
                start_col = random.randint(0, self.grid_size - 1)

                if self.can_place_word(word, start_row, start_col, dx, dy):
                    self.place_word(word, start_row, start_col, dx, dy)
                    self.placed_words[word] = [(start_row, start_col), direction]
                    placed = True
                attempts += 1

    def can_place_word(self, word, row, col, dx, dy):
        # Check boundaries and collisions
        for i in range(len(word)):
            new_row = row + dx * i
            new_col = col + dy * i

            if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
                return False

            current = self.grid[new_row][new_col]
            if current != "_" and current != word[i]:
                return False

        return True

    def place_word(self, word, row, col, dx, dy):
        for i in range(len(word)):
            new_row = row + dx * i
            new_col = col + dy * i
            self.grid[new_row][new_col] = word[i]

    def fill_empty_spaces(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c] == "_":
                    self.grid[r][c] = random.choice(string.ascii_uppercase)

###############################################################################################
###############################################################################################


class WordSearchGUI:
    def __init__(self, ws):
        self.ws = ws
        self.grid = ws.grid
        self.grid_size = ws.grid_size
        self.cell_size = 40
        self.window_size = self.cell_size * self.grid_size
        self.font_size = 24
        self.wordsearch = ws
        # self._request_restart = False  # ← Now it exists!
        self.restart_requested = False


        self.selected = []
        self.found_words = set()

        self.bg_color = (152, 106, 209)
        self.line_color = (70, 38, 110)
        self.text_color = (42, 16, 74)

        pygame.init()
        # self.screen = pygame.display.set_mode((self.window_size, self.window_size + 100))
        self.calculate_screen_dimensions()
        pygame.display.set_caption("Word Search Game")
        self.font = pygame.font.SysFont(None, self.font_size)
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()

    def reset_game(self):
        self.wordsearch.reset()
        self.grid = self.wordsearch.grid
        self.grid_size = self.wordsearch.grid_size
        self.found_words.clear()
        self.selected.clear()
        self.start_time = pygame.time.get_ticks()

        self.screen.fill((120, 0, 120))
        self.draw_grid()
        pygame.display.flip()
        
    def get_cell_from_mouse(self, pos):
        x, y = pos
        # y -= 100
        y -= self.word_list_height
        if y < 0:
            return None
        col = x // self.cell_size
        row = y // self.cell_size
        return (row, col)

    def get_path(self, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max(abs(dx), abs(dy))
        if length == 0:
            return []
        step_x = dx // length
        step_y = dy // length
        return [(start[0] + i * step_x, start[1] + i * step_y) for i in range(length + 1)]

    def get_letters_from_path(self, path):
        return "".join(self.grid[r][c] for r, c in path)

    def draw_word_list(self):
        padding = 10
        top_offset = 30
        max_row_width = self.window_size
        row_spacing = self.font_size + 5

        words = self.ws.word_list
        max_words_per_row = max_row_width // 100
        rows = [words[i:i + max_words_per_row] for i in range(0, len(words), max_words_per_row)]

        for row_idx, row_words in enumerate(rows):
            total_text_width = sum(self.font.size(word)[0] for word in row_words)
            spacing = (max_row_width - total_text_width) // (len(row_words) + 1)

            x = spacing
            y = padding + top_offset + row_idx * row_spacing

            for word in row_words:
                found = word in self.ws.found_words
                color = (180, 240, 240) if found else self.text_color
                text_surface = self.font.render(word, True, color)
                self.screen.blit(text_surface, (x, y))
                x += self.font.size(word)[0] + spacing

    def draw_grid(self):
        self.screen.fill(self.bg_color)
        self.draw_word_list()

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # rect = pygame.Rect(c * self.cell_size, r * self.cell_size + 100, self.cell_size, self.cell_size)
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size + self.word_list_height, self.cell_size, self.cell_size)
                if any((r, c) in path for path in self.found_words):
                    pygame.draw.rect(self.screen, (36, 189, 185), rect)
                elif (r, c) in self.selected:
                    pygame.draw.rect(self.screen, (220, 220, 255), rect)
                else:
                    pygame.draw.rect(self.screen, self.bg_color, rect)

                pygame.draw.rect(self.screen, self.line_color, rect, 2)
                letter = self.grid[r][c]
                text = self.font.render(letter, True, self.text_color)
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)

        # Draw timer and status bar after grid
        elapsed_ms = pygame.time.get_ticks() - self.start_time
        elapsed_sec = elapsed_ms // 1000
        timer_text = self.font.render(f"Time: {elapsed_sec}s", True, (255, 255, 255))
        text_width, text_height = timer_text.get_size()
        padding = 6
        bar_height = text_height + padding * 2
        bar_y = self.word_list_height + self.window_size

        # pygame.draw.rect(self.screen, (0, 0, 0), (0, bar_y, self.window_size, bar_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (0, bar_y, self.window_size, self.status_bar_height))
        self.screen.blit(timer_text, (self.window_size - text_width - padding, bar_y + padding))

    def flash_incorrect(self, path):
        for r, c in path:
            # rect = pygame.Rect(c * self.cell_size, r * self.cell_size + 100, self.cell_size, self.cell_size)
            rect = pygame.Rect(c * self.cell_size, r * self.cell_size + self.word_list_height, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), rect, 2)
        pygame.display.flip()
        time.sleep(0.2)

    def animate_found_word(self, word):
        for intensity in range(100, 256, 30):
            self.text_color = (0, intensity, 0)
            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(30)
        self.text_color = (42, 16, 74)

    def start_new_game(self):
        self.ws.reset()
        self.start_time = pygame.time.get_ticks()
        self.grid = self.ws.grid
        self.grid_size = self.ws.grid_size
        self.word_list = self.ws.word_list
        self.window_size = self.cell_size * self.grid_size
        # self.screen = pygame.display.set_mode((self.window_size, self.window_size + 100))
        self.calculate_screen_dimensions()
        self.found_words.clear()
        self.selected.clear()
        # self.run()
        # self._request_restart = True
        print(f"self.word_list = {self.word_list}")
        print(f"self.ws.placed_words = {self.ws.placed_words}")

    # def calculate_screen_dimensions(self):
    #     self.word_list_height = 100
    #     self.status_bar_height = 40
    #     self.screen_height = self.word_list_height + self.window_size + self.status_bar_height
    #     self.screen = pygame.display.set_mode((self.window_size, self.screen_height))

    def calculate_screen_dimensions(self):
        padding = 10
        top_offset = 30
        row_spacing = self.font_size + 5
        max_row_width = self.window_size
        max_words_per_row = max_row_width // 100

        words = self.ws.word_list
        num_rows = (len(words) + max_words_per_row - 1) // max_words_per_row

        self.word_list_height = padding + top_offset + num_rows * row_spacing
        self.status_bar_height = 40
        self.screen_height = self.word_list_height + self.window_size + self.status_bar_height
        self.screen = pygame.display.set_mode((self.window_size, self.screen_height))
        
    def show_victory_animation(self, elapsed_time):
        # Create a popup-style rectangle
        popup_width, popup_height = 400, 200
        popup_x = (self.window_size - popup_width) // 2
        popup_y = (self.window_size - popup_height) // 2
        popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)

        # Draw background
        pygame.draw.rect(self.screen, (50, 50, 50), popup_rect)
        pygame.draw.rect(self.screen, (200, 200, 200), popup_rect, 2)  # border

        # Render texts
        font = pygame.font.Font(None, 36)
        title_surface = font.render("Well done!", True, (255, 255, 255))
        time_surface = font.render(f"Completed in {elapsed_time:.1f} seconds", True, (255, 255, 255))

        # Button setup
        button_font = pygame.font.Font(None, 32)
        button_surface = button_font.render("Play Again", True, (0, 0, 0))
        button_rect = pygame.Rect(popup_x + 120, popup_y + 130, 160, 40)

        pygame.draw.rect(self.screen, (180, 180, 180), button_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), button_rect, 2)

        # Blit everything
        self.screen.blit(title_surface, title_surface.get_rect(center=(self.window_size // 2, popup_y + 40)))
        self.screen.blit(time_surface, time_surface.get_rect(center=(self.window_size // 2, popup_y + 80)))
        self.screen.blit(button_surface, button_surface.get_rect(center=button_rect.center))

        pygame.display.update()

        # Wait for click inside button before continuing
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if button_rect.collidepoint(event.pos):
                        # self.start_new_game()
                        self.restart_requested = True  # ← Signal to exit the loop and restart
                        waiting = False
    
    def animate_shuffle_transition(self):
        scramble_frames = 10  # Number of animation frames
        temp_grid = self.create_random_grid()

        for frame in range(scramble_frames):
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    temp_grid[r][c] = random.choice(string.ascii_uppercase)

            self.grid = temp_grid
            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(20)

        # Final grid swap
        self.grid = self.wordsearch.grid
        self.draw_grid()
        pygame.display.flip()

    def create_random_grid(self):
        return [[random.choice(string.ascii_uppercase) for _ in range(self.grid_size)] for _ in range(self.grid_size)]

    def run(self):
        running = True
        game_over = False
        finish_time = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                    cell = self.get_cell_from_mouse(event.pos)
                    if cell:
                        self.selected.append(cell)
                        if len(self.selected) == 2:
                            path = self.get_path(self.selected[0], self.selected[1])
                            guessed = self.get_letters_from_path(path)
                            if guessed in self.ws.word_list and guessed not in self.ws.found_words:
                                self.found_words.add(frozenset(path))
                                self.ws.found_words.add(guessed)
                            else:
                                self.flash_incorrect(path)
                            self.selected = []

            if self.restart_requested:
                print("Restart requested via GUI button")
                self.animate_shuffle_transition()  # ← Add this line here
                return

            # Victory detection
            if not game_over and len(self.ws.found_words) == len(self.ws.word_list):
                print("game over")
                game_over = True
                finish_time = pygame.time.get_ticks() // 1000

                # Highlight last found word
                self.draw_grid()
                pygame.display.flip()
                pygame.time.delay(300)  # Optional: gives a moment of visible confirmation

                self.show_victory_animation(finish_time)

            # Draw grid and update frame every tick
            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        

#######################################################################################################
#######################################################################################################
      
def launch_game():
    pygame.init()
    while True:
        wl = WordList(word_count=2)
        ws = WordSearch(wl)
        gui = WordSearchGUI(ws)
        gui.animate_shuffle_transition()
        gui.run()
        print("Restarting new session...")

        if not gui.restart_requested:
            break  # Exit loop and quit game

    pygame.quit()


# Only run if this file is the main script
if __name__ == "__main__":
    launch_game()


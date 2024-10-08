import tkinter as tk
from game import Board
from main import Player


class GomokuApp():
    def __init__(self, root):
        self.root = root
        self.root.title("围棋界面")
        self.board_size = 7  # 围棋棋盘大小
        self.canvas_width = 500  # 画布宽度
        self.canvas_height = 500  # 画布高度
        self.cell_size = self.canvas_width // self.board_size  # 每个格子的大小

        self.canvas = tk.Canvas(
            root, width=self.canvas_width, height=self.canvas_height, bg="yellow"
        )
        self.canvas.pack()

        self.ai = Player()
        self.board = Board()
        self.current_player = 1  # 当前玩家，1表示黑方，-1表示白方

        self.draw_board()
        self.bind_events()

    def draw_board(self):
        # 画棋盘线
        for i in range(self.board_size):
            x0 = (i+0.5) * self.cell_size
            y0 = 0.5 * self.cell_size
            x1 = x0
            y1 = (6 + 0.5) * self.cell_size
            self.canvas.create_line(x0, y0, x1, y1, fill="black")  # 画横线

            y0 = (i + 0.5) * self.cell_size
            x0 = 0.5 * self.cell_size
            y1 = y0
            x1 = (6 + 0.5) * self.cell_size
            self.canvas.create_line(x0, y0, x1, y1, fill="black")  # 画竖线

    def bind_events(self):
        # 绑定鼠标点击事件
        self.canvas.bind("<Button-1>", self.on_click)

    def paint_all(self):
        # 重新画
        self.canvas.delete("all")
        self.draw_board()
        # 绘制所有棋子
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board.board[i][j] == 0:
                    continue
                color = "black" if self.board.board[i][j] == 1 else "white"
                self.canvas.create_oval(
                    j * self.cell_size + self.cell_size // 6,
                    i * self.cell_size + self.cell_size // 6,
                    (j + 1) * self.cell_size - self.cell_size // 6,
                    (i + 1) * self.cell_size - self.cell_size // 6,
                    fill=color,
                    outline=color,
                )

    row: int
    col: int

    def refresh(self):
        if self.current_player == 1:
            return
        act = self.ai.run((self.row, self.col))
        if act is None:
            act = -1, -1
        self.board.place(*act)
        self.paint_all()
        self.current_player = -self.current_player
        self.root.after(100, self.refresh)

    def on_click(self, event):
        if self.current_player == -1:
            return
        # 计算鼠标点击的位置对应的棋盘坐标
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        # 检查点击的位置是否在棋盘范围内
        if 0 <= col < self.board_size and 0 <= row < self.board_size:
            # 检查该位置是否已经有棋子
            if self.board.board[row][col] != 0:
                return
            self.board.place(row, col)
            self.paint_all()
            self.row = row
            self.col = col
            self.current_player = -self.current_player
            self.root.after(100, self.refresh)


if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuApp(root)
    root.mainloop()

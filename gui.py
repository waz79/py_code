import FreeSimpleGUI as sg
import pandas as pd
import pyperclip
import math

class DataFrameViewer:
    def __init__(self, page_size=500, col_width=40, max_page_buttons=7):
        self.page_size = page_size
        self.col_width = col_width
        self.max_page_buttons = max_page_buttons
        self.df = None
        self.df_visible = None
        self.df_loaded = False
        self.current_page = 0
        self.max_pages = 1
        self.window = None
        self.total_records = 0

    def get_preview(self):
        return self.df.iloc[:self.page_size]

    def get_page_data(self):
        start = self.current_page * self.page_size
        end = start + self.page_size
        return self.df_visible.iloc[start:end]

    def update_table(self):
        self.window['TABLE'].update(values=self.get_page_data().values.tolist())

    def update_status(self):
        mode = "Full Dataset" if self.df_loaded else "Preview Only"
        total = len(self.df_visible)
        page = self.current_page + 1
        self.window['STATUS'].update(f"Page {page} of {self.max_pages} | Records: {total} | Mode: {mode}")

    def update_pagination(self):
        self.max_pages = max(1, math.ceil(len(self.df_visible) / self.page_size))
        spread = self.max_page_buttons // 2
        start = max(0, self.current_page - spread)
        end = min(self.max_pages, start + self.max_page_buttons)
        if end - start < self.max_page_buttons:
            start = max(0, end - self.max_page_buttons)

        for i in range(self.max_page_buttons):
            page_num = start + i
            key = f'PAGE_{i}'
            if page_num < self.max_pages:
                label = str(page_num + 1)
                style = {'button_color': ('white', 'blue')} if page_num == self.current_page else {'button_color': sg.theme_button_color()}
                self.window[key].update(label, visible=True, **style)
            else:
                self.window[key].update('', visible=False)

        self.window['<<'].update(disabled=self.current_page == 0)
        self.window['<'].update(disabled=self.current_page == 0)
        self.window['>>'].update(disabled=self.current_page >= self.max_pages - 1)
        self.window['>'].update(disabled=self.current_page >= self.max_pages - 1)

    def load_remaining(self):
        if not self.df_loaded:
            self.df_visible = self.df
            self.df_loaded = True
            self.current_page = 0
            self.update_table()
            self.update_status()
            self.update_pagination()
            self.window['LOAD_ALL'].update(visible=False)

    def show(self, df):
        self.df = df
        self.total_records = len(df)
        self.df_loaded = self.total_records <= self.page_size
        self.df_visible = df if self.df_loaded else self.get_preview()

        context_menu = ['Unused', ['Copy Account Number']]
        col_count = len(df.columns)
        default_col_widths = [self.col_width] * col_count

        pagination_row = [
            sg.Button('<<', key='<<', size=(4, 1)),
            sg.Button('<', key='<', size=(4, 1))
        ]
        for i in range(self.max_page_buttons):
            pagination_row.append(sg.Button('', key=f'PAGE_{i}', size=(3, 1), visible=False))
        pagination_row += [
            sg.Button('>', key='>', size=(4, 1)),
            sg.Button('>>', key='>>', size=(4, 1))
        ]

        layout = [
            [sg.Text("Search:"), sg.Input(key="SEARCH", enable_events=True, size=(40, 1))],
            [sg.Button("Get Remaining Records", key="LOAD_ALL", visible=not self.df_loaded)],
            [sg.Table(values=self.get_page_data().values.tolist(),
                      headings=df.columns.tolist(),
                      col_widths=default_col_widths,
                      auto_size_columns=True,
                      justification='left',
                      display_row_numbers=False,
                      num_rows=min(20, self.page_size),
                      key='TABLE',
                      enable_events=True,
                      vertical_scroll_only=False,
                      right_click_menu=context_menu)],
            [pagination_row],
            [sg.Text("", key='STATUS')],
            [sg.Button('Exit')]
        ]

        self.window = sg.Window("Paginated Data Viewer", layout, resizable=True, finalize=True)

        self.update_table()
        self.update_status()
        self.update_pagination()

        while True:
            event, values = self.window.read()
            if event in (sg.WIN_CLOSED, "Exit"):
                break

            elif event == "SEARCH":
                query = values["SEARCH"].lower()
                target = self.df if self.df_loaded else self.df_visible
                mask = target.apply(lambda row: row.astype(str).str.lower().str.contains(query).any(), axis=1)
                self.df_visible = target[mask]
                self.current_page = 0
                self.update_table()
                self.update_status()
                self.update_pagination()

            elif event == "LOAD_ALL":
                self.load_remaining()

            elif event == "Copy Account Number":
                selected = values["TABLE"]
                page_df = self.get_page_data()
                if selected and selected[0] < len(page_df):
                    pyperclip.copy(str(page_df.iloc[selected[0]]["Account Number"]))

            elif event in ['<<', '<', '>', '>>']:
                if event == '<<':
                    self.current_page = 0
                elif event == '<':
                    self.current_page = max(0, self.current_page - 1)
                elif event == '>':
                    self.current_page = min(self.max_pages - 1, self.current_page + 1)
                elif event == '>>':
                    self.current_page = self.max_pages - 1
                self.update_table()
                self.update_status()
                self.update_pagination()

            elif event.startswith('PAGE_'):
                label = self.window[event].get_text()
                if label.isdigit():
                    self.current_page = int(label) - 1
                    self.update_table()
                    self.update_status()
                    self.update_pagination()

        self.window.close()
